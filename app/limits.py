"""Limits per visitor (group 7d): at most ASK_LIMIT_PER_MINUTE / ASK_LIMIT_PER_DAY questions (0 = off).

Why: every question costs up to 4-5 LLM calls on free Groq/Gemini keys. One script could use up the quota and
every real user would see "busy".

Who is "one visitor"? On Render the app is reached only through Render's proxy, which adds the real visitor
address at the END of the X-Forwarded-For note. A visitor can put anything at the START, so we take the LAST
entry. No note (local run) -> the direct caller. NOT yet confirmed on the live server: see tests/TESTING.md.

Counters live in memory: they reset when the server restarts (fine for this purpose).
"""
import ipaddress
import math
import threading
import time
from collections import OrderedDict, deque

from . import config, observability
from .errors import ApiError

now = time.monotonic  # replaced by a fake clock in the tests


# ---------- who is the visitor ----------

def client_ip(request) -> str:
    """The visitor's address: the last entry of the proxy's X-Forwarded-For note if it's a valid address,
    otherwise the direct caller."""
    notes = request.headers.getlist("x-forwarded-for")
    if notes:
        last = notes[-1].split(",")[-1].strip()
        try:
            return str(ipaddress.ip_address(last))
        except ValueError:
            pass  # garbage in the note: don't trust it
    return request.client.host if request.client else "unknown"


def visitor_key(ip: str) -> str:
    """What we count by. IPv6: the /64 block (one home connection owns a whole block, a bot could change the
    last part of its address endlessly). IPv4: the address itself."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 6:
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def mask(ip: str) -> str:
    """Address for the log with the last part hidden (203.0.113.x): enough to spot "always the same address"
    (= the proxy, not the visitor), without storing personal data."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "other"
    if addr.version == 4:
        return ".".join(str(addr).split(".")[:3]) + ".x"
    return ":".join(f"{int(h, 16):x}" for h in addr.exploded.split(":")[:3]) + ":x"


# ---------- the counter ----------

class RateLimit:
    """Sliding-window counter per visitor. `hit(key, rules)` with rules = [(limit, window_seconds), ...]:
    returns 0 and counts the request if every rule allows it, otherwise the seconds to wait (not counted,
    so pressing "retry" while blocked doesn't extend the wait)."""

    def __init__(self, max_keys: int = 50_000):
        self.max_keys = max_keys              # memory bound: the least recently seen visitors are forgotten first
        self._hits: OrderedDict[str, deque] = OrderedDict()
        self._logged: dict[str, float] = {}
        self._lock = threading.Lock()         # requests run in parallel threads

    def hit(self, key: str, rules: list[tuple[int, float]]) -> float:
        rules = [(limit, window) for limit, window in rules if limit > 0]  # 0 = this rule is off
        if not rules:
            return 0.0
        t = now()
        longest = max(window for _, window in rules)
        with self._lock:
            times = self._hits.pop(key, deque())
            while times and times[0] <= t - longest:  # forget what no rule looks at any more
                times.popleft()
            wait = 0.0
            for limit, window in rules:
                recent = [x for x in times if x > t - window]
                if len(recent) >= limit:
                    wait = max(wait, recent[-limit] + window - t)  # when the oldest counted one expires
            if not wait:
                times.append(t)
            self._hits[key] = times  # re-insert = "recently seen" (the dict is ordered: least recently seen first)
            # Forget visitors with nothing left in any window (they're at the front), and never keep more than
            # max_keys: a bot with many addresses can't fill the server's memory.
            while self._hits:
                first, first_times = next(iter(self._hits.items()))
                if len(self._hits) <= self.max_keys and first_times and first_times[-1] > t - longest:
                    break
                self._hits.popitem(last=False)
                self._logged.pop(first, None)
            return wait

    def first_block_this_minute(self, key: str) -> bool:
        """True at most once a minute per visitor, so a hammering bot doesn't fill the log."""
        t = now()
        with self._lock:
            if t - self._logged.get(key, -math.inf) < 60:
                return False
            self._logged[key] = t
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._logged.clear()

    def __len__(self) -> int:
        return len(self._hits)


ask_limiter = RateLimit()


def _human(seconds: int) -> str:
    if seconds < 120:
        return f"{seconds} s"
    if seconds < 2 * 3600:
        return f"{math.ceil(seconds / 60)} min"
    return f"{math.ceil(seconds / 3600)} h"


def limit_ask(request) -> None:
    """Raise 429 if this visitor asked too many questions. The text contains "429", which the UI already
    shows as "busy, try again" with a retry button."""
    ip = client_ip(request)
    key = visitor_key(ip)
    wait = ask_limiter.hit(key, [(config.ASK_LIMIT_PER_MINUTE, 60), (config.ASK_LIMIT_PER_DAY, 24 * 3600)])
    if not wait:
        return
    seconds = max(1, math.ceil(wait))
    if ask_limiter.first_block_this_minute(key):
        notes = request.headers.getlist("x-forwarded-for")
        observability.log_event("rate_limited", endpoint="/api/ask", visitor=mask(ip),
                                rule="per_day" if seconds > 60 else "per_minute", retry_after_s=seconds,
                                # how many addresses the proxy note had: helps to check the Render setup
                                proxy_note_entries=sum(len(n.split(",")) for n in notes))
    raise ApiError(429, "RATE_LIMITED", headers={"Retry-After": str(seconds)},
                   detail=f"RATE_LIMITED (429): too many questions from your connection. "
                          f"Please try again in {_human(seconds)}.")
