"""Spam checks for the "Report a problem" form (group 7e). No accounts or outside services needed.

Clear BOT signs -> main.py answers a fake {"ok": true} (no e-mail; the bot learns nothing):
  - the hidden field "website" is filled (honeypot),
  - the form was sent less than 3 s after it was opened, or without the page's timer (form_ms),
  - the same visitor sends the same text again within 24 h (also: a double click on "Send").
Too many -> 429: FEEDBACK_LIMIT_PER_HOUR per visitor (5), FEEDBACK_LIMIT_PER_DAY e-mails in total (90;
Resend's free plan allows 100 a day / 3,000 a month). 0 = that limit is off.
Only e-mails really sent count. Everything lives in memory (reset on a server restart) and is bounded.
"""
import hashlib
import json
import math
import threading
from collections import OrderedDict, deque

from . import config, limits
from .errors import ApiError

MIN_FORM_MS = 3_000          # nobody reads the form and writes a report in under 3 s
DAY = 24 * 3600
MAX_REMEMBERED = 20_000      # "same text" memory: at most this many (visitor, text) pairs

report_limiter = limits.RateLimit()           # per visitor per hour (forgets old visitors, bounded)
_lock = threading.Lock()
_sent: deque[float] = deque()                 # when each e-mail of the last 24 h was sent (daily total)
_seen: OrderedDict[str, float] = OrderedDict()  # fingerprint of (visitor, text) -> when it was sent


def _fingerprint(visitor: str, message: str) -> str:
    # Same words with other spaces / capitals = the same text. Only a hash is kept, never the text itself.
    text = " ".join(message.lower().split())
    return hashlib.sha256(f"{visitor}\n{text}".encode()).hexdigest()


def _forget_old(t: float) -> None:
    while _sent and _sent[0] <= t - DAY:
        _sent.popleft()
    while _seen and (next(iter(_seen.values())) <= t - DAY or len(_seen) > MAX_REMEMBERED):
        _seen.popitem(last=False)


def bot_reason(website: str, form_ms: int | None, visitor: str, message: str) -> str | None:
    """Why this report looks like a bot ("honeypot" / "too_fast" / "duplicate"), or None."""
    if website:
        return "honeypot"
    if form_ms is None or form_ms < MIN_FORM_MS:
        return "too_fast"
    with _lock:
        _forget_old(limits.now())
        if _fingerprint(visitor, message) in _seen:
            return "duplicate"
    return None


def log_drop(reason: str, ip: str) -> None:
    print(json.dumps({"event": "report_dropped", "reason": reason, "visitor": limits.mask(ip)}), flush=True)


def check_limits(visitor: str) -> None:
    """429 if this visitor sent too many reports this hour, or the day's total of e-mails is reached."""
    with _lock:
        t = limits.now()
        _forget_old(t)
        cap = config.FEEDBACK_LIMIT_PER_DAY
        if cap > 0 and len(_sent) >= cap:
            wait = max(1, math.ceil(_sent[-cap] + DAY - t))
            raise ApiError(429, "RATE_LIMITED", headers={"Retry-After": str(wait)},
                           detail="Reporting is busy today. Please try again tomorrow.")
    per_hour = config.FEEDBACK_LIMIT_PER_HOUR
    wait = report_limiter.hit(visitor, [(per_hour, 3600)])
    if wait:
        minutes = max(1, math.ceil(wait / 60))
        raise ApiError(429, "RATE_LIMITED", headers={"Retry-After": str(max(1, math.ceil(wait)))},
                       detail=f"Too many reports from your connection (at most {per_hour} an hour). "
                              f"Please try again in {minutes} min.")


def remember_sent(visitor: str, message: str) -> None:
    """Call AFTER the e-mail was really sent: counts towards the daily total and the "same text" check."""
    with _lock:
        t = limits.now()
        _sent.append(t)
        key = _fingerprint(visitor, message)
        _seen.pop(key, None)
        _seen[key] = t
        _forget_old(t)


def remembered() -> int:
    return len(_seen)


def reset() -> None:
    report_limiter.reset()
    with _lock:
        _sent.clear()
        _seen.clear()
