"""Metrics for GET /api/metrics: counted from the SAME final log line every /api/ request writes
(app/observability.py), so the logs and the numbers can never disagree.

- questions: how many, per country / answer mode / question language
- cache: hits, misses (answered fresh and saved), skipped (and why), hit rate
- errors: all /api/ endpoints except /api/health (a 503 there is a status report, not a failed request) and
  /api/metrics itself; split into client (4xx) and server (5xx) errors, and by error code
- issues: answers that were worse than they should be (observability.ISSUE_EVENTS), by name
- latency: p50 / p95 / max over the last METRICS_WINDOW fresh answers, in total and per stage
  (embed, vector_search, llm, judge...). Cache hits are a separate group ("cache_hit"): mixing 2 ms hits
  with 3 s answers would make p95 look better than what a fresh question really costs.
- llm: provider fallbacks (llm_failed) and circuit-breaker trips (llm_benched) per provider
- top_drugs: the products people ask about most, per country (counts only; at most MAX_DRUGS names per country)
- events: how often every log event happened

RAM only: everything starts again at 0 after a restart (started_at / uptime_s say which period it covers).
Never contains question texts or fingerprints.

p95 ("95th percentile"): sort the times; p95 is the value 95 % of the questions were faster than, the
"slow but not the single worst" experience. Nearest-rank method: the ceil(0.95 * n)-th smallest value.
"""
import math
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

from . import config

now = time.time  # replaced by a fake clock in tests
MAX_DRUGS = 1000           # distinct product names remembered per country (the rest counts as "other")
TOP_DRUGS_SHOWN = 20
NOT_COUNTED_AS_ERRORS = {"/api/health", "/api/metrics"}


def percentile(values, p: float) -> float | None:
    """Nearest-rank percentile: the ceil(p/100 * n)-th smallest value (p=95 of 1..100 -> 95)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summary(values) -> dict:
    values = list(values)
    return {"count": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "max": max(values) if values else None}


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()  # requests are recorded from many threads at once
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started_at = now()
            self.questions = Counter()          # total / answered / failed
            self.by_market = Counter()
            self.by_mode = Counter()
            self.by_language = Counter()
            self.cache = Counter()              # hit / miss / skip / off
            self.cache_skipped = Counter()      # skip reason -> n
            self.errors_by_code = Counter()
            self.errors_by_path = Counter()
            self.errors = Counter()             # total / client_4xx / server_5xx
            self.issues = Counter()             # issue event -> n
            self.answers_with_issues = 0
            self.latency: dict[str, deque] = {}
            self.events = Counter()
            self.llm_fallbacks = Counter()      # provider that failed (another one was tried)
            self.llm_benched = Counter()        # provider benched by the circuit breaker
            self.drugs: dict[str, Counter] = {}

    # ------------------------------------------------------------ recording
    def _latency(self, name: str, value) -> None:
        if isinstance(value, (int, float)):
            self.latency.setdefault(name, deque(maxlen=config.METRICS_WINDOW)).append(value)

    def record_request(self, line: dict) -> None:
        """One final "request" log line (observability.RequestLog)."""
        path, status = line.get("path", ""), line.get("status", 0)
        with self._lock:
            if status >= 400 and path not in NOT_COUNTED_AS_ERRORS:
                self.errors["total"] += 1
                self.errors["server_5xx" if status >= 500 else "client_4xx"] += 1
                self.errors_by_code[line.get("error") or str(status)] += 1
                self.errors_by_path[path] += 1
            if path == "/api/ask" and line.get("method") == "POST":
                self._record_question(line, status)

    def _record_question(self, line: dict, status: int) -> None:
        self.questions["total"] += 1
        self.questions["answered" if status < 400 else "failed"] += 1
        if line.get("market"):
            self.by_market[line["market"]] += 1
        if line.get("cache"):
            self.cache[line["cache"]] += 1
            if line["cache"] == "skip":
                self.cache_skipped[line.get("cache_skip_reason") or "unknown"] += 1
        issues = line.get("issues") or []
        self.issues.update(issues)
        self.answers_with_issues += bool(issues)
        if status >= 400:
            return
        if line.get("mode"):
            self.by_mode[line["mode"]] += 1
        if line.get("language"):
            self.by_language[line["language"]] += 1
        if line.get("cache") == "hit":
            self._latency("cache_hit", line.get("duration_ms"))
        else:
            self._latency("total", line.get("duration_ms"))
            for stage, ms in (line.get("timings") or {}).items():
                if stage.endswith("_ms") and stage != "total_ms":
                    self._latency(stage[:-3], ms)
        counter = self.drugs.setdefault(line.get("market") or "?", Counter())
        for drug in line.get("detected_drugs") or []:
            name = str(drug).strip().lower()
            if name in counter or len(counter) < MAX_DRUGS:
                counter[name] += 1
            else:
                counter["(other)"] += 1

    def record_event(self, event: str, fields: dict) -> None:
        """Every other log line (observability.log_event)."""
        with self._lock:
            self.events[event] += 1
            if event == "llm_failed":
                self.llm_fallbacks[fields.get("provider") or "?"] += 1
            elif event == "llm_benched":
                self.llm_benched[fields.get("provider") or "?"] += 1

    # ------------------------------------------------------------ the report
    def snapshot(self) -> dict:
        from . import llm  # here, not at the top: llm -> observability -> metrics would be a circle
        with self._lock:
            lookups = self.cache["hit"] + self.cache["miss"] + self.cache["skip"]
            order = ["total", "embed", "vector_search", "translate", "llm", "judge", "refusal", "cache_hit"]
            names = sorted(self.latency, key=lambda n: (order.index(n) if n in order else len(order), n))
            return {
                "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(timespec="seconds"),
                "uptime_s": round(now() - self.started_at),
                "window": config.METRICS_WINDOW,
                "questions": {**{k: self.questions[k] for k in ("total", "answered", "failed")},
                              "by_market": dict(self.by_market), "by_mode": dict(self.by_mode),
                              "by_language": dict(self.by_language)},
                "cache": {"hits": self.cache["hit"], "misses": self.cache["miss"],
                          "skipped": dict(self.cache_skipped), "off": self.cache["off"],
                          "hit_rate": round(self.cache["hit"] / lookups, 3) if lookups else None},
                "errors": {**{k: self.errors[k] for k in ("total", "client_4xx", "server_5xx")},
                           "by_code": dict(self.errors_by_code), "by_path": dict(self.errors_by_path)},
                "issues": {"answers_with_issues": self.answers_with_issues, "by_event": dict(self.issues)},
                "latency_ms": {n: summary(self.latency[n]) for n in names},
                "llm": {"fallbacks": dict(self.llm_fallbacks), "benched": dict(self.llm_benched),
                        "breaker_now": {k: v["status"] for k, v in llm.breaker_state().items()}},
                "top_drugs": {m: dict(c.most_common(TOP_DRUGS_SHOWN)) for m, c in sorted(self.drugs.items()) if c},
                "events": dict(self.events),
            }


metrics = Metrics()
