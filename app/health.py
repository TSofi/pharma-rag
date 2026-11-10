"""Honest /api/health: checks what a question really needs, instead of always saying "ok".

    status    HTTP  when
    ok        200   everything works
    degraded  200   the site still answers, but with less backup: an LLM provider is benched by the circuit
                    breaker or has no key, a country's Qdrant collection is missing, or the Hugging Face
                    embedding API failed in the last EMBED_FAILURE_WINDOW_S (questions use the local model)
    down      503   questions can't be answered: Qdrant unreachable / slower than HEALTH_QDRANT_TIMEOUT_S,
                    or no LLM can be used (all benched, or no API key at all)

- Qdrant gets a real (cheap) call: that also keeps the free cluster from being suspended (GitHub keepalive).
- LLMs are NOT called (that would use the free quota on every page load): the circuit breaker state is read.
- The result is reused for HEALTH_CACHE_S, because the website calls /api/health on every page load.
- Publicly only states (ok / cooling_down / missing...) are shown; error texts go to the log
  ("health_check_failed").
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timezone

from . import config, llm, observability, rag, store

EMBED_FAILURE_WINDOW_S = 300
_RANK = {"ok": 0, "degraded": 1, "down": 2}

now = time.time  # replaced by a fake clock in tests
_lock = threading.Lock()
_cached: tuple[float, dict] | None = None  # (when it was checked, the report)
# A hanging Qdrant call must not make health hang: it runs here and we stop waiting after the timeout.
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="health")


def reset() -> None:
    global _cached
    _cached = None


def _worst(*states: str) -> str:
    return max(states, key=_RANK.__getitem__)


# ---------------------------------------------------------------- the three checks
def check_qdrant() -> dict:
    t0 = time.perf_counter()
    try:
        future = _pool.submit(lambda: store.get_health_client().get_collections().collections)
        names = {c.name for c in future.result(timeout=config.HEALTH_QDRANT_TIMEOUT_S)}
    except FutureTimeout:
        observability.log_event("health_check_failed", check="qdrant",
                                error=f"no answer within {config.HEALTH_QDRANT_TIMEOUT_S:g} s")
        return {"status": "down", "reason": "timeout", "collections": None}
    except Exception as e:  # noqa: BLE001
        observability.log_event("health_check_failed", check="qdrant", error=f"{type(e).__name__}: {e}")
        return {"status": "down", "reason": "unreachable", "collections": None}
    missing = sorted(m for m, cfg in config.MARKETS.items() if cfg["collection"] not in names)
    if missing:
        observability.log_event("health_check_failed", check="qdrant", error=f"missing collections for {missing}")
    return {"status": "degraded" if missing else "ok", "collections": len(names), "missing_markets": missing,
            "latency_ms": round((time.perf_counter() - t0) * 1000)}


def check_llm() -> dict:
    providers = {k: v["status"] for k, v in llm.breaker_state().items()}
    usable = [k for k, s in providers.items() if s == "ok"]
    if not usable:
        status = "down"
        observability.log_event("health_check_failed", check="llm",
                                error=f"no usable LLM provider: {providers or 'none configured'}")
    else:
        status = "degraded" if len(usable) < len(providers) else "ok"
    return {"status": status, "providers": providers}


def check_embeddings() -> dict:
    remote = config.MULTI_EMBED_MODEL in config.REMOTE_EMBED_MODELS and bool(config.HF_TOKEN)
    if not remote:
        return {"status": "ok", "mode": "local"}
    failed, worked = store.remote_embed_last_failure, store.remote_embed_last_success
    recent_failure = (failed is not None and now() - failed < EMBED_FAILURE_WINDOW_S
                      and (worked is None or worked < failed))
    return {"status": "degraded" if recent_failure else "ok", "mode": "remote",
            **({"reason": "recent_failure"} if recent_failure else {})}


# ---------------------------------------------------------------- the report
def _library_sizes() -> tuple[int | None, dict]:
    """The old fields "drugs" / "markets" (product lists, read from local files, no network)."""
    try:
        return len(rag.library()), {m: len(rag.library(m)) for m in config.MARKETS}
    except Exception as e:  # noqa: BLE001 -- a broken product list must not hide the real status
        observability.log_event("health_check_failed", check="library", error=f"{type(e).__name__}: {e}")
        return None, {}


def build_report() -> dict:
    checks = {"qdrant": check_qdrant(), "llm": check_llm(), "embeddings": check_embeddings()}
    drugs, markets = _library_sizes()
    return {
        "status": _worst(*(c["status"] for c in checks.values())),
        "checks": checks,
        "checked_at": datetime.fromtimestamp(now(), timezone.utc).isoformat(timespec="seconds"),
        # Fields kept from the old /api/health (the GitHub keepalive looks for "qdrant_collections":<number>).
        "qdrant_collections": checks["qdrant"]["collections"],
        "llm": config.LLM_PROVIDER,
        "vector_db": "qdrant-cloud" if config.QDRANT_URL else "qdrant-local",
        "drugs": drugs,
        "markets": markets,
    }


def report() -> dict:
    """The health report, checked at most once per HEALTH_CACHE_S (the lock: many page loads at the same moment
    wait for ONE check instead of each calling Qdrant)."""
    global _cached
    with _lock:
        if _cached is None or now() - _cached[0] >= config.HEALTH_CACHE_S:
            _cached = (now(), build_report())
        return _cached[1]
