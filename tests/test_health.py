"""Part 2 of observability: an honest /api/health (app/health.py).

Before: always {"status": "ok"}, even with Qdrant unreachable or every LLM benched by the circuit breaker.
Now:    ok (200) / degraded (200: the site answers, with less backup) / down (503: questions can't be answered).
Qdrant gets one cheap real call (max HEALTH_QDRANT_TIMEOUT_S); the LLMs are never called (quota), their
circuit-breaker state is read; the result is reused for HEALTH_CACHE_S (the website calls it on every page load).

No network: the Qdrant client is a fake, the clock is fake.
"""
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config, health, llm, main, rag, store

SECRET = "https://secret-cluster.qdrant.io key=abc123"
ALL_COLLECTIONS = [cfg["collection"] for cfg in config.MARKETS.values()]


class FakeQdrant:
    def __init__(self):
        self.calls = 0
        self.collections = list(ALL_COLLECTIONS)
        self.error: Exception | None = None
        self.release = threading.Event()
        self.hang = False
        self.delay = 0.0

    def get_collections(self):
        self.calls += 1
        time.sleep(self.delay)
        if self.hang:
            self.release.wait(10)
        if self.error:
            raise self.error
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])


class Clock:
    def __init__(self):
        self.t = 1_750_000_000.0

    def __call__(self):
        return self.t

    def wait(self, s):
        self.t += s


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(health, "now", c)
    return c


@pytest.fixture
def qdrant(monkeypatch):
    q = FakeQdrant()
    monkeypatch.setattr(store, "get_health_client", lambda: q)
    yield q
    q.release.set()  # never leave a worker thread hanging


@pytest.fixture(autouse=True)
def healthy(monkeypatch, clock, qdrant):
    """Everything set up and working: groq + gemini with keys, nothing benched, remote embeddings fine."""
    health.reset()
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["groq", "gemini"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-main")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", [])
    monkeypatch.setattr(llm, "_cooldown_until", {})
    monkeypatch.setattr(llm, "_discovered", None)
    monkeypatch.setattr(config, "HF_TOKEN", "t")
    monkeypatch.setattr(config, "REMOTE_EMBED_MODELS", [config.MULTI_EMBED_MODEL])
    monkeypatch.setattr(store, "remote_embed_last_failure", None)
    monkeypatch.setattr(store, "remote_embed_last_success", None)
    monkeypatch.setattr(config, "HEALTH_CACHE_S", 15)
    monkeypatch.setattr(config, "HEALTH_QDRANT_TIMEOUT_S", 3)
    monkeypatch.setattr(rag, "library", lambda market="us": [{"drug": "x"}])
    yield
    health.reset()


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def bench(*keys, seconds=300):
    for k in keys:
        llm._cooldown_until[k] = time.time() + seconds


# ================================================================== 1. the three states
def test_everything_working_is_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {name: c["status"] for name, c in body["checks"].items()} == {"qdrant": "ok", "llm": "ok", "embeddings": "ok"}
    assert body["qdrant_collections"] == len(ALL_COLLECTIONS) and body["checks"]["qdrant"]["missing_markets"] == []


def test_qdrant_unreachable_is_down_with_503(client, qdrant):
    """Catches: the old behaviour -- "status": "ok" while no question can be answered."""
    qdrant.error = ConnectionError("connection refused")
    r = client.get("/api/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "down" and body["checks"]["qdrant"]["status"] == "down"
    assert body["error"] == "UNAVAILABLE" and "qdrant" in body["detail"] and body["request_id"]  # the usual error shape
    assert r.headers["retry-after"] == "15"
    assert int(r.headers["content-length"]) == len(r.content)  # a wrong length = curl cuts the answer off


def test_hanging_qdrant_is_down_within_the_timeout(client, qdrant, monkeypatch):
    """Catches: health waiting 60 s (the normal client timeout) while Qdrant hangs -- the keepalive and any
    monitor would just time out instead of getting a clear "down"."""
    monkeypatch.setattr(config, "HEALTH_QDRANT_TIMEOUT_S", 0.3)
    qdrant.hang = True
    t0 = time.perf_counter()
    r = client.get("/api/health")
    assert time.perf_counter() - t0 < 2
    assert r.status_code == 503 and r.json()["checks"]["qdrant"]["reason"] == "timeout"


def test_a_missing_country_collection_is_degraded(client, qdrant):
    """Catches: one country silently answering "not found" to everything because its collection is gone."""
    qdrant.collections = [c for c in ALL_COLLECTIONS if c != config.MARKETS["fr"]["collection"]]
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"
    assert r.json()["checks"]["qdrant"]["missing_markets"] == ["fr"]


# ================================================================== 2. LLMs: circuit breaker, never a real call
def test_one_benched_provider_is_degraded(client):
    """Catches: Groq benched for 5 minutes (every answer now comes from the slower backup) reported as "ok"."""
    bench("groq")
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"
    assert r.json()["checks"]["llm"]["providers"] == {"groq": "cooling_down", "gemini:gemini-main": "ok"}


def test_every_provider_benched_is_down(client):
    bench("groq", "gemini:gemini-main")
    r = client.get("/api/health")
    assert r.status_code == 503 and r.json()["checks"]["llm"]["status"] == "down" and "llm" in r.json()["detail"]


def test_a_discovered_backup_model_still_counts(client, monkeypatch):
    """Catches: "down" while a Gemini backup model found earlier (model discovery) can still answer."""
    monkeypatch.setattr(llm, "_discovered", ["gemini-lite"])
    bench("groq", "gemini:gemini-main")
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["checks"]["llm"]["status"] == "degraded"


def test_no_llm_key_at_all_is_down_and_a_missing_backup_key_degraded(client, monkeypatch):
    """Catches: a deploy without API keys looking healthy; and a backup that can never help looking fine."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["checks"]["llm"]["providers"]["gemini:gemini-main"] == "no_key"
    assert r.json()["status"] == "degraded"
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    health.reset()
    assert client.get("/api/health").status_code == 503


def test_health_never_calls_an_llm(client, monkeypatch):
    """Catches: a health check that uses the free LLM quota on every page load."""
    def forbidden(*_a, **_k):
        raise AssertionError("health must not call an LLM")
    for name in ("generate", "_groq", "_gemini", "_anthropic", "_discover_backup_models"):
        monkeypatch.setattr(llm, name, forbidden)
    assert client.get("/api/health").status_code == 200


# ================================================================== 3. embeddings
def test_a_recent_embedding_api_failure_is_degraded_until_it_works_again(client, clock):
    """Catches: the HF embedding API failing (questions fall back to the slow local model, which may not fit
    into 512 MB) without any sign in health."""
    store.remote_embed_last_failure = clock()
    assert client.get("/api/health").json()["checks"]["embeddings"]["status"] == "degraded"
    clock.wait(health.EMBED_FAILURE_WINDOW_S + 1)                        # long ago -> ok again
    assert client.get("/api/health").json()["checks"]["embeddings"]["status"] == "ok"
    store.remote_embed_last_failure = clock()
    store.remote_embed_last_success = clock() + 1                         # it worked after the failure -> ok
    clock.wait(config.HEALTH_CACHE_S)
    assert client.get("/api/health").json()["checks"]["embeddings"]["status"] == "ok"


def test_embed_query_records_remote_failures_and_successes(monkeypatch):
    """Catches: health never learning about failures because embed_query doesn't record them."""
    monkeypatch.setattr(store, "get_query_embedder", lambda m: SimpleNamespace(embed=lambda t: [0.0]))

    def broken(texts, model):
        raise RuntimeError("402 credits used up")
    monkeypatch.setattr(store, "remote_embed", broken)
    store.embed_query("q", config.MULTI_EMBED_MODEL)
    assert store.remote_embed_last_failure is not None and store.remote_embed_last_success is None
    monkeypatch.setattr(store, "remote_embed", lambda texts, model: [[1.0]])
    store.embed_query("q", config.MULTI_EMBED_MODEL)
    assert store.remote_embed_last_success >= store.remote_embed_last_failure


# ================================================================== 4. cache, privacy, keepalive
def test_the_result_is_reused_for_15_s(client, qdrant, clock):
    """Catches: every page load calling Qdrant (the website pings health on every visit)."""
    client.get("/api/health")
    client.get("/api/health")
    assert qdrant.calls == 1
    clock.wait(14)
    client.get("/api/health")
    assert qdrant.calls == 1
    clock.wait(1)
    client.get("/api/health")
    assert qdrant.calls == 2


def test_many_page_loads_at_once_make_one_check(qdrant):
    """Catches: 20 visitors opening the page at the same moment = 20 Qdrant calls (while the first check is still
    waiting for Qdrant, the others must wait for its result, not start their own)."""
    qdrant.delay = 0.1  # a real network call takes time: that's when the visitors overlap
    threads = [threading.Thread(target=health.report) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert qdrant.calls == 1


def test_internal_details_go_to_the_log_not_to_the_public(client, qdrant, capsys):
    """Catches: the Qdrant URL / key or raw error texts shown to anyone who opens /api/health."""
    qdrant.error = RuntimeError(SECRET)
    capsys.readouterr()
    r = client.get("/api/health")
    assert "secret" not in r.text.lower() and "RuntimeError" not in r.text
    logs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    failed = next(x for x in logs if x["event"] == "health_check_failed")
    assert failed["check"] == "qdrant" and SECRET in failed["error"]
    assert next(x for x in logs if x["event"] == "request")["health"] == "down"


def test_the_keepalive_workflow_still_recognises_a_healthy_answer(client, qdrant):
    """Catches: the GitHub keepalive failing every day because the field it greps for changed. The pattern is
    read from the workflow file itself, so changing either side breaks this test."""
    workflow = (config.ROOT / ".github" / "workflows" / "keepalive.yml").read_text(encoding="utf-8")
    pattern = re.search(r"grep -q '([^']+)'", workflow).group(1)
    assert re.search(pattern, client.get("/api/health").text)
    health.reset()
    qdrant.error = ConnectionError("down")
    r = client.get("/api/health")
    assert r.status_code == 503 and not re.search(pattern, r.text)  # down: curl -f fails anyway, grep too
