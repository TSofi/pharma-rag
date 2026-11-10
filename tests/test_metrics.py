"""Part 4 of observability: GET /api/metrics (app/metrics.py).

Counted from the same final log line every /api/ request writes (Part 1), so logs and numbers never disagree:
questions (per country / mode / language), cache hits / misses / skips, errors (4xx / 5xx, by code), issues,
p50 / p95 / max latency per stage (fresh answers; cache hits separately), LLM fallbacks and circuit-breaker trips,
top drugs per country. Protected: METRICS_TOKEN empty -> 404 (off), missing/wrong token -> 401.

No network: the vector search and the LLM are fakes.
"""
import json
import threading

import pytest
from fastapi.testclient import TestClient

from app import config, llm, main, metrics as metrics_module, rag
from app.metrics import metrics, percentile

TOKEN = "test-metrics-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
QUESTION = "What is metformin used for?"
CHUNK = {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is used for type 2 diabetes.",
         "url": "u", "score": 0.9, "effective_time": "20251101"}


class Clock:
    def __init__(self):
        self.t = 1_750_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(metrics_module, "now", clock)
    monkeypatch.setattr(config, "METRICS_TOKEN", TOKEN)
    monkeypatch.setattr(config, "METRICS_WINDOW", 1000)
    monkeypatch.setattr(config, "LOG_QUESTIONS", False)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_DAY", 0)
    monkeypatch.setattr(config, "ANSWER_CACHE_TTL_S", 900)
    metrics.reset()
    yield clock
    metrics.reset()


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def pipeline(monkeypatch):
    """The real rag.ask, fake vector search (reports embed/search times) and fake LLM."""
    def search(q, top_k, drugs=None, timings=None, **kw):
        if timings is not None:
            timings.update(embed_ms=40, vector_search_ms=60)
        return [dict(CHUNK)]
    monkeypatch.setattr(rag.store, "search", search)
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "RISK: NONE\nMetformin treats type 2 diabetes [1].")


def snapshot(client) -> dict:
    r = client.get("/api/metrics", headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, question=QUESTION, **extra):
    return client.post("/api/ask", json={"question": question, **extra})


# ================================================================== 1. the token
def test_without_a_configured_token_metrics_are_switched_off(client, monkeypatch):
    """Catches: metrics open to everyone on a server where nobody set a token."""
    monkeypatch.setattr(config, "METRICS_TOKEN", "")
    for headers in ({}, AUTH, {"Authorization": "Bearer "}):
        r = client.get("/api/metrics", headers=headers)
        assert r.status_code == 404 and r.json()["error"] == "NOT_FOUND"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": f"Basic {TOKEN}"},
                                     {"Authorization": f"Bearer {TOKEN}x"}, {"Authorization": TOKEN}])
def test_a_missing_or_wrong_token_is_401(client, headers):
    """Catches: error rates / provider outages readable by anyone (they show when the free keys are weakest)."""
    r = client.get("/api/metrics", headers=headers)
    assert r.status_code == 401 and r.json()["error"] == "UNAUTHORIZED"
    assert r.headers["www-authenticate"] == "Bearer"
    assert "questions" not in r.json()


def test_the_right_token_gets_the_metrics(client):
    body = client.get("/api/metrics", headers=AUTH).json()
    for section in ("started_at", "uptime_s", "questions", "cache", "errors", "issues", "latency_ms", "llm",
                    "top_drugs", "events"):
        assert section in body, section


def test_swagger_shows_the_authorize_button():
    """Catches: the token not being declared as security -> no Authorize button in /docs, only curl works."""
    schema = main.app.openapi()
    assert schema["components"]["securitySchemes"]["METRICS_TOKEN"]["scheme"] == "bearer"
    assert schema["paths"]["/api/metrics"]["get"]["security"] == [{"METRICS_TOKEN": []}]


# ================================================================== 2. questions, cache, errors, issues
def test_questions_are_counted_by_country_mode_and_language(client, pipeline):
    ask(client, QUESTION)
    ask(client, "Do czego służy metformina?", market="pl")
    ask(client, "Для чого метформін?", market="ua")
    q = snapshot(client)["questions"]
    assert q["total"] == 3 and q["answered"] == 3 and q["failed"] == 0
    assert q["by_market"] == {"us": 1, "pl": 1, "ua": 1}
    assert q["by_language"] == {"en": 1, "pl": 1, "uk": 1}
    assert sum(q["by_mode"].values()) == 3


def test_cache_hits_misses_and_skips_give_the_hit_rate(client, pipeline, monkeypatch):
    ask(client)                    # miss (saved)
    ask(client)                    # hit
    ask(client)                    # hit
    monkeypatch.setattr(rag, "ask", lambda *a, **k: {"answer": "x", "mode": "emergency", "found": False})
    ask(client, "I took 30 pills")  # skip: emergency
    c = snapshot(client)["cache"]
    assert c["hits"] == 2 and c["misses"] == 1 and c["skipped"] == {"emergency": 1}
    assert c["hit_rate"] == 0.5


def test_errors_are_counted_by_code_but_health_and_metrics_are_not_errors(client, monkeypatch):
    """Catches: the error count blown up by health pings during an outage (503 = a status report there) or by
    someone trying tokens on /api/metrics."""
    def raiser(exc):
        def f(*_a, **_k):
            raise exc
        return f
    monkeypatch.setattr(rag, "ask", raiser(llm.LLMUnavailable("down")))
    ask(client)                                                  # 503 LLM_UNAVAILABLE
    monkeypatch.setattr(rag, "ask", raiser(RuntimeError("qdrant down")))
    ask(client)                                                  # 502 UPSTREAM_ERROR
    ask(client, "")                                              # 400 (never reaches the endpoint)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(rag, "ask", lambda *a, **k: {"answer": "ok", "mode": "sourced"})
    ask(client, "another question?")                             # 200
    ask(client, "and one more?")                                 # 429
    from app import store
    monkeypatch.setattr(store, "get_health_client", raiser(RuntimeError("down")))
    from app import health
    health.reset()
    assert client.get("/api/health").status_code == 503          # not an error
    client.get("/api/metrics", headers={"Authorization": "Bearer wrong"})  # 401, not an error either
    e = snapshot(client)["errors"]
    assert e["by_code"] == {"LLM_UNAVAILABLE": 1, "UPSTREAM_ERROR": 1, "EMPTY": 1, "RATE_LIMITED": 1}
    assert e["total"] == 4 and e["server_5xx"] == 2 and e["client_4xx"] == 2
    assert e["by_path"] == {"/api/ask": 4}
    q = snapshot(client)["questions"]
    assert q["total"] == 5 and q["failed"] == 4 and q["answered"] == 1  # all 5 POST /api/ask were questions
    health.reset()


def test_issues_are_counted_by_name(client, monkeypatch):
    from app import observability

    def with_issue(*_a, **_k):
        observability.log_event("judge_unavailable")
        observability.log_event("translation_failed")
        return {"answer": "x", "mode": "sourced", "found": True}
    monkeypatch.setattr(rag, "ask", with_issue)
    ask(client)
    ask(client, "second question?")
    i = snapshot(client)["issues"]
    assert i["answers_with_issues"] == 2 and i["by_event"] == {"judge_unavailable": 2, "translation_failed": 2}


# ================================================================== 3. latency
def test_percentiles_use_the_nearest_rank_method():
    """Catches: a p95 that is an average, or off by one (p95 of 1..100 must be 95, not 96 or 95.05)."""
    values = list(range(1, 101))
    assert percentile(values, 50) == 50 and percentile(values, 95) == 95 and percentile(values, 100) == 100
    assert percentile([7], 95) == 7 and percentile([], 95) is None
    assert percentile([10, 20], 50) == 10 and percentile([10, 20], 95) == 20
    assert percentile(list(reversed(values)), 95) == 95  # order of arrival doesn't matter


def test_latency_per_stage_comes_from_fresh_answers_and_hits_are_separate(client, pipeline):
    """Catches: 2 ms cache hits mixed into the p95 of real answers (the p95 would look far better than reality)."""
    ask(client)
    ask(client)  # hit
    ask(client, "What is metformin used for in children?")
    lat = snapshot(client)["latency_ms"]
    assert lat["total"]["count"] == 2 and lat["cache_hit"]["count"] == 1
    assert lat["embed"] == {"count": 2, "p50": 40, "p95": 40, "max": 40}
    assert lat["vector_search"]["p95"] == 60 and lat["llm"]["count"] == 2
    assert "judge" not in lat or lat["judge"]["count"] <= 2


def test_only_the_last_n_answers_count(client, monkeypatch):
    """Catches: p95 frozen by a slow morning hours ago (and memory growing forever)."""
    monkeypatch.setattr(config, "METRICS_WINDOW", 3)
    metrics.reset()
    for ms in (9000, 9000, 9000, 100, 200, 300):
        metrics.record_request({"event": "request", "method": "POST", "path": "/api/ask", "status": 200,
                                "duration_ms": ms, "cache": "miss", "timings": {"llm_ms": ms}})
    lat = metrics.snapshot()["latency_ms"]
    assert lat["total"] == {"count": 3, "p50": 200, "p95": 300, "max": 300}
    assert metrics.snapshot()["questions"]["total"] == 6  # counters are not windowed


# ================================================================== 4. LLM, drugs, uptime
def test_llm_fallbacks_and_benched_providers_are_counted(client, monkeypatch):
    """Catches: Groq failing all day, hidden because Gemini always answered in the end."""
    for name, value in {"_cooldown_until": {}, "_last_probe": {}, "_groq_model": None, "_discovered": []}.items():
        monkeypatch.setattr(llm, name, value)
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["groq", "gemini"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", [])
    monkeypatch.setattr(config, "JUDGE_MODE", "off")

    def groq(*_a, **_k):
        raise RuntimeError("401 invalid api key")
    monkeypatch.setattr(llm, "_groq", groq)
    monkeypatch.setattr(llm, "_gemini", lambda m, s, u, timeout=None: "RISK: NONE\nMetformin treats diabetes [1].")
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(CHUNK)])
    assert ask(client).status_code == 200
    m = snapshot(client)["llm"]
    assert m["fallbacks"] == {"groq": 1} and m["benched"] == {"groq": 1}
    assert m["breaker_now"]["groq"] == "cooling_down"


def test_top_drugs_per_country(client, pipeline):
    ask(client, QUESTION)
    ask(client, "Can metformin cause nausea?")
    t = snapshot(client)["top_drugs"]
    assert t == {"us": {"metformin": 2}}


def test_the_drug_list_stops_growing(monkeypatch):
    """Catches: a bot asking about a million made-up names filling the 512 MB server."""
    monkeypatch.setattr(metrics_module, "MAX_DRUGS", 3)
    for name in ("a", "b", "c", "d", "e", "a"):
        metrics.record_request({"event": "request", "method": "POST", "path": "/api/ask", "status": 200,
                                "market": "pl", "detected_drugs": [name]})
    assert metrics.drugs["pl"] == {"a": 2, "b": 1, "c": 1, "(other)": 2}


def test_started_at_and_uptime_show_what_period_the_numbers_cover(client, fresh):
    fresh.t += 7260
    body = snapshot(client)
    assert body["uptime_s"] == 7260 and body["started_at"].startswith("2025-06-15T")


# ================================================================== 5. privacy, threads, own calls
def test_no_question_text_or_fingerprint_in_the_metrics(client, pipeline, capsys):
    """Catches: the metrics becoming a second place where health questions can be read or matched."""
    capsys.readouterr()
    ask(client, "I am pregnant and my name is Anna, can I take metformin?")
    h = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"request"' in x][0]["question_hash"]
    text = json.dumps(snapshot(client))
    assert "Anna" not in text and "pregnant" not in text and h not in text


def test_parallel_questions_are_all_counted(client, monkeypatch):
    """Catches: counters losing updates when many questions finish at the same moment (no lock)."""
    monkeypatch.setattr(rag, "ask", lambda *a, **k: {"answer": "ok", "mode": "sourced", "found": True})
    threads = [threading.Thread(target=ask, args=(client, f"question number {i}?")) for i in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert snapshot(client)["questions"]["total"] == 50


def test_reading_the_metrics_is_not_counted(client):
    for _ in range(3):
        snapshot(client)
    body = snapshot(client)
    assert body["questions"]["total"] == 0 and body["errors"]["total"] == 0


def test_every_snapshot_is_consistent_while_questions_are_recorded():
    """Catches: a missing lock. One question updates several counters one after another (total, then answered
    or failed, then the cache...). Without the lock, /api/metrics can read in between and show e.g.
    total = 101 but answered + failed = 100. (In CPython single += steps don't lose updates -- the GIL -- but
    the snapshot can still be taken halfway through a question.)"""
    stop = threading.Event()
    bad = []

    def record():
        while not stop.is_set():
            metrics.record_request({"event": "request", "method": "POST", "path": "/api/ask", "status": 200,
                                    "market": "pl", "cache": "miss", "duration_ms": 5})
    threads = [threading.Thread(target=record) for _ in range(4)]
    [t.start() for t in threads]
    try:
        for _ in range(1000):
            s = metrics.snapshot()
            q, c = s["questions"], s["cache"]
            if not (q["total"] == q["answered"] + q["failed"] == q["by_market"].get("pl", 0) == c["misses"]):
                bad.append((q["total"], q["answered"], c["misses"]))
    finally:
        stop.set()
        [t.join() for t in threads]
    assert bad == []

