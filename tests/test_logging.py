"""Part 1 of observability: structured logs + request id (app/observability.py).

Every request gets an id: in the X-Request-ID header, in the JSON answer (ask + every error) and in every log
line it caused. Each /api/ call writes ONE final "request" line (JSON) with status, duration and, for a
question, market / languages / mode / model / timings per stage. Extra events ("llm_failed",
"judge_unavailable"...) carry the same id, so one question can be followed through the whole log.
Privacy: LOG_QUESTIONS=false (default) -> the question text is never logged, only its length + a keyed hash.

No network: the vector search and the LLM are fakes.
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import config, llm, main, observability, rag

QUESTION = "What is metformin used for?"
CHUNK = {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is used for type 2 diabetes.",
         "url": "u", "score": 0.9, "effective_time": "20251101"}


@pytest.fixture(autouse=True)
def private_by_default(monkeypatch):
    """Tests must not depend on a developer's .env: default privacy setting, fixed hash key."""
    monkeypatch.setattr(config, "LOG_QUESTIONS", False)
    monkeypatch.setattr(config, "LOG_HASH_SECRET", "test-secret")


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def pipeline(monkeypatch):
    """The real rag.ask with a fake vector search (it reports embed/search timings like the real one) and a
    fake LLM. `replies[system_prompt_start]` can change what the LLM says."""
    def search(q, top_k, drugs=None, timings=None, **kw):
        if timings is not None:
            timings.update(embed_ms=5, vector_search_ms=7)
        return [dict(CHUNK)]
    monkeypatch.setattr(rag.store, "search", search)
    monkeypatch.setattr(rag.llm, "generate", lambda system, user: "RISK: NONE\nMetformin treats type 2 diabetes [1].")


def lines(capsys) -> list[dict]:
    """Every line the app printed, parsed. Fails if any line is not JSON (= not a structured log)."""
    out = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    parsed = []
    for line in out:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pytest.fail(f"log line is not JSON: {line[:200]}")
    return parsed


def request_lines(logs: list[dict]) -> list[dict]:
    return [x for x in logs if x["event"] == "request"]


# ================================================================== 1. the id
def test_every_request_gets_its_own_id(client, pipeline):
    """Catches: a fixed or reused id -- then two users' questions can't be told apart in the log."""
    ids = {client.post("/api/ask", json={"question": QUESTION}).headers["x-request-id"] for _ in range(5)}
    assert len(ids) == 5


def test_the_id_is_the_same_in_header_body_and_log(client, pipeline, capsys):
    """Catches: the id the user sees (and can send in a report) not matching any log line."""
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": QUESTION})
    assert r.status_code == 200
    rid = r.headers["x-request-id"]
    assert r.json()["request_id"] == rid
    [line] = request_lines(lines(capsys))
    assert line["request_id"] == rid


def test_parallel_questions_never_mix_up_their_ids(client, monkeypatch, capsys):
    """Catches: the id kept in a global variable -- FastAPI answers questions in parallel threads, so one user's
    log lines would carry another user's id."""
    def slow_ask(question, top_k=None, answer_language="auto", market="us"):
        observability.log_event("inner", marker=top_k)
        time.sleep(0.05 * (top_k % 3))  # finish in a different order than started
        observability.log_event("inner_done", marker=top_k)
        return {"answer": "ok", "marker": top_k}
    monkeypatch.setattr(rag, "ask", slow_ask)
    capsys.readouterr()
    results = {}

    def one(k):
        r = client.post("/api/ask", json={"question": QUESTION, "top_k": k})
        results[k] = r
    threads = [threading.Thread(target=one, args=(k,)) for k in range(1, 9)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    logs = lines(capsys)
    for k, r in results.items():
        rid = r.headers["x-request-id"]
        assert r.json()["request_id"] == rid and r.json()["marker"] == k
        mine = [x for x in logs if x.get("request_id") == rid]
        assert {x["event"] for x in mine} == {"inner", "inner_done", "request"}
        assert all(x["marker"] == k for x in mine if x["event"].startswith("inner"))
        assert next(x for x in mine if x["event"] == "request")["top_k"] == k
    assert len({r.headers["x-request-id"] for r in results.values()}) == 8


@pytest.mark.parametrize("sent,reused", [
    ("frontend-7f3a9c1e", True),          # a normal id from the frontend / a proxy: follow it end to end
    ("abc", False),                       # too short
    ("x" * 65, False),                    # too long
    ('evil"}\n{"event": "fake', False),   # would break / forge a log line
    ("id with spaces 123", False),
])
def test_a_safe_client_id_is_reused_and_an_unsafe_one_replaced(client, pipeline, capsys, sent, reused):
    """Catches: a client writing fake lines into our log through the X-Request-ID header."""
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": QUESTION}, headers={"X-Request-ID": sent})
    rid = r.headers["x-request-id"]
    assert (rid == sent) is reused
    assert request_lines(lines(capsys))[0]["request_id"] == rid


# ================================================================== 2. the final log line
def test_one_question_writes_exactly_one_final_json_line(client, pipeline, capsys):
    """Catches: the old two-line logging (rag's "ask" line + the API line) or plain-text lines that a log tool
    can't read."""
    capsys.readouterr()
    client.post("/api/ask", json={"question": QUESTION})
    logs = lines(capsys)  # fails on any non-JSON line
    assert len(request_lines(logs)) == 1
    assert not [x for x in logs if x["event"] == "ask"]  # merged into the request line


def test_the_final_line_has_everything_needed_to_debug_a_question(client, pipeline, capsys):
    """Catches: a log line that says "200" but not which country, language, mode, model or which stage was slow."""
    capsys.readouterr()
    client.post("/api/ask", json={"question": QUESTION, "market": "us", "answer_language": "en"})
    [line] = request_lines(lines(capsys))
    for field in ("request_id", "method", "path", "status", "duration_ms", "question_len", "question_hash",
                  "market", "language", "answer_language", "detected_drugs", "found", "mode", "risk", "model",
                  "meaning_check", "timings"):
        assert field in line, field
    assert line["method"] == "POST" and line["path"] == "/api/ask" and line["status"] == 200
    assert line["market"] == "us" and line["mode"] == "sourced" and line["detected_drugs"] == ["metformin"]
    assert {"embed_ms", "vector_search_ms", "llm_ms", "total_ms"} <= set(line["timings"])
    assert line["duration_ms"] >= line["timings"]["total_ms"]


def test_other_api_calls_are_logged_too_but_not_the_website_files(client, capsys):
    """Catches: health pings / drug-list calls missing from the log, or one line per CSS/JS file."""
    capsys.readouterr()
    client.get("/api/fact")
    r = client.get("/")
    logs = request_lines(lines(capsys))
    assert [x["path"] for x in logs] == ["/api/fact"]
    assert r.headers["x-request-id"]  # the header is still there


# ================================================================== 3. errors
def test_llm_down_is_logged_as_503_with_its_code(client, monkeypatch, capsys):
    """Catches: an outage visible to users but not in the log (or logged without the id)."""
    def down(*_a, **_k):
        raise llm.LLMUnavailable("all LLM providers failed: groq: 503; gemini: 503")
    monkeypatch.setattr(rag, "ask", down)
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": "Can I take 6 tablets a day?"})
    assert r.status_code == 503
    logs = lines(capsys)
    [line] = request_lines(logs)
    assert line["status"] == 503 and line["error"] == "LLM_UNAVAILABLE"
    detail = next(x for x in logs if x["event"] == "llm_unavailable")
    assert detail["request_id"] == line["request_id"] == r.json()["request_id"]
    assert "groq: 503" in detail["error"]  # the raw provider errors are in the log (not in the answer)


def test_search_down_is_logged_as_502_with_its_code(client, monkeypatch, capsys):
    def down(*_a, **_k):
        raise RuntimeError("qdrant connection refused")
    monkeypatch.setattr(rag, "ask", down)
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": QUESTION})
    assert r.status_code == 502
    logs = lines(capsys)
    [line] = request_lines(logs)
    assert line["status"] == 502 and line["error"] == "UPSTREAM_ERROR"
    assert any(x["event"] == "ask_failed" and "qdrant connection refused" in x["error"]
               and x["request_id"] == line["request_id"] for x in logs)


def test_a_crash_shows_the_user_the_same_id_that_is_in_the_log(client, monkeypatch, capsys):
    """Catches: "error id X" in the user's message while the log has another id (before Part 1, CatchCrashes
    made its own id) -- the owner could not find the crash a user reported."""
    monkeypatch.setattr(rag, "library", lambda *_a, **_k: 1 / 0)
    capsys.readouterr()
    r = client.get("/api/drugs")
    assert r.status_code == 500
    rid = r.headers["x-request-id"]
    assert r.json()["request_id"] == rid and f"error id {rid}" in r.json()["detail"]
    logs = lines(capsys)
    crash = next(x for x in logs if x["event"] == "crash")
    assert crash["request_id"] == rid and "ZeroDivisionError" in crash["traceback"]
    [line] = request_lines(logs)
    assert line["request_id"] == rid and line["status"] == 500 and line["error"] == "INTERNAL_ERROR"


def test_every_error_answer_carries_the_request_id(client, monkeypatch):
    """Catches: an error the user can report only as "it didn't work" -- with the id the owner finds it."""
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(rag, "ask", lambda *_a, **_k: {"answer": "ok"})
    client.post("/api/ask", json={"question": QUESTION})
    responses = [
        client.post("/api/ask", json={"question": ""}),               # 400
        client.get("/api/nothing-here"),                               # 404
        client.get("/api/ask"),                                        # 405
        client.post("/api/ask", json={"question": "x" * 20_000}),     # 413
        client.post("/api/ask", json={"question": QUESTION}),          # 429 (second question, limit 1)
    ]
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(rag, "library", lambda *_a, **_k: 1 / 0)
    responses.append(client.get("/api/drugs"))                         # 500

    def fail(exc):
        def f(*_a, **_k):
            raise exc
        return f
    monkeypatch.setattr(rag, "ask", fail(RuntimeError("down")))
    responses.append(client.post("/api/ask", json={"question": QUESTION}))   # 502
    monkeypatch.setattr(rag, "ask", fail(llm.LLMUnavailable("down")))
    responses.append(client.post("/api/ask", json={"question": QUESTION}))   # 503
    assert [r.status_code for r in responses] == [400, 404, 405, 413, 429, 500, 502, 503]
    for r in responses:
        assert r.json()["request_id"] == r.headers["x-request-id"], r.status_code


# ================================================================== 4. events inside a question
def test_events_inside_a_question_carry_its_id(client, monkeypatch, capsys):
    """Catches: "groq failed" / "judge unavailable" lines that can't be linked to the question they belong to."""
    for name, value in {"_cooldown_until": {}, "_last_probe": {}, "_groq_model": None, "_discovered": []}.items():
        monkeypatch.setattr(llm, name, value)
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["groq", "gemini"])
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", [])
    monkeypatch.setattr(config, "JUDGE_MODE", "always")

    def groq(*_a, **_k):
        raise RuntimeError("401 invalid api key")  # not retryable: fails at once

    def gemini(model, system, user, timeout=None):
        if system == rag.JUDGE_PROMPT:
            raise RuntimeError("400 judge request rejected")
        return "RISK: NONE\nMetformin treats type 2 diabetes [1]."
    monkeypatch.setattr(llm, "_groq", groq)
    monkeypatch.setattr(llm, "_gemini", gemini)
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(CHUNK)])
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": QUESTION})
    assert r.status_code == 200 and r.json()["meaning_check"] == "unavailable"
    rid = r.headers["x-request-id"]
    logs = lines(capsys)
    events = {x["event"] for x in logs if x.get("request_id") == rid}
    assert {"llm_failed", "judge_unavailable", "request"} <= events
    assert next(x for x in logs if x["event"] == "llm_failed")["provider"] == "groq"


def test_rag_ask_outside_the_api_still_logs_its_summary(pipeline, capsys):
    """Catches: scripts (debug_ask, evaluate) losing their "ask" log line now that the API merges it."""
    capsys.readouterr()
    rag.ask(QUESTION)
    [line] = [x for x in lines(capsys) if x["event"] == "ask"]
    assert line["mode"] == "sourced" and "request_id" not in line and "total_ms" in line["timings"]


# ================================================================== 5. privacy
SENSITIVE = "I am 32 weeks pregnant, my name is Anna Kowalska, can I take metformin?"


def test_by_default_the_question_text_is_in_no_log_line(client, pipeline, monkeypatch, capsys):
    """Catches: health questions (GDPR special category) ending up readable in the server logs -- also through
    an error message that happens to contain the question."""
    capsys.readouterr()
    r = client.post("/api/ask", json={"question": SENSITIVE})
    assert r.status_code == 200

    def leaky(question, *_a, **_k):
        raise RuntimeError(f"embedding failed for input: {question}")
    monkeypatch.setattr(rag, "ask", leaky)
    assert client.post("/api/ask", json={"question": SENSITIVE}).status_code == 502
    out = capsys.readouterr().out
    assert "Anna" not in out and "pregnant" not in out
    logs = [json.loads(x) for x in out.splitlines() if x.strip()]
    ok = request_lines(logs)[0]
    assert "question" not in ok
    assert ok["question_len"] == len(SENSITIVE) and ok["question_hash"] and ok["detected_drugs"] == ["metformin"]
    assert any("[question]" in x.get("error", "") for x in logs if x["event"] == "ask_failed")


def test_with_log_questions_on_the_text_is_logged(client, pipeline, monkeypatch, capsys):
    """Catches: the debugging switch not working."""
    monkeypatch.setattr(config, "LOG_QUESTIONS", True)
    capsys.readouterr()
    client.post("/api/ask", json={"question": QUESTION})
    [line] = request_lines(lines(capsys))
    assert line["question"] == QUESTION


def test_the_question_fingerprint(monkeypatch):
    """Catches: the same question getting different fingerprints (can't count repeats), different questions
    sharing one, or a fingerprint anyone could rebuild without the server's secret."""
    h = observability.question_hash
    assert h("Czy mogę brać Ibuprom z alkoholem?") == h("  czy MOGĘ brać   ibuprom z alkoholem? ")
    assert h("Ibuprom z alkoholem?") != h("Ibuprom z kawą?")
    assert h(QUESTION, secret="a") != h(QUESTION, secret="b")
    assert len(h(QUESTION)) == 12 and QUESTION.lower() not in h(QUESTION)
