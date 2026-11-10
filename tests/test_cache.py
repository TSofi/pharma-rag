"""Part 3 of observability: answer cache (app/cache.py).

The same question again within 15 min (same country, answer language, top_k; case and extra spaces don't
matter) gets the saved answer: no Qdrant, no LLM. Only clean answers are saved: a known complete mode
(sourced / general / playful / none), never an emergency, an error, an unchecked meaning or an answer with an
"issue" (a step failed or a safety check rejected it) -- those get a fresh try next time.

No network: the vector search and the LLM are fakes; the cache clock is fake.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import cache, config, llm, main, observability, rag

QUESTION = "What is metformin used for?"
CHUNK = {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is used for type 2 diabetes.",
         "url": "u", "score": 0.9, "effective_time": "20251101"}
ANSWER = {"answer": "Metformin treats type 2 diabetes [1].", "found": True, "mode": "sourced", "sources": [],
          "language": "en", "answer_language": "en", "detected_drugs": ["metformin"], "model": "groq/x",
          "meaning_check": "not_checked", "timings": {"embed_ms": 200, "llm_ms": 1500, "total_ms": 1800}}


class Clock:
    def __init__(self):
        self.t = 1_750_000_000.0

    def __call__(self):
        return self.t

    def wait(self, s):
        self.t += s


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(config, "ANSWER_CACHE_TTL_S", 900)
    monkeypatch.setattr(config, "ANSWER_CACHE_MAX", 300)
    monkeypatch.setattr(config, "LOG_QUESTIONS", False)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_DAY", 0)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(cache, "now", c)
    return c


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def pipeline(monkeypatch):
    """A fake rag.ask that counts its calls. `pipeline.result` / `pipeline.events` / `pipeline.error` decide
    what the next call returns, logs or raises."""
    class P:
        calls: list = []
        result = ANSWER
        events: list = []
        error: Exception | None = None

        def __call__(self, question, top_k=None, answer_language="auto", market="us"):
            self.calls.append((question, top_k, answer_language, market))
            for e in self.events:
                observability.log_event(e)
            if self.error:
                raise self.error
            return json.loads(json.dumps(self.result))
    p = P()
    p.calls = []
    monkeypatch.setattr(rag, "ask", p)
    return p


@pytest.fixture
def real_pipeline(monkeypatch):
    """The real rag.ask with a fake vector search and a fake LLM; counts the LLM calls."""
    calls = []
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(CHUNK)])

    def fake_llm(system, user):
        calls.append(system)
        return calls_reply[0]
    calls_reply = ["RISK: NONE\nMetformin treats type 2 diabetes [1]."]
    monkeypatch.setattr(rag.llm, "generate", fake_llm)
    return calls, calls_reply


def ask(client, question=QUESTION, **extra):
    return client.post("/api/ask", json={"question": question, **extra})


def request_line(capsys) -> dict:
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    return [x for x in lines if x["event"] == "request"][-1]


# ================================================================== 1. hits and misses
def test_the_same_question_again_is_answered_from_the_cache(client, real_pipeline):
    """Catches: a repeated question paying again for the vector search + LLM (seconds + free quota)."""
    llm_calls, _ = real_pipeline
    first = ask(client).json()
    second = ask(client).json()
    assert len(llm_calls) == 1  # the LLM ran once
    assert first["cached"] is False and second["cached"] is True
    for field in ("answer", "sources", "mode", "found", "model", "answer_language"):
        assert second[field] == first[field], field


def test_case_and_extra_spaces_do_not_matter(client, pipeline):
    ask(client, "What is metformin used for?")
    r = ask(client, "  what IS   metformin used for?  ")
    assert r.json()["cached"] is True and len(pipeline.calls) == 1


@pytest.mark.parametrize("other", [{"market": "pl"}, {"answer_language": "uk"}, {"top_k": 3}])
def test_another_country_language_or_top_k_is_a_different_question(client, pipeline, other):
    """Catches: a Polish user getting the answer saved for the US leaflets, or a Ukrainian answer shown in English."""
    ask(client)
    r = ask(client, **other)
    assert r.json()["cached"] is False and len(pipeline.calls) == 2


def test_the_default_top_k_and_an_explicit_equal_one_share_an_entry(client, pipeline):
    ask(client)
    assert ask(client, top_k=config.TOP_K).json()["cached"] is True


def test_an_answer_is_kept_for_15_minutes(client, pipeline, clock):
    """Catches: answers kept forever (a fixed leaflet index or an improved prompt would never show)."""
    ask(client)
    clock.wait(899)
    assert ask(client).json()["cached"] is True
    clock.wait(1)  # 900 s after it was saved
    assert ask(client).json()["cached"] is False and len(pipeline.calls) == 2


def test_a_hit_has_its_own_request_id_and_its_own_log_line(client, pipeline, capsys):
    """Catches: a cached answer carrying the first asker's request id (two users, one id), or hits invisible
    in the log (the metrics count them)."""
    first = ask(client)
    capsys.readouterr()
    second = ask(client)
    assert second.json()["request_id"] == second.headers["x-request-id"] != first.json()["request_id"]
    line = request_line(capsys)
    assert line["cache"] == "hit" and line["request_id"] == second.json()["request_id"]
    assert line["mode"] == "sourced" and line["detected_drugs"] == ["metformin"] and "question_hash" in line


def test_a_hit_shows_only_the_lookup_time(client, pipeline):
    """Catches: the UI showing "LLM 1.5 s" for an answer that took 2 ms (the old stage timings)."""
    ask(client)
    timings = ask(client).json()["timings"]
    assert set(timings) == {"total_ms"} and timings["total_ms"] < 100


# ================================================================== 2. what is never cached
def test_a_red_emergency_card_is_never_cached(client, real_pipeline, capsys):
    """Catches: someone in an emergency getting a 15-minute-old card instead of a fresh check."""
    llm_calls, reply = real_pipeline
    reply[0] = "RISK: EMERGENCY\nTaking too much metformin can be dangerous [1]."
    q = "I took 20 metformin tablets by mistake"
    assert ask(client, q).json()["mode"] == "emergency"
    r = ask(client, q)
    assert r.json()["mode"] == "emergency" and r.json()["cached"] is False and len(llm_calls) == 2
    assert request_line(capsys)["cache_skip_reason"] == "emergency"


def test_the_soft_red_card_shown_while_the_llm_is_down_is_never_cached(client, monkeypatch):
    """Catches: the "our AI is busy" red card staying for 15 min after the LLM is back."""
    def down(*_a, **_k):
        raise llm.LLMUnavailable("all down")
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(CHUNK)])
    monkeypatch.setattr(rag.llm, "generate", down)
    q = "My son accidentally swallowed my blood pressure pills"
    first = ask(client, q).json()
    assert first["llm_unavailable"] is True and first["cached"] is False
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "RISK: EMERGENCY\nCall now [1].")
    second = ask(client, q).json()
    assert second["cached"] is False and "llm_unavailable" not in second  # a fresh, real check


@pytest.mark.parametrize("error, status", [(llm.LLMUnavailable("down"), 503), (RuntimeError("qdrant down"), 502)])
def test_errors_are_never_cached(client, pipeline, capsys, error, status):
    pipeline.error = error
    assert ask(client).status_code == status
    line = request_line(capsys)
    assert line["cache"] == "skip" and line["cache_skip_reason"] == "error"
    pipeline.error = None
    assert ask(client).json()["cached"] is False and len(pipeline.calls) == 2


def test_an_answer_whose_meaning_could_not_be_checked_is_not_cached(client, pipeline, capsys):
    """Catches: an unchecked answer (judge LLM down) served for 15 min after the judge is back."""
    pipeline.result = {**ANSWER, "meaning_check": "unavailable"}
    ask(client)
    assert request_line(capsys)["cache_skip_reason"] == "meaning_check_unavailable"
    assert ask(client).json()["cached"] is False


@pytest.mark.parametrize("event", sorted(observability.ISSUE_EVENTS))
def test_an_answer_with_an_issue_is_not_cached(client, pipeline, capsys, event):
    """Catches: a weaker answer (translation failed, repair failed, answer rejected...) kept for 15 min although
    the next try may give the full answer."""
    pipeline.events = [event]
    ask(client)
    line = request_line(capsys)
    assert line["cache"] == "skip" and line["cache_skip_reason"] == "issues" and line["issues"] == [event]
    pipeline.events = []
    assert ask(client).json()["cached"] is False and len(pipeline.calls) == 2


def test_another_provider_answering_is_not_an_issue(client, pipeline, capsys):
    """Catches: every answer during a Groq outage being uncached (Gemini's answer is just as good)."""
    pipeline.events = ["llm_failed", "remote_embed_failed"]
    ask(client)
    line = request_line(capsys)
    assert line["cache"] == "miss" and "issues" not in line
    assert ask(client).json()["cached"] is True


def test_unknown_answer_shapes_are_not_cached(client, pipeline):
    """Catches: something new or broken (no mode, a future mode) being saved by accident -- the allowlist."""
    for result in ({"answer": "ok"}, {**ANSWER, "mode": "something_new"}):
        pipeline.result = result
        cache.answers.clear()
        ask(client)
        assert ask(client).json()["cached"] is False


def test_not_found_because_the_llm_failed_is_logged_and_not_cached(client, monkeypatch, capsys):
    """Catches: the silent case found while planning: the "not found" step's LLM call fails -> the plain "not
    found" text with no log line at all -> (with a cache) "not found" for 15 min after the LLM is back."""
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [{**CHUNK, "score": 0.01}])  # nothing relevant

    def down(system, user):
        raise RuntimeError("503 overloaded")
    monkeypatch.setattr(rag.llm, "generate", down)
    capsys.readouterr()
    r = ask(client, "What is zzzfakedrug used for?")
    assert r.status_code == 200 and r.json()["mode"] == "none"
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert any(x["event"] == "fallback_failed" for x in lines)
    req = [x for x in lines if x["event"] == "request"][-1]
    assert req["issues"] == ["fallback_failed"] and req["cache_skip_reason"] == "issues"


def test_a_clean_not_found_answer_is_cached(client, monkeypatch):
    """Catches: the allowlist forgetting the plain "not found" (mode none) when nothing went wrong."""
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [{**CHUNK, "score": 0.01}])
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "PLAIN")
    assert ask(client, "What is the capital of France?").json()["mode"] == "none"
    assert ask(client, "What is the capital of France?").json()["cached"] is True


# ================================================================== 3. the store itself
def test_the_least_recently_used_answer_goes_first_when_full(client, pipeline, monkeypatch):
    """Catches: the cache growing without limit on the 512 MB server."""
    monkeypatch.setattr(config, "ANSWER_CACHE_MAX", 3)
    for q in ("question one?", "question two?", "question three?"):
        ask(client, q)
    ask(client, "question one?")        # used again -> the most recent
    ask(client, "question four?")       # full: "two" (least recently used) goes
    assert len(cache.answers) == 3
    assert ask(client, "question one?").json()["cached"] is True
    assert ask(client, "question two?").json()["cached"] is False


def test_changing_a_returned_answer_does_not_change_the_saved_one():
    """Catches: code that edits its copy of an answer (e.g. adds a note) silently editing every later hit."""
    key = cache.answers.key(QUESTION, "us", "auto", None)
    cache.answers.put(key, {**ANSWER, "sources": [{"id": 1, "text": "original"}]})
    got = cache.answers.get(key)
    got["sources"][0]["text"] = "changed"
    got["answer"] = "changed"
    again = cache.answers.get(key)
    assert again["sources"][0]["text"] == "original" and again["answer"] == ANSWER["answer"]
    assert "timings" not in again and "request_id" not in again  # per-request fields are not saved


def test_ttl_zero_switches_the_cache_off(client, pipeline, monkeypatch, capsys):
    monkeypatch.setattr(config, "ANSWER_CACHE_TTL_S", 0)
    ask(client)
    r = ask(client)
    assert r.json()["cached"] is False and len(pipeline.calls) == 2 and len(cache.answers) == 0
    assert request_line(capsys)["cache"] == "off"


def test_a_cached_answer_still_counts_towards_the_question_limit(client, pipeline, monkeypatch):
    """Catches: a script asking the same question forever, never limited because every answer is a hit."""
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 2)
    assert [ask(client).status_code for _ in range(3)] == [200, 200, 429]


def test_each_rule_blocks_on_its_own():
    """Catches: a rule that only works because another one happens to catch the same answer (e.g. the soft red
    card is also mode "emergency"; if the code ever marks a normal answer llm_unavailable, that rule must hold)."""
    clean = {**ANSWER}
    assert cache.skip_reason(clean, []) is None
    assert cache.skip_reason({**clean, "mode": "emergency"}, []) == "emergency"
    assert cache.skip_reason({**clean, "llm_unavailable": True}, []) == "llm_unavailable"
    assert cache.skip_reason({**clean, "meaning_check": "unavailable"}, []) == "meaning_check_unavailable"
    assert cache.skip_reason(clean, ["translation_failed"]) == "issues"
    assert cache.skip_reason({**clean, "mode": None}, []) == "unknown_mode"
