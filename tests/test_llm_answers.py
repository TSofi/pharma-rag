"""What the USER gets when the LLM misbehaves: rag.ask() and /api/ask with a scripted llm.generate.

test_llm.py checks the provider layer (retries, Groq -> Gemini, circuit breaker). This file checks the
promise of the product: a violet "sourced" answer must be backed by real [n] citations, an outage must
never be presented as "the leaflets don't say", and the user always gets a clear next step.

`xfail(strict=True)` = known bug, red first (see TESTING.md); remove the marker when the fix lands.
"""
import re

import pytest
from fastapi.testclient import TestClient

from app import llm, main, rag

BUG = lambda reason: pytest.mark.xfail(strict=True, reason=reason)  # noqa: E731

CHUNKS = [
    {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is indicated for type 2 diabetes.",
     "url": "https://example.org/metformin", "score": 0.9, "effective_time": "20251101"},
    {"drug": "metformin", "section": "Warnings", "text": "Avoid excessive alcohol intake.",
     "url": "https://example.org/metformin", "score": 0.8, "effective_time": "20251101"},
]


class ScriptedLLM:
    """llm.generate replacement. `rules` = list of (predicate(system, user), reply-or-exception);
    the first matching rule wins, `default` otherwise. Every call is recorded."""
    def __init__(self, default="Metformin treats type 2 diabetes [1]."):
        self.default, self.rules, self.calls = default, [], []

    def when(self, pred, reply):
        self.rules.append((pred, reply))
        return self

    def __call__(self, system, user):
        self.calls.append((system, user))
        reply = next((r for p, r in self.rules if p(system, user)), self.default)
        if isinstance(reply, list):  # a sequence: one item per call, the last one repeats
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


is_answer = lambda s, u: s == rag.SYSTEM_PROMPT          # noqa: E731
is_triage = lambda s, u: s.startswith("You are PharmaRAG's fallback voice")  # noqa: E731
is_translation = lambda s, u: s.startswith("Translate this drug-information answer")  # noqa: E731
asks_polish = lambda s, u: is_answer(s, u) and u.rstrip().endswith("Write the answer in Polish.")  # noqa: E731


@pytest.fixture
def llm_script(monkeypatch):
    fake = ScriptedLLM()
    monkeypatch.setattr(rag.llm, "generate", fake)
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(c) for c in CHUNKS])
    monkeypatch.setattr(llm, "last_model_used", "")
    return fake


# frontend/app.js: `const busy = /503|UNAVAILABLE|overload|high demand|429|RESOURCE_EXHAUSTED/i.test(e.message)`
FRONTEND_BUSY = r"503|UNAVAILABLE|overload|high demand|429|RESOURCE_EXHAUSTED"


def outage():
    return RuntimeError("503 UNAVAILABLE. The model is overloaded.")


# ================================================================== 1. provider outage
def test_outage_during_a_sourced_answer_is_an_error_not_a_refusal(llm_script):
    """Catches: an outage being swallowed and shown as "I couldn't find this in the leaflets", which
    would tell the user something false about the leaflet."""
    llm_script.default = outage()
    with pytest.raises(RuntimeError, match="503"):
        rag.ask("What is metformin used for?")


def test_outage_reaches_the_browser_as_an_error_status(llm_script):
    """Catches: /api/ask answering 200 with an empty body on an outage (the UI would render nothing)."""
    llm_script.default = outage()
    r = TestClient(main.app).post("/api/ask", json={"question": "What is metformin used for?"})
    assert r.status_code >= 500 and "503" in r.json()["detail"]


def test_total_outage_is_a_503_with_retry_after_and_a_clean_message(llm_script):
    """Catches: the user seeing "Something went wrong" plus a raw SDK error, instead of "the service is
    busy, retry in N seconds"."""
    llm_script.default = llm.LLMUnavailable("groq: 429 ...; gemini: 503 ...", retry_after_s=20)
    r = TestClient(main.app).post("/api/ask", json={"question": "What is metformin used for?"})
    assert r.status_code == 503 and r.headers["Retry-After"] == "20"
    detail = r.json()["detail"]
    assert re.search(FRONTEND_BUSY, detail, re.I), "the UI must show 'Service busy, try again' + retry button"
    assert "gemini: 503" not in detail, "no raw provider text for the user"


def test_triage_outage_after_empty_retrieval_still_answers_politely(llm_script, monkeypatch):
    """Catches: the optional triage call (no sources found) failing the whole request."""
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [])
    llm_script.default = outage()
    res = rag.ask("What is metformin used for?")
    assert res["found"] is False and res["mode"] == "none" and res["answer"] == rag.NOT_FOUND["en"]


def test_model_label_is_never_taken_from_a_previous_request(llm_script, monkeypatch):
    """Catches: the UI naming a model that did not produce this answer (also a race between two users,
    since FastAPI serves requests from a thread pool)."""
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "last_model_used", "groq/from-someone-elses-request")
    llm_script.default = outage()
    assert not rag.ask("What is metformin used for?")["model"]


# ================================================================== 2. answers without citations
def test_partly_cited_answer_stays_sourced_and_reports_the_gap(llm_script):
    """Catches: the honesty counter disappearing, or one unsourced sentence throwing away a good answer."""
    llm_script.default = "Metformin treats type 2 diabetes [1]. It is usually taken with meals."
    res = rag.ask("What is metformin used for?")
    assert res["mode"] == "sourced" and res["uncited_sentences"] == 1 and res["sources"][0]["cited"]


def test_answer_without_any_citation_is_shown_orange_not_violet(llm_script):
    """Catches: outside knowledge ("Metformin is safe with alcohol") presented in violet as leaflet
    content. Agreed: no citation (even after the repair attempt) -> orange "general AI info"."""
    llm_script.when(is_answer, "Metformin is safe with alcohol in any amount.")
    llm_script.when(is_triage, "GENERAL: Metformin is a diabetes medicine. Ask your pharmacist.")
    res = rag.ask("What is metformin used for?")
    assert res["mode"] == "general" and res["found"] is False
    assert not any(s.get("cited") for s in res["sources"])


def test_uncited_answer_gets_one_repair_attempt(llm_script):
    """Catches: a correct-but-uncited answer being thrown away when one reminder would fix it, and the
    repair being retried in a loop (cost, latency)."""
    llm_script.when(is_answer, ["Metformin treats type 2 diabetes.", "Metformin treats type 2 diabetes [1]."])
    res = rag.ask("What is metformin used for?")
    assert res["mode"] == "sourced" and res["answer"] == "Metformin treats type 2 diabetes[1]."
    answer_calls = [u for s, u in llm_script.calls if is_answer(s, u)]
    assert len(answer_calls) == 2, "exactly one repair attempt"
    assert answer_calls[1] != answer_calls[0], "the repair prompt must remind the model to cite"


def test_answer_citing_only_nonexistent_sources_is_shown_orange(llm_script):
    """Catches: "[7]" with 2 sources being deleted and the now unsourced text still shown in violet."""
    llm_script.when(is_answer, "Metformin may be combined with alcohol [7].")
    llm_script.when(is_triage, "GENERAL: Metformin is a diabetes medicine. Ask your pharmacist.")
    res = rag.ask("What is metformin used for?")
    assert res["mode"] == "general" and res["found"] is False


@pytest.mark.parametrize("reply", ["NOT_FOUND", "**NOT_FOUND**", "NOT_FOUND.", "  NOT_FOUND\n"])
def test_refusal_variants_are_recognised(llm_script, reply):
    """Catches: a refusal wrapped in markdown/punctuation being shown as an answer."""
    llm_script.when(is_answer, reply).when(is_triage, "PLAIN")
    res = rag.ask("What is metformin used for?")
    assert res["found"] is False and res["answer"] == rag.NOT_FOUND["en"]


def test_reasoning_trace_is_never_shown_to_the_user(llm_script):
    """Catches: the model's private notes appearing in the answer, after a Groq model swap picked a
    reasoning model."""
    llm_script.default = "<think>The user asks about metformin; source 1 says diabetes.</think>\nMetformin treats type 2 diabetes [1]."
    assert rag.ask("What is metformin used for?")["answer"] == "Metformin treats type 2 diabetes[1]."


# ================================================================== 3. answer-language fallback (refuse -> English -> translate)
def test_translated_answer_keeps_its_citations(llm_script):
    """Catches: the English-then-translate path losing the sources (baseline for the two bugs below)."""
    llm_script.when(asks_polish, "NOT_FOUND").when(is_translation, "Metformina leczy cukrzycę typu 2 [1].")
    res = rag.ask("Na co jest metformina?", answer_language="pl")
    assert res["mode"] == "sourced" and res["answer"] == "Metformina leczy cukrzycę typu 2[1]."


def test_failed_translation_returns_the_cited_english_answer(llm_script):
    """Catches: throwing away a good, sourced answer because only the translation call failed."""
    llm_script.when(asks_polish, "NOT_FOUND").when(is_translation, outage())
    res = rag.ask("Na co jest metformina?", answer_language="pl")
    assert res["mode"] == "sourced" and "[1]" in res["answer"] and res["answer_language"] == "en"


def test_translation_that_drops_the_citations_is_not_used(llm_script):
    """Catches: the translator silently removing the sources, so a leaflet answer is shown uncited."""
    llm_script.when(asks_polish, "NOT_FOUND").when(is_translation, "Metformina leczy cukrzycę typu 2.")
    res = rag.ask("Na co jest metformina?", answer_language="pl")
    assert "[1]" in res["answer"]
