"""Unit tests for the parts of the pipeline that don't need a model: drug detection, typo
correction, language detection, chunking and the citation guard."""
import re

import pytest

from app import rag
from scripts.ingest import chunk_text


# ---------- drug detection ----------
@pytest.mark.parametrize("question, expected", [
    ("What is metformin used for?", ["metformin"]),
    ("Can I take Nurofen with Glucophage?", ["ibuprofen", "metformin"]),  # brand names -> generics
    ("What is Ozempic used for?", []),                                     # not in the library
])
def test_detect_drugs(question, expected):
    assert rag.detect_drugs(question) == expected


def test_typo_is_corrected_and_reported():
    drugs, corrections = rag.detect_drugs_fuzzy("What is the dose of ibumporfen?")
    assert drugs == ["ibuprofen"]
    assert corrections == [{"typed": "ibumporfen", "matched": "ibuprofen"}]


def test_long_misspelling_is_corrected_but_similar_everyday_word_is_not():
    assert rag.detect_drugs("What is ibuprol?") == ["ibuprofen"]
    assert rag.detect_drugs("Take it in the morning, Nurofen fan") == ["ibuprofen"]   # via brand, not "morning"
    assert rag.detect_drugs("Should I take it in the morning?") == []


def test_everyday_words_are_not_mistaken_for_brands():
    # "leave" is close to the brand "Aleve" but must not trigger a drug filter
    assert rag.detect_drugs("Can patients leave the hospital early?") == []


# ---------- language ----------
@pytest.mark.parametrize("text, lang", [
    ("What is metformin used for?", "en"),
    ("Для чого використовують метформін?", "uk"),
    ("Czy można brać paracetamol w ciąży?", "pl"),
    ("Jaka jest dawka dla dzieci", "pl"),  # no Polish letters, detected by common words
])
def test_detect_language(text, lang):
    assert rag.detect_language(text) == lang


# ---------- chunking ----------
def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 450 for c in chunks)
    # consecutive chunks share some text (the overlap)
    assert chunks[0][-30:].split()[-1] in chunks[1]


def test_giant_sentence_is_hard_split():
    assert len(chunk_text("x" * 3000, size=900, overlap=100)) >= 3


# ---------- the full ask() with the vector DB and LLM mocked ----------
@pytest.fixture
def mocked_pipeline(monkeypatch):
    chunk = {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is indicated...",
             "url": "u", "score": 0.9, "effective_time": "20251101"}
    monkeypatch.setattr(rag.store, "search", lambda q, top_k, drugs=None, timings=None: [chunk, {**chunk, "score": 0.2}])
    calls = {}

    def fake_llm(system, user):
        calls["system"], calls["user"] = system, user
        return calls.get("reply", "Metformin treats type 2 diabetes [1]. Also something [7].")
    monkeypatch.setattr(rag.llm, "generate", fake_llm)
    return calls


def test_low_score_chunks_are_dropped_and_fake_citations_removed(mocked_pipeline):
    res = rag.ask("What is metformin used for?")
    assert res["found"] is True
    assert len(res["sources"]) == 1               # the 0.2-score chunk is below MIN_SCORE
    assert "[7]" not in res["answer"]             # citation to a non-existent source is stripped
    assert res["sources"][0]["cited"] is True


def test_refusal_when_llm_says_not_found(mocked_pipeline):
    mocked_pipeline["reply"] = "NOT_FOUND"
    res = rag.ask("What is metformin used for?")
    assert res["found"] is False and res["sources"] == []


def _triage(reply):
    """Fake LLM: NOT_FOUND for the label-answer step, `reply` for the fallback triage step."""
    return lambda system, user: reply if "fallback voice" in system else "NOT_FOUND"


def test_absurd_question_gets_playful_reply(monkeypatch, mocked_pipeline):
    monkeypatch.setattr(rag.llm, "generate", _triage("JOKE: Expanding foam is for window frames, not stomachs."))
    res = rag.ask("Can I eat expanding foam?")
    assert res["found"] is False and res["mode"] == "playful" and res["playful"] is True and "foam" in res["answer"]


def test_emergency_gets_fixed_message_not_generated_text(monkeypatch, mocked_pipeline):
    monkeypatch.setattr(rag.llm, "generate", _triage("EMERGENCY"))
    res = rag.ask("My child swallowed 20 metformin tablets")
    assert res["mode"] == "emergency" and res["answer"] == rag.EMERGENCY["en"] and res["playful"] is False


def test_general_answer_is_marked_and_has_no_citations(monkeypatch, mocked_pipeline):
    monkeypatch.setattr(rag.llm, "generate", _triage(
        "GENERAL: Home tests detect hCG in urine [1]. A blood test is the most reliable. Confirm with a doctor."))
    res = rag.ask("How do pregnancy tests work?")
    assert res["found"] is False and res["mode"] == "general" and res["sources"] == []
    assert "[1]" not in res["answer"] and res["answer"].startswith("Home tests")


def test_non_health_question_gets_plain_message(monkeypatch, mocked_pipeline):
    monkeypatch.setattr(rag.llm, "generate", _triage("PLAIN"))
    res = rag.ask("Who won the football match yesterday?")
    assert res["mode"] == "none" and res["answer"] == rag.NOT_FOUND["en"]


def test_sourced_answer_mode(mocked_pipeline):
    assert rag.ask("What is metformin used for?")["mode"] == "sourced"


def test_answer_language_is_passed_to_the_llm(mocked_pipeline):
    rag.ask("What is metformin used for?", answer_language="uk")
    assert "Write the answer in Ukrainian" in mocked_pipeline["user"]


def test_refusal_in_other_language_falls_back_to_english_then_translates(monkeypatch, mocked_pipeline):
    replies = iter(["NOT_FOUND", "Metformin treats type 2 diabetes [1].", "Метформін лікує діабет 2 типу [1]."])
    monkeypatch.setattr(rag.llm, "generate", lambda system, user: next(replies))
    res = rag.ask("What is metformin used for?", answer_language="uk")
    assert res["found"] is True
    assert res["answer"].startswith("Метформін") and res["sources"][0]["cited"]


def test_unknown_market_is_rejected():
    with pytest.raises(ValueError):
        rag.ask("What is metformin used for?", market="pl")


# ---------- loading-screen facts ----------
def test_facts_file_is_valid_and_matches_library():
    facts = rag._facts()
    library = {line.split(":")[0].strip() for line in open(rag.config.ROOT / "data" / "drugs.txt", encoding="utf-8")
               if line.strip() and not line.startswith("#")}
    assert len(facts) >= 20
    for f in facts:
        assert f["drug"] in library, f["drug"]
        assert 40 <= len(f["text"]) <= 220, f["text"]   # short enough to read at a glance
        assert f["url"].startswith("https://")


def test_facts_do_not_repeat_seen_drugs():
    seen = {f["drug"] for f in rag._facts()} - {"metformin"}
    assert rag.random_fact(seen)["drug"] == "metformin"


def test_facts_are_translated():
    for f in rag._facts():
        assert f.get("text_uk") and f.get("text_pl")
    assert re.search("[а-яіїє]", rag.random_fact(lang="uk")["text"])


def test_response_includes_stage_timings(mocked_pipeline):
    res = rag.ask("What is metformin used for?")
    assert {"llm_ms", "total_ms"} <= res["timings"].keys()
    assert res["timings"]["total_ms"] >= res["timings"]["llm_ms"]


@pytest.mark.parametrize("raw, clean", [
    ("Avoid NSAIDs 【2】.", "Avoid NSAIDs[2]."),
    ("Symptoms 【1†L1-L3】 【2†L1-L3】.", "Symptoms[1][2]."),
    ("See [1, 3].", "See[1][3]."),
    ("See [2-4].", "See[2][3][4]."),
    ("Plain [1] stays.", "Plain[1] stays."),
])
def test_citation_styles_are_normalized(raw, clean):
    assert rag.normalize_citations(raw) == clean


def test_chatty_translation_is_rejected(monkeypatch):
    advice = "I'm sorry you're hurt. If you've injured your finger, clean it and apply a cold pack. " * 3
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: advice)
    assert rag.translate_query("я вдарив палець що робити") == "я вдарив палець що робити"
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "I hit my finger, what should I do?")
    assert rag.translate_query("я вдарив палець що робити") == "I hit my finger, what should I do?"
