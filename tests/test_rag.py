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
    monkeypatch.setattr(rag.store, "search", lambda q, top_k, drugs=None: [chunk, {**chunk, "score": 0.2}])
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


def test_absurd_question_gets_playful_reply(monkeypatch, mocked_pipeline):
    def fake(system, user):
        return "Expanding foam is for window frames, not stomachs. Ask me about a real medicine!" \
            if "witty front-desk" in system else "NOT_FOUND"
    monkeypatch.setattr(rag.llm, "generate", fake)
    res = rag.ask("Can I eat expanding foam?")
    assert res["found"] is False and res["playful"] is True and "foam" in res["answer"]


def test_serious_or_plain_refusal_is_not_joked_about(monkeypatch, mocked_pipeline):
    monkeypatch.setattr(rag.llm, "generate", lambda system, user: "PLAIN" if "witty" in system else "NOT_FOUND")
    res = rag.ask("My child swallowed 20 metformin tablets")
    assert res["playful"] is False and res["answer"] == rag.NOT_FOUND["en"]


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
