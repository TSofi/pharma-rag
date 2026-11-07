"""Point 2 of the API hardening (group 7b): input checks.

1. Language / country codes: known codes pass (also "EN" or " pl "), unknown ones -> 400 listing the allowed
   codes, and common mix-ups get a hint (language 'ua' -> 'uk', country 'uk' -> 'ua', 'cz' <-> 'cs').
2. The question is cleaned (invisible + control characters removed, spaces collapsed) BEFORE the checks, then:
   not empty, 3-500 characters, at least one letter in any alphabet. The cleaned text goes to the pipeline.
We agreed NOT to block questions by language: German / Chinese / Arabic questions still pass.

No network: rag.ask / rag.library / rag.random_fact are fakes that record their calls.
"""
import pytest
from fastapi.testclient import TestClient

from app import main, rag

OK_Q = "What is metformin?"
LANGS = ["auto", "en", "uk", "pl", "cs", "es", "fr"]
MARKETS = ["us", "pl", "ua", "cz", "es", "fr"]


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def calls(monkeypatch):
    """Every call that reached the pipeline: ("ask", question, answer_language, market) / ("drugs", market) /
    ("fact", lang). Empty list = the request was stopped at the door."""
    seen = []

    def fake_ask(question, top_k=None, answer_language="auto", market="us"):
        seen.append(("ask", question, answer_language, market))
        return {"answer": "ok [1]", "found": True, "sources": []}
    monkeypatch.setattr(rag, "ask", fake_ask)
    monkeypatch.setattr(rag, "library", lambda market="us": seen.append(("drugs", market)) or [])
    monkeypatch.setattr(rag, "random_fact", lambda seen_=None, lang="en": seen.append(("fact", lang)) or {})
    return seen


def error(r, status=400):
    assert r.status_code == status, r.text
    body = r.json()
    assert isinstance(body["detail"], str)
    return body


# ================================================================== 1. codes

@pytest.mark.parametrize("lang", LANGS)
def test_every_supported_answer_language_still_works(client, calls, lang):
    """Catches: a supported language code being rejected by the new check."""
    assert client.post("/api/ask", json={"question": OK_Q, "answer_language": lang}).status_code == 200
    assert calls == [("ask", OK_Q, lang, "us")]


@pytest.mark.parametrize("market", MARKETS)
def test_every_supported_country_still_works(client, calls, market):
    """Catches: a supported country code being rejected by the new check."""
    assert client.post("/api/ask", json={"question": OK_Q, "market": market}).status_code == 200
    assert calls == [("ask", OK_Q, "auto", market)]


@pytest.mark.parametrize("field, sent, cleaned", [
    ("answer_language", "EN", "en"), ("answer_language", " pl ", "pl"), ("answer_language", "Uk", "uk"),
    ("market", "UA", "ua"), ("market", " cz", "cz"),
])
def test_code_in_capitals_or_with_spaces_is_accepted_and_cleaned(client, calls, field, sent, cleaned):
    """Catches: "EN" rejected although it's obviously "en", or the uncleaned " pl " reaching the pipeline."""
    assert client.post("/api/ask", json={"question": OK_Q, field: sent}).status_code == 200
    assert calls[0][2 if field == "answer_language" else 3] == cleaned


@pytest.mark.parametrize("field, sent, hint", [
    ("answer_language", "ua", "uk"), ("answer_language", "cz", "cs"), ("answer_language", "gb", "en"),
    ("market", "uk", "ua"), ("market", "cs", "cz"), ("market", "en", "us"), ("market", "USA", "us"),
])
def test_a_mixed_up_code_gets_a_hint(client, calls, field, sent, hint):
    """Catches: 'ua' as a LANGUAGE (it's 'uk') or 'uk' as a COUNTRY (it's 'ua') with no hint what to use."""
    body = error(client.post("/api/ask", json={"question": OK_Q, field: sent}))
    assert body["error"] == "INVALID_VALUE" and body["field"] == field
    assert f"Did you mean '{hint}'" in body["detail"]
    assert calls == []


@pytest.mark.parametrize("field, sent", [("answer_language", "de"), ("answer_language", ""),
                                         ("market", "xx"), ("market", "")])
def test_an_unknown_code_lists_the_allowed_ones_and_gives_no_false_hint(client, calls, field, sent):
    """Catches: an unknown code accepted, or a made-up hint for a code that isn't a known mix-up."""
    body = error(client.post("/api/ask", json={"question": OK_Q, field: sent}))
    assert body["error"] == "INVALID_VALUE"
    for code in (LANGS if field == "answer_language" else MARKETS):
        assert f"'{code}'" in body["detail"]
    assert "Did you mean" not in body["detail"]
    assert calls == []


@pytest.mark.parametrize("sent", [5, None, ["en"]], ids=["number", "null", "list"])
def test_a_code_that_is_not_text_is_400(client, calls, sent):
    """Catches: a non-text code crashing the check (500) instead of a clear 400."""
    body = error(client.post("/api/ask", json={"question": OK_Q, "answer_language": sent}))
    assert body["error"] == "WRONG_TYPE" and body["field"] == "answer_language"
    assert calls == []


@pytest.mark.parametrize("url, call", [("/api/fact?lang=PL", ("fact", "pl")), ("/api/fact?lang=%20uk", ("fact", "uk")),
                                       ("/api/drugs?market=CZ", ("drugs", "cz"))])
def test_codes_in_the_address_are_cleaned_too(client, calls, url, call):
    """Catches: ?lang= / ?market= following other rules than the JSON body."""
    assert client.get(url).status_code == 200
    assert calls == [call]


@pytest.mark.parametrize("url, field, hint", [
    ("/api/fact?lang=ua", "lang", "uk"), ("/api/drugs?market=uk", "market", "ua"),
    ("/api/fact?lang=auto", "lang", None),   # "auto" is an ANSWER language, the loading-screen facts need a real one
    ("/api/drugs?market=de", "market", None),
])
def test_bad_codes_in_the_address_are_400_with_the_same_hints(client, calls, url, field, hint):
    """Catches: the address parameters missing the hint, or hinting at a code that isn't allowed there."""
    body = error(client.get(url))
    assert body["error"] == "INVALID_VALUE" and body["field"] == field
    assert (f"Did you mean '{hint}'" in body["detail"]) if hint else ("Did you mean" not in body["detail"])
    assert calls == []


# ================================================================== 2. the question

@pytest.mark.parametrize("sent, cleaned", [
    ("   What is metformin?   ", "What is metformin?"),                 # spaces around
    ("What   is\tmetformin?", "What is metformin?"),                    # many spaces / tab
    ("What is\nmetformin?", "What is metformin?"),                      # new line
    ("What\u200b is\u200d metformin?\ufeff", "What is metformin?"),     # invisible: zero-width space/joiner, BOM
    ("What is\u00a0metformin?", "What is metformin?"),                  # non-breaking space
    ("What is\x00 metformin?\x07", "What is metformin?"),               # control characters
    ("Can I take Nurofe\u0301n?", "Can I take Nurofén?"),               # e + accent mark -> one letter é
], ids=["around", "tab", "newline", "invisible", "nbsp", "control", "accent"])
def test_the_question_is_cleaned_before_it_reaches_the_pipeline(client, calls, sent, cleaned):
    """Catches: invisible or control characters reaching the search and the LLM (wasted tokens, odd matches)."""
    assert client.post("/api/ask", json={"question": sent}).status_code == 200
    assert calls[0][1] == cleaned


def test_a_broken_character_is_removed_not_a_crash(client, calls):
    r"""Catches: a lone "\ud800" (valid JSON, invalid text) crashing later when the question is sent to the LLM."""
    r = client.post("/api/ask", content=b'{"question": "What is metformin\\ud800?"}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert calls[0][1] == "What is metformin?"
    calls[0][1].encode("utf-8")  # must not raise


@pytest.mark.parametrize("sent", ["", "     ", "\u200b\u200b\u200b\u200b", "\n\t\n\t", "\x00\x00\x00", "\ufeff \u00a0 "],
                         ids=["empty", "spaces", "invisible", "newlines-tabs", "control", "mixed"])
def test_an_empty_question_is_400(client, calls, sent):
    """Catches: a question of only spaces / invisible characters being sent to the LLM."""
    body = error(client.post("/api/ask", json={"question": sent}))
    assert body["error"] == "EMPTY" and body["field"] == "question"
    assert calls == []


@pytest.mark.parametrize("sent", ["?!?!", "12345", "1.5 / 2,5", "+++ --- ***", "😀😀😀", "???\u200b???"])
def test_a_question_with_no_letters_is_400(client, calls, sent):
    """Catches: "?!?!" or "12345" being sent to the LLM (costs a call, can't be answered from a leaflet)."""
    body = error(client.post("/api/ask", json={"question": sent}))
    assert body["error"] == "NO_LETTERS" and body["field"] == "question"
    assert "letter" in body["detail"] or "word" in body["detail"]
    assert calls == []


@pytest.mark.parametrize("sent, status, code", [
    ("  ab  ", 400, "TOO_SHORT"),                              # 6 characters sent, 2 real ones
    ("a\u200bb", 400, "TOO_SHORT"),                            # 3 sent, 2 real
    ("abc", 200, None),                                        # exactly 3
    ("a" * 500, 200, None),                                    # exactly 500
    ("  " + "a" * 500 + "\u200b" * 20 + "  ", 200, None),     # 524 sent, 500 real
    ("a" * 501, 400, "TOO_LONG"),
    ("a " * 300, 400, "TOO_LONG"),                             # 599 after cleaning
])
def test_length_is_counted_after_cleaning(client, calls, sent, status, code):
    """Catches: padding with spaces / invisible characters getting a 2-letter question through, or pushing a
    valid 500-letter question over the limit."""
    r = client.post("/api/ask", json={"question": sent})
    assert r.status_code == status, r.text
    if code:
        body = r.json()
        assert body["error"] == code and body["field"] == "question"
        assert ("3" if code == "TOO_SHORT" else "500") in body["detail"]
        assert calls == []
    else:
        assert 3 <= len(calls[0][1]) <= 500


@pytest.mark.parametrize("sent", [
    "Nurofen 400?", "B12?", "Ibuprom з алкоголем", "Czy mogę brać Ibuprom z alkoholem?", "Můžu užívat Paralen?",
    "¿Puedo tomar ibuprofeno?", "Puis-je prendre du Doliprane ?", "Kann ich Ibuprofen nehmen?",
    "布洛芬的剂量是多少？", "ما هي جرعة الإيبوبروفين؟", "Ibuprofen 😀 dose?",
], ids=["brand+number", "short", "uk", "pl", "cs", "es", "fr", "de", "zh", "ar", "emoji"])
def test_real_questions_in_any_language_still_pass_unchanged(client, calls, sent):
    """Catches: the letter check rejecting a real question (other alphabets, short names, numbers, emoji)."""
    assert client.post("/api/ask", json={"question": sent}).status_code == 200
    assert calls == [("ask", sent, "auto", "us")]


def test_code_lists_match_the_rest_of_the_app(client, calls):
    """Catches: a language or country added to the app (rag.LANGUAGES / config.MARKETS) but not to the API
    checks, or the other way round."""
    from app import config, inputs
    assert set(inputs.UI_LANGUAGES) == set(rag.LANGUAGES)
    assert set(inputs.ANSWER_LANGUAGES) == set(rag.LANGUAGES) | {"auto"}
    for market in config.MARKETS:  # every configured country passes the check
        assert client.get(f"/api/drugs?market={market}").status_code != 400
    assert set(config.MARKETS) == set(MARKETS)


def test_a_hint_never_points_to_a_code_that_is_not_allowed_there():
    """Catches: a future hint (e.g. 'ua' -> 'uk') shown where 'uk' itself isn't accepted, sending the user
    from one 400 to the next."""
    from pydantic_core import PydanticCustomError
    from app import inputs
    check = inputs._code_check(lambda: ("en", "pl"), {"ua": "uk", "eng": "en"})
    with pytest.raises(PydanticCustomError) as bad:
        check("ua")
    assert "Did you mean" not in str(bad.value)
    with pytest.raises(PydanticCustomError) as good:
        check("eng")
    assert "Did you mean 'en'" in str(good.value)
