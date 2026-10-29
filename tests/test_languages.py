"""Group 2: the six languages (en, uk, pl, cs, es, fr).

PharmaRAG promises: question in any language, answer in the chosen language, and fixed safety texts
(refusal, emergency) that never come from the LLM. These tests make sure no language is a second-class
citizen. Each test names the mistake it catches.
"""
import re
import typing

import pytest

from app import main, rag

ALL = sorted(rag.LANGUAGES)  # ["cs", "en", "es", "fr", "pl", "uk"]


# ---------- detection of the question's language ----------
@pytest.mark.parametrize("text, lang", [
    ("Can I take ibuprofen with alcohol?", "en"),
    ("Чи можна приймати нурофен з алкоголем?", "uk"),
    ("Czy mogę brać Ibuprom z alkoholem?", "pl"),
    ("Mohu užívat Ibalgin s alkoholem?", "cs"),
    ("¿Puedo tomar ibuprofeno con alcohol?", "es"),
    ("Puis-je prendre de l'ibuprofène avec de l'alcool ?", "fr"),
])
def test_each_language_is_detected_from_a_typical_question(text, lang):
    """Catches: a language's letter/word list being edited so its questions fall through to "en" or "auto"."""
    assert rag.detect_language(text) == lang


@pytest.mark.parametrize("text, lang", [
    ("Puedo tomar este medicamento con alcohol", "es"),      # no ¿ ñ: detected by common words
    ("Je peux prendre ce medicament avec de l alcool", "fr"),  # no accents: detected by common words
    ("Jaka jest dawka dla dzieci", "pl"),                     # no ą ę ł
])
def test_language_is_detected_without_special_letters(text, lang):
    """Catches: a user typing on a phone keyboard without accents being treated as English."""
    assert rag.detect_language(text) == lang


def test_cyrillic_wins_over_mixed_latin_words():
    """Catches: a Ukrainian question with an English brand name ("Нурофен and alcohol") detected as English."""
    assert rag.detect_language("Нурофен and alcohol") == "uk"


@pytest.mark.parametrize("text", ["Nurofen", "ibuprofen 400", "Mohu uzivat lek s alkoholem?"])
def test_undecidable_text_is_auto_not_english(text):
    """Catches: a bare drug name or an unaccented Czech sentence forced to English. "auto" lets the LLM
    answer in the question's own language instead."""
    assert rag.detect_language(text) == "auto"


def test_lang_name_covers_every_language_and_auto():
    """Catches: a language code that the prompt cannot name (the LLM would get "Write the answer in cs")."""
    for code in ALL:
        assert rag.lang_name(code) == rag.LANGUAGES[code] and rag.lang_name(code)[0].isupper()
    assert rag.lang_name("auto") == "the same language as the user's question"


# ---------- fixed texts that must never be generated ----------
@pytest.mark.parametrize("lang", ALL)
def test_refusal_and_emergency_texts_exist_in_every_language(lang):
    """Catches: a Czech/Spanish/French user getting the English "This may be an emergency" text
    (the UI is in 6 languages; the safety texts must be too)."""
    assert lang in rag.NOT_FOUND, f"NOT_FOUND has no {lang} text"
    assert lang in rag.EMERGENCY, f"EMERGENCY has no {lang} text"


@pytest.mark.parametrize("lang", ALL)
def test_emergency_text_names_an_emergency_number_and_is_not_a_template(lang):
    """Catches: a translated text that lost the phone number, or a "{...}" placeholder left in."""
    text = rag.EMERGENCY[lang]
    assert "112" in text or "911" in text
    assert "{" not in text and len(text) > 60


_SCRIPT = {"uk": r"[а-яіїєґ]", "pl": r"[ąćęłńóśźż]", "cs": r"[ěščřžýáíéůú]", "es": r"[áéíóúñ¿]", "fr": r"[éèêàçù]"}


@pytest.mark.parametrize("lang", [l for l in ALL if l != "en"])
def test_fixed_texts_are_really_in_that_language(lang):
    """Catches: a copy-paste of the English text under a new key."""
    for text in (rag.NOT_FOUND[lang], rag.EMERGENCY[lang]):
        assert re.search(_SCRIPT[lang], text, re.I), f"{lang}: {text[:40]}"
        assert text != rag.NOT_FOUND["en"] and text != rag.EMERGENCY["en"]


# ---------- fallback modes (orange / red / playful / none) respect the answer language ----------
@pytest.mark.parametrize("lang", ALL)
def test_emergency_mode_returns_the_fixed_text_in_the_answer_language(monkeypatch, lang):
    """Catches: the red card showing English to a non-English user, or LLM text instead of the fixed one."""
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "EMERGENCY -- call 112 now, here is some advice...")
    text, mode = rag.fallback("I swallowed 20 pills", lang)
    assert mode == "emergency" and text == rag.EMERGENCY[lang]


@pytest.mark.parametrize("lang", ALL)
def test_plain_and_llm_failure_return_the_refusal_in_the_answer_language(monkeypatch, lang):
    """Catches: the "not found" card in the wrong language, in both the PLAIN case and the crash case."""
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: "PLAIN")
    assert rag.fallback("football scores", lang) == (rag.NOT_FOUND[lang], "none")

    def boom(s, u):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(rag.llm, "generate", boom)
    assert rag.fallback("football scores", lang) == (rag.NOT_FOUND[lang], "none")


@pytest.mark.parametrize("lang", ALL)
def test_triage_prompt_asks_for_the_answer_language(monkeypatch, lang):
    """Catches: the orange/playful answer generated in the question's language instead of the chosen one
    (the prompt must say the language name in both the system and the user message)."""
    seen = {}
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: seen.update(system=s, user=u) or "GENERAL: ok advice.")
    text, mode = rag.fallback("How do pregnancy tests work?", lang)
    name = rag.LANGUAGES[lang]
    assert mode == "general" and text == "ok advice."
    assert f"in {name}" in seen["system"] and f"Write any answer in {name}" in seen["user"]


# ---------- the sourced (violet) answer respects the answer language ----------
@pytest.fixture
def pipeline(monkeypatch):
    chunk = {"drug": "metformin", "section": "Indications and Usage", "text": "Metformin is indicated...",
             "url": "u", "score": 0.9}
    monkeypatch.setattr(rag.store, "search", lambda q, top_k, drugs=None, timings=None, **kw: [chunk])
    seen = {}
    monkeypatch.setattr(rag.llm, "generate", lambda s, u: seen.setdefault("prompts", []).append(u) or "Answer [1].")
    return seen


@pytest.mark.parametrize("lang", ALL)
def test_explicit_answer_language_is_requested_from_the_llm(pipeline, lang):
    """Catches: the answer_language dropdown being ignored for some language."""
    res = rag.ask("What is metformin used for?", answer_language=lang)
    assert res["answer_language"] == lang
    assert pipeline["prompts"][-1].endswith(f"Write the answer in {rag.LANGUAGES[lang]}.")


@pytest.mark.parametrize("question, lang", [
    ("¿Para qué sirve la metformina?", "es"),
    ("À quoi sert la metformine ?", "fr"),
    ("K čemu se používá metformin?", "cs"),
])
def test_auto_answer_language_follows_the_question(pipeline, question, lang):
    """Catches: "auto" always answering in English."""
    res = rag.ask(question, answer_language="auto", market="pl")
    assert res["language"] == lang and res["answer_language"] == lang
    assert f"Write the answer in {rag.LANGUAGES[lang]}." in pipeline["prompts"][-1]


def test_auto_with_undecidable_question_lets_the_llm_mirror_the_user(pipeline):
    """Catches: a bare "Nurofen alkohol" question being answered in English by default."""
    rag.ask("Nurofen alkohol", answer_language="auto", market="pl")
    assert "Write the answer in the same language as the user's question." in pipeline["prompts"][-1]


# ---------- API and UI stay in sync with LANGUAGES ----------
def test_api_accepts_exactly_the_supported_languages():
    """Catches: adding a language to rag.LANGUAGES but not to the FastAPI Literal (or vice versa)."""
    ask_langs = set(typing.get_args(main.AskRequest.model_fields["answer_language"].annotation))
    assert ask_langs == set(ALL) | {"auto"}


def test_frontend_ui_offers_every_language():
    """Catches: a language the API supports but the UI cannot select or has no translations for."""
    i18n = (rag.config.ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
    html = (rag.config.ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for code in ALL:
        assert re.search(rf"^\s*{code}: \{{", i18n, re.M), f"no UI translations for {code}"
    select = re.search(r'<select id="answerLang">(.*?)</select>', html, re.S).group(1)
    assert set(re.findall(r'value="(\w+)"', select)) == set(ALL) | {"auto"}
