"""Part 6: how urgent is the user's situation? Three levels.

  RED    (emergency) -- something dangerous ALREADY happened ("I took 5 x 500 mg instead of 100 mg"):
                        fixed, never-generated "call 112 now" card, no leaflet answer next to it.
  ORANGE (caution)   -- a dosing mistake / "what if" that is usually not an emergency ("forgot my lunch
                        pill, took it at dinner"): the normal violet leaflet answer + a fixed safety line.
  NONE               -- everything else, including following a doctor's instruction ("crush the pill
                        into food"): the normal answer, no warning.

Who decides:
  LLM working -> the LLM, through one "RISK: ..." line at the start of its reply (no extra call).
  LLM down    -> keywords: something was ALREADY taken + a sign of too much (or "overdosed" etc.).
                 The text is then conditional ("If someone has taken too much ...") + "service busy".

The LLM is scripted here, so these tests check the PLUMBING (the right card for each level), not the
model's judgement. The model's judgement needs real calls: part 8 (tests/integration, run by hand).
"""
import pytest
from fastapi.testclient import TestClient

from app import llm, main, rag

CHUNKS = [
    {"drug": "paracetamol", "section": "If you take more than you should", "url": "https://example.org/para",
     "text": "Adults: 1 tablet; never within 4 hours of the next dose. If you take too much, contact a doctor immediately, even if you feel well.", "score": 0.9},
    {"drug": "paracetamol", "section": "How to take", "url": "https://example.org/para",
     "text": "The tablet may be crushed and mixed with a spoon of food. If you forget a dose, take it as "
             "soon as you remember, but not within 4 hours of the next dose.", "score": 0.8},
]


class FakeLLM:
    """llm.generate replacement: `answer` for the leaflet answer (a list = one item per call), `other`
    for every other prompt (translation, triage). An exception instance is raised."""
    def __init__(self):
        self.answer: object = "RISK: NONE\nAdults: 1 tablet [1]."
        self.other: object = "PLAIN"
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        reply = self.answer if system == rag.SYSTEM_PROMPT else self.other
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(rag.llm, "generate", f)
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(c) for c in CHUNKS])
    return f


def down():
    return llm.LLMUnavailable("all LLM providers failed: groq: 503; gemini: 503")


# ================================================================== 1. LLM working: the RISK line decides
def test_clear_emergency_with_a_leaflet_found_shows_the_red_card(fake):
    """Catches: "I took 5 x 500 mg" getting a normal violet answer because the leaflet WAS found (before
    part 6 the red card appeared only when nothing was found)."""
    fake.answer = "RISK: EMERGENCY\nTaking too much paracetamol can damage the liver [1]."
    res = rag.ask("I took 5 tablets of 500 mg, I thought they were 100 mg each")
    assert res["mode"] == "emergency" and res["answer"] == rag.EMERGENCY["en"]
    assert res["sources"] == [] and res["found"] is False
    assert "llm_unavailable" not in res and "safety_note" not in res


def test_possible_risk_gets_the_leaflet_answer_plus_a_safety_note(fake):
    """Catches: a missed dose being treated like an emergency (scary red card) or getting no hint at all."""
    fake.answer = "RISK: CAUTION\nTake the missed dose as soon as you remember, not within 4 hours of the next [2]."
    res = rag.ask("I forgot my lunch pill and took it at dinner, is that ok?")
    assert res["mode"] == "sourced" and res["sources"][1]["cited"]
    assert res["safety_note"] == rag.SAFETY_NOTE["en"]


def test_following_a_doctors_instruction_is_a_normal_answer(fake):
    """Catches: your example -- "swallowed" + "child" alone must not raise any alarm."""
    fake.answer = "RISK: NONE\nThe tablet may be crushed and mixed with food [2]."
    res = rag.ask("My child swallowed a pill that the doctor told me to crush and put in his food, is that ok?")
    assert res["mode"] == "sourced" and "safety_note" not in res


@pytest.mark.parametrize("reply", ["RISK: NONE\nAdults: 1 tablet [1].", "Adults: 1 tablet [1]."],
                         ids=["risk-none", "no-risk-line"])
def test_normal_question_has_no_warning(fake, reply):
    """Catches: a warning on every answer, or a crash when the model forgets the RISK line."""
    fake.answer = reply
    res = rag.ask("What is the dose of paracetamol?")
    assert res["mode"] == "sourced" and res["answer"] == "Adults: 1 tablet[1]." and "safety_note" not in res


@pytest.mark.parametrize("reply", [
    "**RISK: CAUTION**\nTake it when you remember [2].",
    "Risk: caution\nTake it when you remember [2].",
    "RISK:CAUTION\n\nTake it when you remember [2].",
    "Take it when you remember [2].\nRISK: CAUTION",
])
def test_risk_line_is_never_shown_to_the_user(fake, reply):
    """Catches: "RISK: CAUTION" appearing in the answer text when the model formats it differently."""
    fake.answer = reply
    res = rag.ask("I forgot my morning pill, what now?")
    assert res["answer"] == "Take it when you remember[2]." and res["safety_note"] == rag.SAFETY_NOTE["en"]


def test_emergency_wins_over_not_found(fake):
    """Catches: an emergency being answered "not in the leaflet" because the leaflet had no answer."""
    fake.answer = "RISK: EMERGENCY\nNOT_FOUND"
    assert rag.ask("My son drank a whole bottle of cough syrup")["mode"] == "emergency"


def test_emergency_found_during_the_citation_retry_still_gives_the_red_card(fake):
    """Catches: the red card being lost when the first answer had no citation and only the retry (asked to
    add citations) marks the situation as an emergency."""
    fake.answer = ["RISK: NONE\nParacetamol can harm the liver.", "RISK: EMERGENCY\nContact a doctor now [1]."]
    assert rag.ask("I took 5 tablets of 500 mg, I thought they were 100 mg each")["mode"] == "emergency"


def test_the_most_urgent_risk_line_wins(fake):
    """Catches: a model that contradicts itself ("RISK: NONE" ... "RISK: EMERGENCY") being read as NONE."""
    fake.answer = "RISK: NONE\nContact a doctor immediately [1].\nRISK: EMERGENCY"
    assert rag.ask("My son drank a whole bottle of cough syrup")["mode"] == "emergency"


def test_emergency_in_the_english_retry_gives_the_red_card_in_the_users_language(fake):
    """Catches: the red card being lost (or shown in English) on the refuse -> English -> translate path."""
    fake.answer = ["RISK: NONE\nNOT_FOUND", "RISK: EMERGENCY\nToo much paracetamol damages the liver [1]."]
    res = rag.ask("Wziąłem 10 tabletek paracetamolu", answer_language="pl")
    assert res["mode"] == "emergency" and res["answer"] == rag.EMERGENCY["pl"]


def test_translator_never_sees_the_risk_line_and_the_note_is_in_the_answer_language(fake):
    """Catches: "RISK: CAUTION" being translated into the answer, or the safety line in English."""
    fake.answer = ["RISK: CAUTION\nNOT_FOUND", "RISK: CAUTION\nTake it as soon as you remember [2]."]
    fake.other = "Przyjmij ją, gdy tylko sobie przypomnisz [2]."
    res = rag.ask("Zapomniałem tabletki, co teraz?", answer_language="pl")
    translation_input = [u for s, u in fake.calls if s.startswith("Translate this")][0]
    assert "RISK" not in translation_input
    assert res["mode"] == "sourced" and res["safety_note"] == rag.SAFETY_NOTE["pl"]


def test_prompt_defines_the_three_levels():
    """Catches: the RISK rule (or the doctor's-instruction exception) being dropped from the prompt."""
    prompt = " ".join(rag.SYSTEM_PROMPT.split())  # ignore line breaks
    for text in ("RISK: EMERGENCY", "RISK: CAUTION", "RISK: NONE", "ALREADY", "crushing a tablet into food"):
        assert text in prompt


@pytest.mark.parametrize("lang", list(rag.LANGUAGES))
def test_fixed_texts_exist_in_every_language(lang):
    """Catches: a KeyError or an English text for a Czech/Spanish/... user on the orange line or the
    LLM-down red card."""
    assert "112" in rag.SAFETY_NOTE[lang] and "112" in rag.EMERGENCY_SOFT[lang]
    assert rag.EMERGENCY_SOFT[lang] != rag.EMERGENCY[lang]


def test_keywords_are_not_used_while_the_llm_works(fake):
    """Catches: the keyword list overruling the LLM. When the LLM works it judges the context ("as the
    doctor said"), keywords would only add false alarms."""
    fake.answer = "RISK: NONE\nThe tablet may be crushed and mixed with food [2]."
    res = rag.ask("My child swallowed 20 paracetamol tablets crushed in food as the doctor said")
    assert res["mode"] == "sourced"


# ================================================================== 2. LLM down: keywords decide
EMERGENCY_WHEN_DOWN = [
    ("en", "My child swallowed 20 paracetamol tablets, what do I do?"),
    ("en", "I took 5 tablets of 500 mg, I thought they were 100 mg each"),
    ("en", "My son accidentally swallowed my blood pressure pills"),
    ("en", "I think I overdosed on ibuprofen"),
    ("uk", "Дитина випила пляшку сиропу, що робити?"),
    ("uk", "Я випадково прийняла 10 таблеток парацетамолу"),
    ("pl", "Połknąłem całe opakowanie tabletek nasennych"),
    ("pl", "Dziecko przez pomyłkę zjadło moje tabletki"),
    ("cs", "Dítě omylem snědlo celé balení tablet"),
    ("es", "Mi hijo se tomó toda la caja de pastillas, ¿qué hago?"),
    ("fr", "Mon enfant a avalé toute la boîte de paracétamol"),
]
MARKET = {"en": "us", "uk": "ua", "pl": "pl", "cs": "cz", "es": "es", "fr": "fr"}  # each language's own country
NOT_EMERGENCY_WHEN_DOWN = [
    ("en", "My child swallowed a pill that the doctor told me to crush and put in his food"),
    ("en", "I forgot my lunch pill and took it at dinner, is that ok?"),
    ("en", "What happens in case of an overdose of paracetamol?"),
    ("en", "Can I take 6 tablets a day?"),
    ("en", "Does this medicine cause suicidal thoughts?"),
    ("en", "I took my pill with a bottle of water, is that fine?"),
    ("uk", "Лікар сказав розтерти таблетку і додати в їжу дитині, чи можна?"),
    ("uk", "Що буде, якщо випити дві таблетки?"),
    ("pl", "Zapomniałem wziąć tabletkę rano, czy mogę wziąć ją wieczorem?"),
    ("cs", "Co když zapomenu vzít tabletku?"),
    ("es", "¿Qué pasa si tomo dos comprimidos?"),
    ("fr", "Que faire si j'oublie de prendre un comprimé ?"),
]


@pytest.mark.parametrize("leaflet_found", [True, False], ids=["leaflet-found", "nothing-found"])
@pytest.mark.parametrize("lang, question", EMERGENCY_WHEN_DOWN)
def test_llm_down_and_clear_emergency_words_give_the_soft_red_card(fake, monkeypatch, leaflet_found, lang, question):
    """Catches: a possible poisoning getting only an error page or "not found" during an LLM outage."""
    if not leaflet_found:
        monkeypatch.setattr(rag.store, "search", lambda *a, **k: [])
    fake.answer = fake.other = down()
    res = rag.ask(question, market=MARKET[lang])
    assert res["mode"] == "emergency" and res["llm_unavailable"] is True
    assert res["answer"] in rag.EMERGENCY_SOFT.values()


@pytest.mark.parametrize("lang, question", NOT_EMERGENCY_WHEN_DOWN)
def test_llm_down_and_an_ordinary_question_is_just_busy(fake, monkeypatch, lang, question):
    """Catches: false alarms -- "what if", a missed dose or a doctor's instruction must not get the red
    card; with the LLM down the user gets "busy, try again" (or "not found" if nothing was found)."""
    fake.answer = fake.other = down()
    with pytest.raises(llm.LLMUnavailable):
        rag.ask(question, market=MARKET[lang])
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [])
    assert rag.ask(question, market=MARKET[lang])["mode"] == "none"


def test_llm_down_and_a_non_english_question_on_the_us_market_is_busy_not_an_alarm(fake):
    """Catches: the US-market translation step (which needs the LLM) crashing with a raw error; an
    ordinary question must end as LLMUnavailable (-> 503 "busy"), an emergency as the red card."""
    fake.answer = fake.other = down()
    with pytest.raises(llm.LLMUnavailable):
        rag.ask("Zapomniałem wziąć tabletkę rano, czy mogę wziąć ją wieczorem?", market="us")
    assert rag.ask("Połknąłem całe opakowanie tabletek nasennych", market="us")["mode"] == "emergency"


def test_soft_red_card_is_in_the_users_language(fake):
    """Catches: an English emergency text for a Ukrainian user during an outage."""
    fake.answer = fake.other = down()
    assert rag.ask("Дитина випила пляшку сиропу, що робити?", market="ua")["answer"] == rag.EMERGENCY_SOFT["uk"]


def test_api_llm_down_emergency_is_a_200_red_card_and_ordinary_question_a_503(fake):
    """Catches: the red card being hidden behind an HTTP error the UI shows as "busy"."""
    fake.answer = fake.other = down()
    c = TestClient(main.app)
    r = c.post("/api/ask", json={"question": "My son accidentally swallowed my blood pressure pills"})
    assert r.status_code == 200 and r.json()["mode"] == "emergency" and r.json()["llm_unavailable"] is True
    assert c.post("/api/ask", json={"question": "Can I take 6 tablets a day?"}).status_code == 503
