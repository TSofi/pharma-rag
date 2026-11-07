"""Group 6d: the MEANING of a cited sentence must match its source (an LLM "judge").

The number check (6c) can't see "safe in pregnancy [1]" when [1] says "do not use". A second LLM call
(the judge) gets every cited sentence with the source(s) it cites and answers per sentence:
SUPPORTED / CONTRADICTED / NOT_IN_SOURCE.

Decisions (settings in .env):
  1b  JUDGE_MODE=risky   -> only when the question or the answer touches a risky topic (pregnancy,
                           breastfeeding, children, alcohol, driving, other medicines, allergy, kidney/liver);
                           "always" / "off" also possible.
  2a  JUDGE_STRICT=true  -> NOT_IN_SOURCE counts like CONTRADICTED.
  3a  judge down / unreadable -> the answer stays violet (numbers are already checked), logged as
                           meaning_check="unavailable".
A problem -> one repair attempt naming the sentence -> judged again -> still wrong -> orange.

The judge is scripted here: these tests check the plumbing, not how good the judge is (real-API testing).
"""
import pytest

from app import config, llm, rag

SOURCES = [
    {"drug": "Nurofen Forte", "section": "Pregnancy and breast-feeding", "url": "u", "score": 0.9,
     "text": "Do not use in the last 3 months of pregnancy."},
    {"drug": "Nurofen Forte", "section": "How to take", "url": "u", "score": 0.8,
     "text": "Adults: 1 tablet every 6-8 hours."},
]


class FakeLLM:
    """`answers` for the leaflet answer, `verdicts` for the judge (one item per call, the last repeats;
    an exception is raised). Everything else (triage) -> GENERAL."""
    def __init__(self):
        self.answers: list = ["Do not use it in the last 3 months of pregnancy [1]."]
        self.verdicts: list = ["1: SUPPORTED"]
        self.answer_calls: list[str] = []
        self.judge_calls: list[str] = []

    @staticmethod
    def _next(items):
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, Exception):
            raise item
        return item

    def __call__(self, system, user):
        if system == rag.SYSTEM_PROMPT:
            self.answer_calls.append(user)
            return self._next(self.answers)
        if system == rag.JUDGE_PROMPT:
            self.judge_calls.append(user)
            return self._next(self.verdicts)
        return "GENERAL: Ask your pharmacist."


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(rag.llm, "generate", f)
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(c) for c in SOURCES])
    monkeypatch.setattr(config, "JUDGE_MODE", "risky")
    monkeypatch.setattr(config, "JUDGE_STRICT", True)
    return f


PREGNANCY_Q = "Can I take Nurofen Forte during pregnancy?"


def order(question):
    """Source numbers after re-ranking: {section: [n]}."""
    ranked = rag.rerank([dict(c) for c in SOURCES], question, set())
    return {c["section"]: i for i, c in enumerate(ranked, 1)}


# ------------------------------------------------------------------ when the judge runs
def test_supported_answer_on_a_risky_topic_stays_violet(fake):
    """Catches: the judge not being asked on a pregnancy question, or a SUPPORTED verdict rejected."""
    n = order(PREGNANCY_Q)["Pregnancy and breast-feeding"]
    fake.answers = [f"Do not use it in the last 3 months of pregnancy [{n}]."]
    res = rag.ask(PREGNANCY_Q)
    assert res["mode"] == "sourced" and res["meaning_check"] == "passed" and len(fake.judge_calls) == 1


def test_judge_sees_the_sentence_and_only_the_sources_it_cites(fake):
    """Catches: the judge comparing with the wrong passage (or with all passages, hiding a wrong citation)."""
    n = order(PREGNANCY_Q)["Pregnancy and breast-feeding"]
    fake.answers = [f"Do not use it in the last 3 months of pregnancy [{n}]."]
    rag.ask(PREGNANCY_Q)
    msg = fake.judge_calls[0]
    assert "Do not use it in the last 3 months of pregnancy" in msg and "Do not use in the last 3 months" in msg
    assert "every 6-8 hours" not in msg and "[" + str(n) + "]" not in msg.split("Claims:")[1].split("(cites")[0]


def test_ordinary_question_and_answer_are_not_judged(fake):
    """Catches: decision 1b ignored (an extra LLM call on every question = slower, more 429s)."""
    fake.answers = ["Adults: 1 tablet every 6-8 hours [1][2]."]
    res = rag.ask("How often can I take Nurofen Forte?")
    assert res["mode"] == "sourced" and res["meaning_check"] == "not_checked" and fake.judge_calls == []


def test_a_risky_word_in_the_answer_is_enough(fake):
    """Catches: "what is the dose?" answered with a pregnancy statement escaping the check."""
    fake.answers = ["Adults: 1 tablet every 6-8 hours; it is safe during pregnancy [1][2]."]
    fake.verdicts = ["1: CONTRADICTED"]
    res = rag.ask("What is the dose of Nurofen Forte?")
    assert fake.judge_calls and res["mode"] != "sourced"


@pytest.mark.parametrize("mode, judged", [("always", True), ("off", False)])
def test_judge_mode_setting(fake, monkeypatch, mode, judged):
    """Catches: JUDGE_MODE=always / off being ignored."""
    monkeypatch.setattr(config, "JUDGE_MODE", mode)
    fake.answers = ["Adults: 1 tablet every 6-8 hours [1][2]."]
    rag.ask("How often can I take Nurofen Forte?")
    assert bool(fake.judge_calls) is judged


@pytest.mark.parametrize("question", [
    "Can I take it while breastfeeding?", "Can my child take it?", "Can I drink alcohol with it?",
    "Can I drive after taking it?", "Can I take it with other medicines?", "I am allergic to aspirin",
    "I have kidney problems", "Is it bad for the liver?",
    "Чи можна під час вагітності?", "Чи можна дитині?", "Чи можна з алкоголем?",
    "Czy mogę w ciąży?", "Czy można karmiąc piersią?", "Czy mogę prowadzić samochód?",
    "Mohu v těhotenství?", "Mohu při kojení?", "¿Puedo tomarlo en el embarazo?", "¿Puedo conducir?",
    "Puis-je le prendre pendant la grossesse ?", "Puis-je boire de l'alcool ?",
])
def test_risky_topics_are_recognised_in_every_language(question):
    """Catches: a pregnancy/alcohol/children question in Polish or French escaping the judge."""
    assert rag.risky_topic(question)


@pytest.mark.parametrize("question", ["How often can I take it?", "What is the dose?", "Jak często mogę brać?",
                                      "Як часто можна приймати?", "¿Con qué frecuencia?", "Quelle est la dose ?"])
def test_ordinary_questions_are_not_risky(question):
    """Catches: every question being "risky" (then 1b would be the same as "always")."""
    assert not rag.risky_topic(question)


# ------------------------------------------------------------------ what a bad verdict does
def test_contradicted_sentence_gets_one_repair_and_is_judged_again(fake):
    """Catches: "safe in pregnancy [1]" (source: do not use) shown in violet, a repair prompt that doesn't
    name the sentence, or the repaired answer not being judged again."""
    n = order(PREGNANCY_Q)["Pregnancy and breast-feeding"]
    fake.answers = [f"It is safe during pregnancy [{n}].", f"Do not use it in the last 3 months of pregnancy [{n}]."]
    fake.verdicts = ["1: CONTRADICTED", "1: SUPPORTED"]
    res = rag.ask(PREGNANCY_Q)
    assert res["mode"] == "sourced" and res["meaning_check"] == "repaired"
    assert res["answer"] == f"Do not use it in the last 3 months of pregnancy[{n}]."
    assert len(fake.answer_calls) == 2 and "It is safe during pregnancy" in fake.answer_calls[1].split("Previous answer")[0]
    assert len(fake.judge_calls) == 2


def test_still_contradicted_after_the_repair_goes_orange(fake):
    """Catches: a second wrong answer accepted, or more than one repair (latency, cost)."""
    fake.answers = ["It is safe during pregnancy [1]."]
    fake.verdicts = ["1: CONTRADICTED"]
    res = rag.ask(PREGNANCY_Q)
    assert res["mode"] == "general" and len(fake.answer_calls) == 2


@pytest.mark.parametrize("strict, mode", [(True, "general"), (False, "sourced")])
def test_not_in_source_follows_the_strict_setting(fake, monkeypatch, strict, mode):
    """Catches: JUDGE_STRICT ignored. Strict (2a): NOT_IN_SOURCE is a problem; not strict: only counted."""
    monkeypatch.setattr(config, "JUDGE_STRICT", strict)
    fake.answers = ["It can be taken with orange juice during pregnancy [1]."]
    fake.verdicts = ["1: NOT_IN_SOURCE"]
    res = rag.ask(PREGNANCY_Q)
    assert res["mode"] == mode
    if not strict:
        assert res["unsupported_sentences"] == 1


def test_repaired_answer_with_a_wrong_number_goes_orange(fake):
    """Catches: the meaning repair sneaking in a number the source doesn't have."""
    fake.answers = ["It is safe during pregnancy [1].", "Do not use it in the last 5 months of pregnancy [1]."]
    fake.verdicts = ["1: CONTRADICTED", "1: SUPPORTED"]
    assert rag.ask(PREGNANCY_Q)["mode"] == "general"


def test_emergency_in_the_meaning_repair_gives_the_red_card(fake):
    """Catches: the red card lost when only the repaired answer says RISK: EMERGENCY."""
    fake.answers = ["It is safe during pregnancy [1].", "RISK: EMERGENCY\nDo not use it [1]."]
    fake.verdicts = ["1: CONTRADICTED"]
    assert rag.ask(PREGNANCY_Q)["mode"] == "emergency"


def test_failed_repair_call_goes_orange_because_the_answer_is_known_to_be_wrong(fake):
    """Catches: decision 3a (fail open) wrongly applied to a sentence the judge already rejected."""
    fake.answers = ["It is safe during pregnancy [1].", llm.LLMUnavailable("all providers failed")]
    fake.verdicts = ["1: CONTRADICTED"]
    assert rag.ask(PREGNANCY_Q)["mode"] == "general"


# ------------------------------------------------------------------ judge down or unreadable (decision 3a)
@pytest.mark.parametrize("verdict", [llm.LLMUnavailable("all providers failed"), "I think it is fine.", ""],
                         ids=["judge-down", "unreadable", "empty"])
def test_judge_down_or_unreadable_keeps_the_answer_violet(fake, verdict):
    """Catches: an outage of the judge turning every risky answer orange (3a: fail open, but logged)."""
    fake.verdicts = [verdict]
    res = rag.ask(PREGNANCY_Q)
    assert res["mode"] == "sourced" and res["meaning_check"] == "unavailable"


def test_claims_the_judge_skipped_are_not_treated_as_wrong(fake):
    """Catches: a partial verdict ("1: SUPPORTED" for 2 claims) turning the answer orange."""
    fake.answers = ["Do not use it in the last 3 months of pregnancy [1]. Adults: 1 tablet every 6-8 hours [2]."]
    fake.verdicts = ["1: SUPPORTED"]
    assert rag.ask(PREGNANCY_Q)["mode"] == "sourced"


@pytest.mark.parametrize("verdict", ["1: CONTRADICTED", "1) CONTRADICTED", "Claim 1: contradicted",
                                     "**1**: CONTRADICTED", "1 - NOT IN SOURCE", "1: NOT_IN_SOURCE."])
def test_verdict_formats_are_understood(verdict):
    """Catches: a judge that writes "1) contradicted" or markdown being read as "no verdict" (= fail open)."""
    assert rag.parse_verdicts(verdict, 1) in ({1: "contradicted"}, {1: "not_in_source"})


def test_no_judge_call_when_there_is_nothing_to_check(fake):
    """Catches: wasted judge calls on NOT_FOUND / red-card answers."""
    fake.answers = ["NOT_FOUND"]
    rag.ask(PREGNANCY_Q)
    fake.answers = ["RISK: EMERGENCY\nDo not use [1]."]
    rag.ask(PREGNANCY_Q)
    assert fake.judge_calls == []


def test_judge_prompt_allows_translation_and_paraphrase():
    """Catches: a strict judge rejecting a correct Ukrainian answer to a Polish leaflet (decision 2a is only
    safe if paraphrase/translation counts as SUPPORTED)."""
    prompt = " ".join(rag.JUDGE_PROMPT.split()).lower()
    assert "translation" in prompt and "paraphrase" in prompt and "supported" in prompt
