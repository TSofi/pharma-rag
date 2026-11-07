"""Group 6c: a citation must be TRUE, not only present.

Checked without any extra LLM call: every NUMBER in a cited sentence (dose, strength, interval, duration,
age, percentage) must appear in the source(s) that sentence cites. Numbers are where a wrong answer is
most dangerous ("take 3 tablets [1]" while source [1] says 2).

What happens on a mismatch (decision "A"): one repair attempt ("copy the numbers exactly from the source
you cite"); still wrong -> the answer is not shown as violet, it goes the "not found" way (orange).

Not false alarms: 2,5 = 2.5; 1 000 = 1000; "400 мг" vs "400 mg" (only numbers are compared); a number
word in the leaflet ("dwie tabletki") vs a digit in the answer; ranges "1-2"; a number the user wrote in
the question; list numbering "1." at the start of a line.
"""
import pytest

from app import rag

SOURCES = [
    {"drug": "Nurofen Forte", "section": "How to take", "url": "u", "score": 0.9,
     "text": "Adults: 1 tablet (400 mg) every 6-8 hours. Do not take more than 3 tablets in 24 hours."},
    {"drug": "Nurofen Forte", "section": "If you take more than you should", "url": "u", "score": 0.8,
     "text": "Children under 12 years: do not use. Weź dwie tabletki tylko po konsultacji z lekarzem."},
]


class FakeLLM:
    """Leaflet answers are taken from `answers` one per call (the last one repeats); triage gets GENERAL."""
    def __init__(self):
        self.answers: list[str] = ["Adults: 1 tablet every 6-8 hours [1]."]
        self.calls: list[str] = []

    def __call__(self, system, user):
        if system != rag.SYSTEM_PROMPT:
            return "GENERAL: Ask your pharmacist about the dose."
        self.calls.append(user)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(rag.llm, "generate", f)
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [dict(c) for c in SOURCES])
    return f


def ask(question="What is the dose of Nurofen Forte?"):
    return rag.ask(question)


# ------------------------------------------------------------------ the check itself
def test_numbers_that_match_the_cited_source_stay_violet(fake):
    """Catches: the check rejecting a correct, cited dose (a false alarm turns good answers orange)."""
    res = ask()
    assert res["mode"] == "sourced" and len(fake.calls) == 1


def test_wrong_dose_with_a_valid_looking_citation_is_never_violet(fake):
    """Catches: "3 tablets every 4 hours [1]" shown as confirmed by the leaflet, which says every 6-8 hours."""
    fake.answers = ["Adults: 1 tablet every 4 hours [1]."]
    res = ask()
    assert res["mode"] != "sourced" and res["found"] is False


def test_right_number_but_wrong_source_is_caught(fake):
    """Catches: a citation pointing to the wrong passage ("under 12 years" is in [2], not in [1])."""
    fake.answers = ["Do not give it to children under 12 years [1]."]
    assert ask()["mode"] != "sourced"


def test_a_number_found_in_any_of_the_cited_sources_is_fine(fake):
    """Catches: "[1][2]" being checked against the first source only."""
    fake.answers = ["Not for children under 12 years; adults take 1 tablet [1][2]."]
    assert ask()["mode"] == "sourced"


def test_wrong_number_gets_one_repair_attempt_that_can_fix_it(fake):
    """Catches: a fixable mistake thrown away, the repair prompt not naming the bad number, or a loop."""
    fake.answers = ["Do not take more than 4 tablets in 24 hours [1].",
                    "Do not take more than 3 tablets in 24 hours [1]."]
    res = ask()
    assert res["mode"] == "sourced" and res["answer"] == "Do not take more than 3 tablets in 24 hours[1]."
    assert len(fake.calls) == 2 and '"4"' in fake.calls[1], "the repair prompt names the wrong number"


def test_still_wrong_after_the_repair_goes_orange(fake):
    """Catches: a second wrong answer being accepted, or more than one repair call (latency, cost)."""
    fake.answers = ["Take 2 tablets every 4 hours [1]."]
    res = ask()
    assert res["mode"] == "general" and len(fake.calls) == 2


@pytest.mark.parametrize("answer", [
    "Adults: 1 tablet (400 мг) every 6-8 hours [1].",      # unit in another language
    "Adults: 1 tablet every 6–8 hours [1].",                # en dash range
    "Adults: 1 tablet every 6 to 8 hours [1].",             # range written out
    "Take 2 tablets only after asking a doctor [2].",       # "dwie" in the leaflet = 2
    "1) Adults: 1 tablet every 6-8 hours [1].\n2) Do not take more than 3 tablets in 24 hours [1].",  # list numbering is not a fact
    "Adults: one tablet every 6-8 hours [1].",              # a word in the answer is not checked
])
def test_no_false_alarm_for_the_same_number_written_differently(fake, answer):
    """Catches: a correct translated/reformatted answer being turned orange."""
    fake.answers = [answer]
    assert ask()["mode"] == "sourced"


@pytest.mark.parametrize("source_text, answer", [
    ("Take 2,5 ml twice a day.", "Take 2.5 ml twice a day [1]."),    # decimal comma vs point
    ("Maximum 1 000 mg per day.", "Maximum 1000 mg per day [1]."),   # thousands separator
    ("Maximum 1000 mg per day.", "Maximum 1 000 mg per day [1]."),
    ("Maximum 1,000 mg per day.", "Maximum 1000 mg per day [1]."),   # English thousands comma
    ("Take half a tablet.", "Take 0.5 tablet [1]."),                  # half = 0.5
])
def test_number_formats_are_compared_by_value(fake, monkeypatch, source_text, answer):
    """Catches: the same value in another format (EU vs US) counting as a wrong number."""
    monkeypatch.setattr(rag.store, "search", lambda *a, **k: [{**SOURCES[0], "text": source_text}])
    fake.answers = [answer]
    assert ask()["mode"] == "sourced"


def test_a_number_from_the_users_question_may_be_repeated(fake):
    """Catches: "you took 5 tablets" being rejected because 5 is the user's number, not the leaflet's."""
    fake.answers = ["RISK: CAUTION\nYou took 5 tablets; do not take more than 3 tablets in 24 hours [1]."]
    res = ask("I took 5 tablets today, is that too much?")
    assert res["mode"] == "sourced"


def test_numbers_inside_names_are_not_checked(fake):
    """Catches: "vitamin B12" or "CYP3A4" counted as doses."""
    fake.answers = ["Adults: 1 tablet every 6-8 hours; it is metabolised by CYP2C9 and does not contain B12 [1]."]
    assert ask()["mode"] == "sourced"


def test_uncited_sentence_is_not_number_checked(fake):
    """Catches: the number check overruling the uncited counter (an uncited sentence has no source to
    compare with; it is already reported in uncited_sentences)."""
    fake.answers = ["Adults: 1 tablet every 6-8 hours [1]. Ibuprofen was discovered in 1961."]
    res = ask()
    assert res["mode"] == "sourced" and res["uncited_sentences"] == 1


def test_emergency_in_the_number_repair_still_gives_the_red_card(fake):
    """Catches: the red card being lost when only the repaired answer marks the situation RISK: EMERGENCY."""
    fake.answers = ["Take 9 tablets [1].", "RISK: EMERGENCY\nDo not take more than 3 tablets in 24 hours [1]."]
    assert ask("I took too many tablets, what now?")["mode"] == "emergency"


def test_wrong_numbers_lists_each_problem_with_the_cited_and_the_real_source():
    """Catches: a repair prompt that can't tell the model WHAT to fix."""
    problems = rag.wrong_numbers("Do not give it to children under 12 years [1]. Take 7 tablets [1].", SOURCES)
    assert [p["number"] for p in problems] == ["12", "7"]
    assert problems[0]["cited"] == [1] and problems[0]["found_in"] == [2] and problems[1]["found_in"] == []
