"""Group 6c-2: the number check (test_citation_truth.py) understands UNITS.

The leaflet says "1 g", the answer says "1000 mg": the same amount, so it must not be treated as a wrong
number. A really different amount ("2 g" vs "1000 mg") must still be caught. Mass: mcg / µg / мкг, mg / мг,
g / г, kg / кг. Volume: ml / мл, l / л. Only the SOURCE side is converted: the answer's number must match
the leaflet's amount in some unit.
"""
import pytest

from app import rag


def src(text):
    return [{"drug": "Paracetamol", "section": "How to take", "text": text}]


def wrong(answer, text):
    return [p["number"] for p in rag.wrong_numbers(answer, src(text))]


@pytest.mark.parametrize("leaflet, answer", [
    ("Maximum 1 g per dose.", "Maximum 1000 mg per dose [1]."),
    ("Maximum 1000 mg per dose.", "Maximum 1 g per dose [1]."),
    ("Maximum 4 g per day.", "Maximum 4000 mg per day [1]."),
    ("Each tablet contains 0,5 g.", "Each tablet contains 500 mg [1]."),
    ("Each tablet contains 500 mg.", "Each tablet contains 0.5 g [1]."),
    ("Maximum 2,01 g per day.", "Maximum 2010 mg per day [1]."),         # no float noise (2.01 * 1000 != 2010.0)
    ("Contains 500 µg.", "Contains 0.5 mg [1]."),
    ("Contains 250 mcg.", "Contains 0.25 mg [1]."),
    ("Максимальна доза 1 г.", "Максимальна доза 1000 мг [1]."),          # Ukrainian units
    ("Разова доза 500 мг.", "Single dose 0.5 g [1]."),                    # UA leaflet, EN answer
    ("Містить 400 мкг.", "Contains 0.4 mg [1]."),
    ("Maksymalnie 1 g na dobę.", "Maksymalnie 1000 mg na dobę [1]."),
    ("Flacon de 1 l.", "Flacon de 1000 ml [1]."),
    ("Lahvička 100 ml.", "Lahvička 0,1 l [1]."),
    ("Dosis 5 mL.", "Dosis 5 ml [1]."),                                   # unit case
])
def test_the_same_amount_in_another_unit_is_not_a_wrong_number(leaflet, answer):
    """Catches: a correct answer turned orange only because the model wrote the amount in another unit."""
    assert wrong(answer, leaflet) == []


@pytest.mark.parametrize("leaflet, answer, bad", [
    ("Maximum 1 g per dose.", "Maximum 2000 mg per dose [1].", "2000"),
    ("Maximum 4 g per day.", "Maximum 400 mg per day [1].", "400"),
    ("Contains 500 µg.", "Contains 500 mg [1].", "500"),     # µg vs mg: 1000x more, NOT the same amount
    ("Максимальна доза 1 г.", "Максимальна доза 100 мг [1].", "100"),
])
def test_a_different_amount_is_still_caught(leaflet, answer, bad):
    """Catches: the conversion making the check so loose that a wrong dose passes."""
    assert wrong(answer, leaflet) == [bad]


def test_units_are_not_guessed_from_ordinary_words():
    """Catches: "2 glasses" read as "2 g" (then "2000 mg" would wrongly pass)."""
    assert wrong("Take 2000 mg [1].", "Take it with 2 glasses of water.") == ["2000"]


def test_mass_is_never_converted_into_volume():
    """Catches: "1 g" accepting "1000 ml" (different quantities)."""
    assert wrong("Take 1000 ml [1].", "Take 1 g.") == ["1000"]


def test_concentration_keeps_working():
    """Catches: "100 mg/5 ml" breaking the check (the unit after "/" is not a number of its own)."""
    assert wrong("The syrup contains 100 mg in 5 ml [1].", "Syrup: 100 mg/5 ml.") == []
    assert wrong("The syrup contains 0.1 g in 5 ml [1].", "Syrup: 100 mg/5 ml.") == []
