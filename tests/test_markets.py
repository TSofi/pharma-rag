"""Group 3: product-name detection per country (app/markets.py).

Every product of a national register (~100k names) must be recognised in a question, including inflected
forms (Polish/Czech/Ukrainian endings) and US names typed in an EU market. The result decides which leaflet
groups Qdrant is filtered to, so a miss here means "not found" and a false hit means the wrong leaflet.
Each test names the mistake it catches. Fixture registers are tiny but use the real file format.
"""
import json

import pytest

from app import markets

IBU = {"inn": "Ibuprofenum", "en": "Ibuprofen", "form": "oral solid", "cands": [{"name": "x"}]}
PARA = {"inn": "Paracetamolum", "en": "Paracetamol", "form": "oral solid", "cands": [{"name": "x"}]}
DROTA = {"inn": "Drotaverini hydrochloridum", "en": "Drotaverine", "form": "oral solid", "cands": [{"name": "x"}]}
NO_LEAFLET = {"inn": "Zincum", "en": "Zinc", "form": "oral solid", "cands": []}

REGISTERS = {
    "pl": {"groups": {"pl:ibu": IBU, "pl:para": PARA, "pl:zinc": NO_LEAFLET},
           "products": [["Nurofen Forte", "pl:ibu", "https://pl/nurofen-forte", False],
                        ["Nurofen", "pl:ibu", "https://pl/nurofen", False],
                        ["Ibuprom", "pl:ibu", "https://pl/ibuprom", False],
                        ["Apap", "pl:para", "https://pl/apap", False],
                        ["Witamina C Forte", "pl:zinc", "https://pl/witc", False],
                        ["Alkohol etylowy 70%", "pl:zinc", "https://pl/etoh", True],
                        ["ACC", "pl:zinc", "https://pl/acc", False],                # 3 letters: too short
                        ["500 mg Magnez", "pl:zinc", "https://pl/mg500", False],   # first word is a number
                        ["3-Plus", "pl:zinc", "https://pl/3plus", False]]},        # a real product name
    "ua": {"groups": {"ua:ibu": {**IBU, "inn": "Ібупрофен"}, "ua:drota": {**DROTA, "inn": "Дротаверин"}},
           "products": [["НУРОФЄН®", "ua:ibu", "https://ua/nurofen", False],
                        ["НУРОФЄН ФОРТЕ", "ua:ibu", "https://ua/nurofen-forte", False],
                        ["НО-ШПА", "ua:drota", "https://ua/noshpa", False],
                        ["Алкоголь медичний", "ua:drota", "https://ua/etoh", True]]},
    "cz": {"groups": {"cz:ibu": IBU, "cz:para": PARA},
           "products": [["IBALGIN 400", "cz:ibu", "https://cz/ibalgin", False],
                        ["PARALEN", "cz:para", "https://cz/paralen", False]]},
    "es": {"groups": {"es:ibu": IBU, "es:para": PARA},
           "products": [["ESPIDIFEN 400 mg", "es:ibu", "https://es/espidifen", False],
                        ["GELOCATIL 1 g", "es:para", "https://es/gelocatil", True]]},
    "fr": {"groups": {"fr:ibu": IBU, "fr:para": PARA},
           "products": [["ADVIL 400 mg, comprimé", "fr:ibu", "https://fr/advil", False],
                        ["DOLIPRANE 1000 mg", "fr:para", "https://fr/doliprane", False]]},
}


@pytest.fixture
def registers(tmp_path, monkeypatch):
    """Write the fixture registers as data/markets/<cc>.json would be, and point the module at them."""
    for cc, data in REGISTERS.items():
        (tmp_path / f"{cc}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(markets, "MARKET_FILE", tmp_path / "{}.json")
    markets.index.cache_clear()
    yield tmp_path
    markets.index.cache_clear()


def asked(market: str, question: str) -> list[str]:
    return [m["asked"] for m in markets.detect(market, question)]


# ---------- each country recognises its own brands, with inflection ----------
@pytest.mark.parametrize("market, question, expected", [
    ("pl", "Czy mogę brać Ibupromu z alkoholem?", ["Ibuprom"]),
    ("pl", "Nurofenem a ciąża", ["Nurofen"]),
    ("ua", "Чи можна нурофеном з алкоголем?", ["Нурофєн"]),
    ("ua", "Скільки но-шпу можна пити на добу?", ["Но-шпа"]),
    ("cz", "Mohu užívat Ibalginu s alkoholem?", ["Ibalgin"]),
    ("cz", "Paralenem při horečce", ["Paralen"]),
    ("es", "¿Puedo tomar Espidifen con alcohol?", ["Espidifen"]),
    ("fr", "Puis-je prendre du Doliprane avec de l'alcool ?", ["Doliprane"]),
])
def test_brand_is_found_in_every_market_even_when_inflected(registers, market, question, expected):
    """Catches: the stem/prefix rule changing so "Ibupromu"/"нурофеном"/"Ibalginu" stop matching."""
    assert asked(market, question) == expected


@pytest.mark.parametrize("market, question", [
    ("pl", "Czy mogę prowadzić samochód po lekach?"),
    ("ua", "Чи можна пити каву з таблетками?"),
    ("cz", "Mohu řídit auto po lécích?"),
    ("es", "¿Puedo conducir después de tomar pastillas?"),
    ("fr", "Puis-je conduire après des comprimés ?"),
])
def test_question_without_a_product_finds_nothing(registers, market, question):
    """Catches: everyday words matching a product by prefix (a false hit filters Qdrant to the wrong leaflet)."""
    assert markets.detect(market, question) == []


def test_ukrainian_letter_є_in_the_register_matches_е_typed_by_users(registers):
    """Catches: dropping the є->е normalisation. Registers spell "НУРОФЄН"; people type "нурофен"."""
    found = markets.detect("ua", "нурофен дітям")
    assert found and found[0]["asked"] == "Нурофєн" and found[0]["url"] == "https://ua/nurofen"


def test_inflection_allows_at_most_three_extra_letters(registers):
    """Catches: the prefix rule becoming too loose (any word starting with "nurof" would match)."""
    assert asked("pl", "nurofenem") == ["Nurofen"]       # +2 letters: ok
    assert asked("pl", "nurofenowego") == []             # +5 letters: not an inflection


# ---------- US/English names in an EU market ----------
@pytest.mark.parametrize("question, expected_asked, expected_group", [
    ("Can I take Tylenol with alcohol?", "Tylenol", "pl:para"),      # US brand -> paracetamol group
    ("acetaminophen dose", "Acetaminophen", "pl:para"),              # US generic -> EU generic
    ("Is Advil safe in pregnancy?", "Advil", "pl:ibu"),
    ("ibuprofen and alcohol", "Ibuprofen", "pl:ibu"),                # substance name -> group
    ("Ibuprofenum 400", "Ibuprofenum", "pl:ibu"),                    # Latin INN -> group
])
def test_english_and_substance_names_map_to_the_group(registers, question, expected_asked, expected_group):
    """Catches: EN_ALIASES or the substance-name index being dropped; a tourist asking about Tylenol in
    Poland would get "not found" although the paracetamol leaflet is there. The user's own word is kept
    as the display name (the UI says 'you asked about Tylenol')."""
    found = markets.detect("pl", question)
    assert [m["asked"] for m in found] == [expected_asked]
    assert found[0]["groups"] == [expected_group]


# ---------- things that must never count as a product ----------
@pytest.mark.parametrize("market, word", [
    ("pl", "alkohol"), ("pl", "alkoholem"), ("pl", "witamina"),
    ("ua", "алкоголь"), ("ua", "вітаміни"),
    ("cz", "alkohol"), ("es", "alcohol"), ("fr", "alcool"),
])
def test_stop_words_are_never_products(registers, market, word):
    """Catches: a product literally named "Alkohol etylowy" or "Witamina C" hijacking every question
    that mentions alcohol or vitamins."""
    assert markets.detect(market, f"{word} question") == []
    assert word not in markets.index(market)["keys"]


def test_very_short_and_numeric_names_are_ignored(registers):
    """Catches: "ACC" (3 letters) matching "acc..." in any question, or "500 mg X" matching every dose."""
    keys = markets.index("pl")["keys"]
    assert "acc" not in keys and "500" not in keys
    assert "3-plus" in keys                                  # a real name with a digit is still a product


def test_store_brand_labelling_many_products_is_not_a_drug_name(registers):
    """Catches: "Equate"/"CVS Health" (hundreds of unrelated products) being treated as a product name,
    while a real generic like ibuprofen may legitimately cover many groups."""
    data = json.loads((registers / "pl.json").read_text(encoding="utf-8"))
    for i in range(markets.MAX_GROUPS_PER_NAME + 1):
        data["groups"][f"pl:g{i}"] = {**NO_LEAFLET, "inn": f"Sub{i}", "en": f"Sub{i}"}
        data["products"].append([f"Equate Product {i}", f"pl:g{i}", "https://pl/eq", False])
        data["products"].append([f"Ibuprofen Generic {i}", f"pl:g{i}", "https://pl/ibu", False])
    (registers / "pl.json").write_text(json.dumps(data), encoding="utf-8")
    markets.index.cache_clear()
    keys = markets.index("pl")["keys"]
    assert "equate" not in keys
    assert "ibuprofen" in keys and len(keys["ibuprofen"]) <= 25   # kept (it is a substance), list capped


# ---------- what detect() returns for the rest of the pipeline ----------
def test_display_name_url_and_main_group_come_from_the_shortest_product_name(registers):
    """Catches: the UI showing "Nurofen Forte" and linking to its leaflet when the user typed "Nurofen"."""
    m = markets.detect("pl", "Nurofen")[0]
    assert m["asked"] == "Nurofen" and m["url"] == "https://pl/nurofen" and m["main"] == "pl:ibu"
    assert m["rx"] is False and m["key"] == "nurofen"


def test_upper_case_register_names_are_prettified(registers):
    """Catches: the UI shouting "IBALGIN" / "НУРОФЄН" because registers store names in capitals."""
    assert asked("cz", "ibalgin") == ["Ibalgin"]
    assert asked("ua", "нурофен") == ["Нурофєн"]
    assert asked("fr", "advil") == ["Advil"]


def test_two_products_in_one_question_are_both_found_once(registers):
    """Catches: an interaction question ("Ibuprom with Apap?") filtering to only one leaflet, or the same
    product being listed twice when it appears twice."""
    found = markets.detect("pl", "Czy Ibuprom i Apap razem? Ibuprom rano, Apap wieczorem.")
    assert [m["asked"] for m in found] == ["Ibuprom", "Apap"]
    assert sorted(g for m in found for g in m["groups"]) == ["pl:ibu", "pl:para"]


# ---------- library (the "drugs in this country" drawer) ----------
def test_library_lists_one_entry_per_group_with_its_brands(registers):
    """Catches: the UI drawer listing a group that has no leaflet, or losing the brand names."""
    lib = markets.library("pl")
    assert [d["drug"] for d in lib] == ["ibuprofen", "paracetamol"]            # pl:zinc has no leaflet
    ibu = next(d for d in lib if d["group"] == "pl:ibu")
    assert ibu["brands"] == ["Nurofen Forte", "Nurofen", "Ibuprom"] and ibu["form"] == "oral solid"


def test_library_uses_the_indexed_flag_once_ingest_has_written_it(registers):
    """Catches: after ingest marks groups as indexed, the drawer still listing un-indexed ones."""
    data = json.loads((registers / "pl.json").read_text(encoding="utf-8"))
    data["groups"]["pl:ibu"]["indexed"] = True
    data["groups"]["pl:para"]["indexed"] = False
    (registers / "pl.json").write_text(json.dumps(data), encoding="utf-8")
    markets.index.cache_clear()
    assert [d["drug"] for d in markets.library("pl")] == ["ibuprofen"]


def test_missing_register_file_gives_an_empty_index_not_a_crash(registers):
    """Catches: a market listed in config before its register exists crashing startup (/api/health)."""
    assert markets.index("at") == {"keys": {}, "prefix": {}, "display": {}, "library": []}
    assert markets.detect("at", "Nurofen") == [] and markets.library("at") == []


# ---------- the helpers the index is built from ----------
@pytest.mark.parametrize("name, key", [
    ("NUROFEN FORTE, 400 mg, tabletki", "nurofen"),
    ("Но-шпа® 40 мг", "но-шпа"),
    ("IBALGIN 400/tablety", "ibalgin"),
    ("Nurofen (forte)", "nurofen"),
])
def test_first_word_is_the_lookup_key(name, key):
    """Catches: a register name with punctuation producing keys like "nurofen," that never match."""
    assert markets.first_word(name) == key


@pytest.mark.parametrize("key, stem", [("nurofen", "nurofe"), ("ibuprom", "ibupro"), ("polopiryna", "polopiry"),
                                       ("apap", "apap"), ("no-shpa", "no-shp")])
def test_stem_cuts_one_or_two_letters_depending_on_length(key, stem):
    """Catches: the stem rule changing so short names lose letters ("apap" -> "apa" would match "apart")."""
    assert markets.stem(key) == stem
