"""Group 4: hybrid re-ranking (app/rag.py: keyword_stems, rerank, _CHILD).

Vector search returns 30 candidates; rerank() re-scores them with a stemmed keyword bonus ("alkoholem"
matches "alkoholu"), a preference for the product the user named, and a small penalty for children's
leaflets unless the question is about children. The top-6 after this step are what the LLM sees, so a
wrong order here means a wrong or missing passage in the answer. Each test names the mistake it catches.
"""
import pytest

from app import rag


def chunk(text: str, score: float, drug: str = "Nurofen", group: str = "g-adult") -> dict:
    return {"text": text, "score": score, "drug": drug, "group": group}


def order(chunks: list[dict], question: str, skip=frozenset(), prefer=None) -> list[str]:
    return [c["drug"] + ":" + c["text"][:12] for c in rag.rerank([dict(c) for c in chunks], question, set(skip), prefer=prefer)]


# ---------- keyword stems ----------
@pytest.mark.parametrize("question, skip, must_have, must_not_have", [
    ("Czy mogę brać Ibuprom z alkoholem w ciąży?", {"ibuprom"}, {"alkohol", "ciąż"}, {"mogę", "brać", "ibuprom"}),
    ("Чи можна приймати нурофен при вагітності?", {"нурофен"}, {"вагітнос"}, {"можна", "приймати", "нурофен"}),
    ("Mohu užívat Ibalgin s alkoholem v těhotenství?", {"ibalgin"}, {"alkohol", "těhotenst"}, {"mohu", "užívat", "ibalgin"}),
    ("¿Puedo tomar Espidifen con alcohol durante el embarazo?", {"espidifen"}, {"alcoh", "embara"}, {"puedo", "tomar", "espidifen"}),
    ("Puis-je prendre Doliprane avec de l'alcool pendant la grossesse ?", {"doliprane"}, {"alco", "grosses"}, {"prendre", "avec", "doliprane"}),
    ("Can I take Nurofen with alcohol during pregnancy?", {"nurofen"}, {"alcoh", "pregnan"}, {"take", "with", "nurofen"}),
])
def test_stems_keep_content_words_and_drop_stop_words_and_the_product_name(question, skip, must_have, must_not_have):
    """Catches: a language's stop words leaking into the stems (every passage with "można" gets a bonus),
    or the product name itself counting as a keyword (every passage of that leaflet gets +0.12)."""
    stems = rag.keyword_stems(question, skip)
    assert must_have <= stems, stems
    assert not (must_not_have & stems), stems
    assert all(len(s) >= 4 for s in stems)


@pytest.mark.parametrize("question_word, passage", [
    ("alkoholem", "Nie należy pić alkoholu podczas leczenia."),            # pl instrumental -> genitive
    ("вагітності", "Застосування у період вагітності або годування груддю."),   # uk (leaflet heading form)
    ("těhotenství", "Během těhotenství se nedoporučuje."),                # cs
    ("embarazo", "No tomar durante el embarazo."),                        # es
    ("grossesse", "Déconseillé pendant la grossesse."),                   # fr
])
def test_stem_of_an_inflected_question_word_is_found_in_the_passage(question_word, passage):
    """Catches: the stem being cut too little (full word never matches another ending)."""
    stems = rag.keyword_stems(question_word, set())
    assert any(st in passage.lower() for st in stems), stems


# ---------- stop words: "during" must not be a keyword ----------
@pytest.mark.parametrize("question, only_stem", [
    ("Can I take this medicine during pregnancy?", "pregnan"),
    ("Should I use it before or after meals while pregnant?", "pregna"),     # "meals" is NOT a stop word
    ("Czy mogę brać ten lek podczas ciąży?", "ciąż"),
    ("Чи можна приймати цей препарат під час вагітності?", "вагітнос"),
    ("Mohu užívat tento lék během těhotenství?", "těhotenst"),
    ("¿Puedo tomar este medicamento durante el embarazo?", "embara"),
    ("Puis-je prendre ce médicament pendant la grossesse ?", "grosses"),
])
def test_function_words_and_generic_medicine_words_are_not_keywords(question, only_stem):
    """Catches: "during" / "podczas" / "durante" / "pendant" / "během" / "цей" / "this" / "medicine" becoming
    keywords, so "take during meals" earns the same bonus as a pregnancy passage."""
    stems = rag.keyword_stems(question, set())
    assert stems == {only_stem} or stems == {only_stem, "meal"}, stems


def test_during_meals_passage_gets_no_bonus_for_a_pregnancy_question():
    """Catches: the stop list losing "during" -- the meals passage would tie with (or beat) the pregnancy one."""
    meals = chunk("Take the tablets during meals with a glass of water.", 0.60)
    preg = chunk("Do not use during the last 3 months of pregnancy.", 0.60)
    out = rag.rerank([meals, preg], "Can I take Nurofen during pregnancy?", {"nurofen"})
    assert out[0]["text"].startswith("Do not use") and out[0]["score"] == pytest.approx(0.72)
    assert out[1]["score"] == 0.60                                           # no bonus at all for "meals"


@pytest.mark.parametrize("word", [
    "pregnancy", "alcohol", "dose", "liver", "children", "syrup", "high", "blood",
    "ciąża", "alkohol", "dawka", "wątroba", "dzieci", "syrop",
    "вагітність", "алкоголь", "доза", "печінка", "діти", "сироп",
    "těhotenství", "dávka", "játra", "děti", "embarazo", "dosis", "hígado", "niños", "jarabe",
    "grossesse", "dose", "foie", "enfants", "sirop",
])
def test_medically_meaningful_words_are_never_stop_words(word):
    """Catches: someone "tidying" the stop list with a generic list (stopwords-iso has "high", "use",
    "problem"...) and silently weakening retrieval for real questions."""
    assert word not in rag._STOP


# ---------- Ukrainian grammatical cases (dictionary of word forms) ----------
@pytest.mark.parametrize("question_word, passage", [
    ("вагітності", "Вагітність та лактація"),                 # genitive in the question, nominative in the leaflet
    ("вагітність", "у період вагітності"),                   # and the other way round
    ("болю", "Головний біль, запаморочення."),                # о/і alternation in a short word
    ("біль", "для зменшення болю"),
    ("ночі", "приймати на ніч"),                              # о/і
    ("нирок", "захворювання нирки або печінки"),              # fleeting о
    ("дітям", "Дитина віком до 6 років"),                     # a completely different-looking form
])
def test_ukrainian_case_change_in_the_middle_of_the_word_still_matches(question_word, passage):
    """Catches: Ukrainian vowel alternation (вагітність / вагітності, біль / болю) being lost -- cutting
    2 letters from the end can never fix a letter in the middle, so the passage got no keyword bonus."""
    assert rag.keyword_hits(passage, rag.keyword_groups(question_word, set())) == 1


def test_ukrainian_forms_are_matched_as_whole_words():
    """Catches: the dictionary forms being matched as substrings -- "біль" would "find" "більше" (more),
    which appears in almost every dosing passage ("не більше 3 таблеток")."""
    for word in ("болю", "біль"):       # "біль" is also the old 4-letter stem -> must not match inside "більше"
        assert rag.keyword_hits("не більше 3 таблеток на добу", rag.keyword_groups(word, set())) == 0, word


def test_unknown_ukrainian_word_falls_back_to_the_old_stem_rule():
    """Catches: a brand name or typo the dictionary doesn't know crashing the lookup or losing its stem."""
    assert rag.uk_forms("нурофен") == set()
    groups = rag.keyword_groups("нурофен", set())
    assert rag.keyword_hits("Нурофен Форте таблетки", groups) == 1


def test_ambiguous_ukrainian_word_takes_forms_of_every_reading():
    """Catches: only the first dictionary reading being used ("солі" -> "сол" misses "сіль")."""
    assert {"сіль", "солі"} <= rag.uk_forms("солі")


def test_old_stem_rule_is_kept_for_known_ukrainian_words():
    """Catches: a known word being matched ONLY by its dictionary forms -- "алкоголь" would stop matching
    the adjective "алкогольних напоїв" (a different word for the dictionary, same thing for the user)."""
    groups = rag.keyword_groups("алкоголь", set())
    assert rag.keyword_hits("не вживати алкогольних напоїв", groups) == 1


def test_one_point_per_question_word_however_many_forms_appear():
    """Catches: a passage with "біль" and "болю" and "болем" earning 3 points for one question word,
    or the same word asked twice (вагітності / вагітність) counting twice."""
    groups = rag.keyword_groups("болю", set())
    assert rag.keyword_hits("біль у шлунку, головний біль, при болю", groups) == 1
    assert len(rag.keyword_groups("вагітності вагітність", set())) == 1


def test_ukrainian_pregnancy_passage_wins_on_equal_vector_score():
    """Catches: the whole chain (groups -> hits -> bonus) -- for "при вагітності" the passage headed
    "Вагітність" must beat a tablet-composition passage with the same vector score."""
    preg = chunk("Вагітність та лактація. Не рекомендується.", 0.60)
    pill = chunk("Таблетки, вкриті оболонкою, 200 мг.", 0.60)
    assert order([pill, preg], "Чи можна при вагітності?")[0].endswith("Вагітність т")


def test_non_ukrainian_stems_are_unchanged_by_the_dictionary():
    """Catches: the dictionary being applied to Latin-script words (it would return nothing and drop them)."""
    assert rag.uk_forms("alkoholem") == set()
    groups = rag.keyword_groups("alkoholem ciąży", {"ibuprom"})
    assert [g[0] for g in groups] == [{"alkohol"}, {"ciąż"}] and all(g[1] == set() for g in groups)


def test_stems_never_cut_below_four_letters_and_skip_short_words():
    """Catches: the stem rule cutting "dawka" to "daw" (matches "dawać", "sprzedawca"...) or 3-letter
    words like "ból" becoming stems that match everywhere."""
    assert rag.keyword_stems("dawka leku", set()) == {"dawk"}       # "leku" is a stop word
    assert rag.keyword_stems("ból głowy", set()) == {"głow"}        # "ból" too short


# ---------- keyword bonus ----------
def test_passage_with_the_question_keyword_beats_a_higher_vector_score():
    """Catches: the hybrid step being switched off -- a passage "about ibuprofen" would beat the one that
    actually mentions alcohol."""
    chunks = [chunk("Nurofen tablets contain ibuprofen 400 mg.", 0.66),
              chunk("Nie należy pić alkoholu podczas stosowania leku.", 0.60)]
    assert order(chunks, "Czy mogę brać Ibuprom z alkoholem?", {"ibuprom"})[0].endswith("Nie należy p")


def test_keyword_bonus_is_capped_at_three_hits():
    """Catches: a passage that happens to contain 6 question words getting +0.72 and burying everything."""
    q = "alkohol ciąża wątroba nerki serce żołądek"        # 6 content words
    six = chunk("alkohol ciąża wątroba nerki serce żołądek", 0.50)
    three = chunk("alkohol ciąża wątroba", 0.50)
    out = rag.rerank([six, three], q, set())
    assert out[0]["score"] == out[1]["score"] == pytest.approx(0.50 + 3 * 0.12)


def test_equal_scores_keep_the_vector_search_order():
    """Catches: a sort that shuffles ties, so the same question gives different sources on each call."""
    chunks = [chunk(f"passage {i}", 0.5) for i in range(5)]
    assert [c["text"] for c in rag.rerank(chunks, "question", set())] == [f"passage {i}" for i in range(5)]


def test_scores_are_rounded_to_three_decimals():
    """Catches: floating-point noise like 0.6200000001 showing up in the UI's score badge."""
    out = rag.rerank([chunk("alkohol", 0.1 + 0.2)], "alkohol", set())
    assert out[0]["score"] == 0.42


# ---------- children's leaflets ----------
ADULT = chunk("Nie pić alkoholu podczas leczenia.", 0.60, drug="Nurofen", group="g-adult")
KIDS = chunk("Nie pić alkoholu podczas leczenia.", 0.60, drug="Nurofen dla dzieci", group="g-kids")


def test_children_leaflet_ranks_below_an_equal_adult_leaflet_by_default():
    """Catches: the penalty being removed -- an adult asking about alcohol would get the children's
    syrup leaflet first (it exists for almost every substance)."""
    assert order([KIDS, ADULT], "Czy można z alkoholem?") == ["Nurofen:Nie pić alko", "Nurofen dla dzieci:Nie pić alko"]


@pytest.mark.parametrize("question", [
    "Nurofen dla dzieci a alkohol", "Нурофен для дітей доза", "Nurofen pro děti dávka",
    "Dosis para niños", "Dose pour enfants", "Dose for children", "pediatric dose",
])
def test_no_penalty_when_the_question_is_about_children(question):
    """Catches: a child word in one language missing from _CHILD, so a parent's question gets the adult leaflet."""
    assert order([KIDS, ADULT], question)[0].startswith("Nurofen dla dzieci")


@pytest.mark.parametrize("question", ["Nurofen Baby dawkowanie", "Ibufen Junior dawka", "syrop dla niemowląt",
                                      "сироп для малюків", "dosis pediátrica", "sirop pour bébé"])
def test_more_child_words_are_recognised(question):
    """Catches: common ways to say "for a small child" (baby, junior, niemowlę, малюк, pediátrico, bébé)
    not switching the penalty off."""
    assert order([KIDS, ADULT], question)[0].startswith("Nurofen dla dzieci")


def test_named_children_product_is_not_penalised():
    """Catches: the user explicitly asks about a children's product (its group is in `prefer`), the question
    has no child word, and the +0.04 preference loses to the -0.05 penalty -> the adult leaflet wins."""
    out = order([KIDS, ADULT], "Czy można z alkoholem?", prefer={"g-kids"})
    assert out[0].startswith("Nurofen dla dzieci")


def test_child_penalty_is_smaller_than_one_keyword_hit():
    """Catches: the penalty growing past the keyword bonus -- a children's leaflet that actually answers
    the question must still beat an adult leaflet that doesn't."""
    kids_hit = chunk("Nie pić alkoholu.", 0.60, drug="Nurofen dla dzieci", group="g-kids")
    adult_miss = chunk("Tabletki powlekane 400 mg.", 0.60, drug="Nurofen", group="g-adult")
    assert order([adult_miss, kids_hit], "alkohol")[0].startswith("Nurofen dla dzieci")


# ---------- preference for the product the user named ----------
def test_named_product_group_ranks_first_on_equal_passages():
    """Catches: the `prefer` bonus being dropped -- asking about Ibuprom would cite Nurofen's leaflet
    even when Ibuprom's own leaflet has the same passage."""
    a = chunk("Nie pić alkoholu.", 0.60, drug="Nurofen", group="g-nurofen")
    b = chunk("Nie pić alkoholu.", 0.60, drug="Ibuprom", group="g-ibuprom")
    assert order([a, b], "alkohol", prefer={"g-ibuprom"})[0].startswith("Ibuprom")
    assert order([a, b], "alkohol", prefer=None)[0].startswith("Nurofen")     # no preference: input order


def test_prefer_bonus_does_not_override_a_keyword_hit():
    """Catches: the preference growing so large that the named product's irrelevant passage beats
    another product's relevant one."""
    named_miss = chunk("Tabletki 400 mg.", 0.60, drug="Ibuprom", group="g-ibuprom")
    other_hit = chunk("Nie pić alkoholu.", 0.60, drug="Nurofen", group="g-nurofen")
    assert order([named_miss, other_hit], "alkohol", prefer={"g-ibuprom"})[0].startswith("Nurofen")
