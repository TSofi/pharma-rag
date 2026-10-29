"""The RAG pipeline: question -> retrieve relevant label sections -> LLM answer with citations."""
import difflib
import json
import random
import re
import time
from functools import lru_cache

from . import config, llm, markets, store

SYSTEM_PROMPT = """You are a drug-information assistant for pharmacists.
Answer ONLY using the numbered sources provided (excerpts from official drug labels / patient leaflets).
Rules:
- Cite every factual sentence with the source number in plain ASCII square brackets, e.g. [1] or [2][3].
  Cite each source separately: [2][3], never [2-3] or [2, 3]. Cite only the 1-3 sources that really
  contain the fact, not all of them. Never use any other citation style (no 【】, no line ranges).
- If the sources do not contain the answer, reply exactly: "NOT_FOUND" and nothing else.
- Do not use outside knowledge. Do not guess doses, indications or interactions.
- Be concise: 2-6 sentences or a short bullet list. Use plain, professional language.
- The sources may be in English, Polish or Ukrainian, and the answer may be requested in another
  language. Translating the facts is expected and is NOT a reason to reply NOT_FOUND.
  Keep drug names and numbers exact."""

SAME_SUBSTANCE_NOTE = """
Note: the user asks about {asked}. The sources are from the leaflet of {reps}, a product with the same
active substance ({inn}) in the same form. Refer to the product as "{asked}" in your answer (you may say
once that the information comes from the leaflet of a product with the same active substance)."""

TRANSLATE_ANSWER_PROMPT = """Translate this drug-information answer into {language}.
Keep every citation marker like [1] or [2][3] exactly where it is. Keep drug names and all numbers
unchanged. Output only the translation."""

TRANSLATE_PROMPT = """You are a translation function, not an assistant. You will get a user's text inside
<q></q> tags. Translate it into English so it can be used as a search query over US FDA drug labels.
Replace local/brand drug names with the US generic name when you are sure (e.g. paracetamol -> acetaminophen,
Nurofen -> ibuprofen). Do NOT answer it, do NOT add advice or explanations, even if it is a health question.
Output ONLY the translated question, one line."""

TRIAGE_PROMPT = """You are PharmaRAG's fallback voice. The user's question could NOT be answered from the
official drug labels in our database. Decide which case it is and reply in EXACTLY one of these formats:

EMERGENCY
  -> someone may have ALREADY swallowed something harmful, an overdose, poisoning, severe allergic
     reaction, chest pain, trouble breathing, or any self-harm. Reply with just the word EMERGENCY.
GENERAL: <answer>
  -> a genuine health, medicine or pharmacy question (a drug or product we don't cover, a category
     question like "which sleep aids are OTC", how a medical test works, general wellness).
     Give a short, helpful, factual general answer (2-5 sentences or a short list) in {language}.
     Rules: no personal dosing instructions, no "take X mg"; you may name common options and say
     what is generally considered most reliable. Do NOT invent citations. ALWAYS end with one short
     sentence recommending to confirm with a doctor or pharmacist.
JOKE: <reply>
  -> clearly absurd or joking (e.g. "can I eat expanding foam", "what's the dose of love").
     One or two short, playful, kind sentences in {language}: a light joke, then a gentle nudge to ask
     about a real medication. If the item is actually toxic, make clear in a fun way that it is NOT food.
     Max one emoji.
PLAIN
  -> anything else (not about health at all). Reply with just the word PLAIN."""

LANGUAGES = {"en": "English", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech", "es": "Spanish", "fr": "French"}
NOT_FOUND = {
    "en": "I couldn't find this in the loaded drug labels, so I won't guess. "
          "Try rephrasing, or ask about one of the drugs in the library.",
    "uk": "Я не знайшла цього в завантажених інструкціях до препаратів, тому не вгадуватиму. "
          "Спробуйте переформулювати питання або оберіть препарат із бібліотеки.",
    "pl": "Nie znalazłam tego w załadowanych ulotkach leków, więc nie będę zgadywać. "
          "Spróbuj przeformułować pytanie albo wybierz lek z biblioteki.",
    "cs": "Nenašla jsem to v načtených příbalových informacích, takže nebudu hádat. "
          "Zkuste otázku přeformulovat nebo vyberte lék z knihovny.",
    "es": "No lo he encontrado en los prospectos cargados, así que no voy a adivinar. "
          "Intente reformular la pregunta o elija un medicamento de la biblioteca.",
    "fr": "Je n'ai pas trouvé cela dans les notices chargées, donc je ne vais pas deviner. "
          "Essayez de reformuler la question ou choisissez un médicament dans la bibliothèque.",
}
EMERGENCY = {
    "en": "This may be an emergency. Call your local emergency number (112 in the EU, 911 in the US) "
          "or a poison control centre now. Don't wait for an online answer.",
    "uk": "Це може бути невідкладний стан. Негайно телефонуйте на екстрений номер (112 в ЄС, 103 в Україні) "
          "або в токсикологічний центр. Не чекайте відповіді онлайн.",
    "pl": "To może być stan nagły. Zadzwoń natychmiast na numer alarmowy 112 lub do ośrodka toksykologicznego. "
          "Nie czekaj na odpowiedź online.",
    "cs": "Může jít o stav ohrožení života. Okamžitě volejte tísňovou linku 112 (nebo 155) "
          "nebo Toxikologické informační středisko. Nečekejte na odpověď online.",
    "es": "Esto puede ser una emergencia. Llame ahora al 112 o al Servicio de Información Toxicológica. "
          "No espere una respuesta en línea.",
    "fr": "Il peut s'agir d'une urgence. Appelez immédiatement le 112 (ou le 15, SAMU) "
          "ou un centre antipoison. N'attendez pas une réponse en ligne.",
}
_PL_WORDS = {"jest", "czy", "jaka", "jaki", "jakie", "dawka", "dawkowanie", "lek", "leku", "można", "mozna",
             "się", "sie", "na", "przy", "ciąży", "ciazy", "skutki", "uboczne", "działa", "dziala", "dla",
             "przeciwwskazania", "dlaczego", "stosuje", "stosowac", "moge", "brac", "jak", "ile", "kiedy", "czym",
             "mam", "dziecko", "dzieci", "tabletki", "tabletke", "po", "ze", "bez", "razem", "leki", "lekow"}
# Common English words: without any of them, "en" is only a guess (e.g. Polish typed without ą, ę, ł).
_EN_WORDS = {"the", "is", "can", "what", "how", "with", "for", "of", "and", "i", "my", "do", "does", "take",
             "are", "it", "to", "in", "on", "while", "when", "should", "side", "effects", "during", "pregnancy"}

NOT_FOUND_MSG = NOT_FOUND["en"]


_CS_WORDS = {"je", "jak", "mohu", "můžu", "lék", "léku", "užívat", "těhotenství", "dávka", "nežádoucí", "při", "se", "s", "a"}
_ES_WORDS = {"puedo", "tomar", "con", "el", "la", "es", "qué", "cómo", "embarazo", "dosis", "efectos", "para", "los", "las"}
_FR_WORDS = {"je", "peux", "prendre", "avec", "le", "la", "est", "quels", "comment", "grossesse", "dose", "effets", "pour", "les"}


def detect_language(text: str) -> str:
    """Cheap heuristic, no extra API call: Cyrillic -> Ukrainian, then letters/words typical of each language."""
    if re.search(r"[а-яіїєґА-ЯІЇЄҐ]", text):
        return "uk"
    low = text.lower()
    words = set(re.findall(r"[a-záéíóúñüàâçèêëîïôœùûąćęłńóśźżčďěňřšťůý]+", low))
    if re.search(r"[ąęłńśźż]", low) or len(words & _PL_WORDS) >= 2:
        return "pl"
    if re.search(r"[ěřůčšžďťň]", low) or len(words & _CS_WORDS) >= 3:
        return "cs"
    if re.search(r"[ñ¿¡]", low) or len(words & _ES_WORDS) >= 2:
        return "es"
    if re.search(r"[àâçèêëîïôœùû]", low) or len(words & _FR_WORDS) >= 2:
        return "fr"
    return "en" if words & _EN_WORDS else "auto"  # "auto": let the LLM answer in the question's own language


def lang_name(code: str) -> str:
    return LANGUAGES.get(code, "the same language as the user's question")


@lru_cache(maxsize=1)
def drug_aliases() -> dict[str, str]:
    """Map every lowercase generic AND brand name -> canonical generic name.
    Built from the same data file we ingested, so no extra database call is needed."""
    aliases: dict[str, str] = {}
    if not config.DATA_FILE.exists():
        return aliases
    with open(config.DATA_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            aliases[rec["drug"].lower()] = rec["drug"]
            for b in rec.get("brands", []):
                aliases[b.lower()] = rec["drug"]
    return aliases


@lru_cache(maxsize=1)
def drug_brands() -> dict[str, list[str]]:
    brands: dict[str, list[str]] = {}
    if config.DATA_FILE.exists():
        with open(config.DATA_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                brands[rec["drug"]] = rec.get("brands", [])
    return brands


def library(market: str = "us") -> list[dict]:
    """Drugs available in the knowledge base (for the UI)."""
    if market != "us" or config.US_FULL:
        return markets.library(market)
    return [{"drug": d, "brands": b} for d, b in sorted(drug_brands().items())]


def detect_drugs(question: str) -> list[str]:
    """Find drug names mentioned in the question (whole-word match on generic or brand names).
    If found, retrieval is restricted to those drugs -> far fewer irrelevant chunks."""
    return detect_drugs_fuzzy(question)[0]


def detect_drugs_fuzzy(question: str) -> tuple[list[str], list[dict]]:
    """Exact match first; words (6+ letters, so everyday words like "leave" don't match "Aleve")
    that look like a misspelled drug name ("ibumporfen") are
    matched with difflib's similarity ratio. Returns (drugs, corrections)."""
    q = question.lower()
    aliases = drug_aliases()
    found = {drug for alias, drug in aliases.items() if re.search(rf"\b{re.escape(alias)}\b", q)}
    corrections = []
    known_words = set(aliases)
    for word in set(re.findall(r"[a-z][a-z-]{5,}", q)) - known_words:
        match = difflib.get_close_matches(word, known_words, n=1, cutoff=0.8)
        if not match and len(word) >= 7:
            # Long names tolerate a bit more damage ("ibuprol" -> ibuprofen), but only against long names,
            # otherwise everyday words slip through ("morning" ~ "Motrin").
            match = difflib.get_close_matches(word, [a for a in known_words if len(a) >= 7], n=1, cutoff=0.75)
        if match and aliases[match[0]] not in found:
            found.add(aliases[match[0]])
            corrections.append({"typed": word, "matched": aliases[match[0]]})
    return sorted(found), corrections


FACTS_FILE = config.ROOT / "data" / "facts.json"


@lru_cache(maxsize=1)
def _facts() -> list[dict]:
    """Short, curated 'did you know?' facts about drugs in the library (each with a source link).
    Kept separate from the label data: they are for the loading screen, never used to answer questions."""
    return json.loads(FACTS_FILE.read_text(encoding="utf-8")) if FACTS_FILE.exists() else []


def random_fact(seen: set[str] | None = None, lang: str = "en") -> dict:
    """A random fact the viewer hasn't seen yet in this session (by drug), in the UI language."""
    facts = _facts()
    fresh = [f for f in facts if f["drug"] not in (seen or set())]
    if not facts:
        return {}
    f = random.choice(fresh or facts)
    return {"drug": f["drug"], "text": f.get(f"text_{lang}", f["text"]), "source": f["source"], "url": f["url"]}


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {c['drug']} — {c['section']}\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )


# One knowledge base per country ("market"). An EU country = EMA centrally authorised products
# (valid in every EU state) + that country's national register.
MARKETS = {k: v["source"] for k, v in config.MARKETS.items()}


# Words that carry no information for keyword matching: question words, pronouns, prepositions, helper
# verbs and the generic "medicine" nouns that appear in every leaflet passage. Only words of 4+ letters
# matter here (shorter ones never become keywords). Hand-curated on purpose: ready-made stop lists contain
# words that matter medically ("high", "low", "use", "problem"). Form words (tablet, syrup) are NOT here.
_STOP_BY_LANG = {
    "en": """this that these those what when where which while during before after with without about
            from into over under between through because should could would might must shall will
            have having been being does doing take taking took taken give given make made need needs
            also only very more most some such than then them they their there here your yours mine
            medicine medicines medication medications drug drugs product remedy okay safe allowed""",
    "pl": """czy mogę moge można mozna brać brac stosować stosowac przyjmować przyjmowac używać uzywac
            jest być bedzie będzie jaki jaka jakie jakich które który która ktore ktory ktora kiedy gdzie
            podczas przed przez między miedzy oraz jeśli jesli jeżeli jezeli żeby zeby także takze również
            rowniez tylko bardzo jego jej tego tej tych ten tym razem wraz leku lekiem leków lekow leki
            lekarstwo lekarstwa preparat preparatu bezpiecznie bezpieczny wolno powinienem powinnam""",
    "uk": """можна потрібно треба приймати прийняти пити вживати застосовувати використовувати разом
            який яка яке які якої якого після перед через разі коли куди звідки також тільки лише дуже
            цей цього цьому цією цієї цим тому того тієї вона вони воно їхній його може можуть мені
            препарат препарату препаратом препарати ліки ліків ліками лікарський безпечно дозволено""",
    "cs": """mohu můžu můžeme užívat uzivat používat pouzivat brát tento tato toto tyto tohoto této
            během behem před pred kdy kde jaký jaká jaké který která které také pouze velmi jeho její
            jestli pokud nebo proto tedy léku leku lékem lekem léky leky přípravek pripravek
            přípravku bezpečně bezpecne smím""",
    "es": """puedo puede pueden debo debe tomar tomarlo usar utilizar este esta esto estos estas
            durante antes después despues mientras para como cuál cual cuáles cuales cuándo cuando
            dónde donde también tambien sólo solo muy pero porque desde hasta entre sobre
            medicamento medicamentos fármaco farmaco producto seguro""",
    "fr": """puis peux peut pouvez prendre utiliser avec sans pour pendant avant après apres lors
            quel quels quelle quelles quand comment pourquoi cette ceci cela ceux celle celles leur leurs
            aussi très tres mais donc alors parce entre dans vers chez
            médicament medicament médicaments medicaments produit sûr""",
}
_STOP = {w for words in _STOP_BY_LANG.values() for w in words.split()}


_UK_WORD = re.compile(r"[а-яіїєґ']+")


@lru_cache(maxsize=1)
def _uk_morph():
    """The Ukrainian dictionary (pymorphy3, ~20 MB RAM, loaded once on first use). None if not installed,
    so the app keeps working with the plain stem rule only."""
    try:
        import pymorphy3
        return pymorphy3.MorphAnalyzer(lang="uk")
    except Exception:  # noqa: BLE001 -- missing package or dictionary: degrade, never crash
        return None


@lru_cache(maxsize=4096)
def uk_forms(word: str) -> frozenset[str]:
    """Every grammatical form of a Ukrainian word the dictionary knows (болю -> біль, болі, болем, ...),
    from every reading (солі -> сіль and сол). Empty for Latin-script words and for words the dictionary
    doesn't know (brand names, typos), which then rely on the stem rule alone."""
    morph = _uk_morph()
    if morph is None or not _UK_WORD.fullmatch(word) or not morph.word_is_known(word):
        return frozenset()
    return frozenset(f.word for p in morph.parse(word) for f in p.lexeme)


def _stem(word: str) -> str:
    return word[:max(4, len(word) - 2)]


def keyword_groups(text: str, skip: set[str]) -> list[tuple[set[str], set[str]]]:
    """One entry per content word of the question: (stems, forms).
    - stems: the word cut to a stem so Polish/Ukrainian endings still match (alkoholem -> alkohol);
    - forms: all grammatical forms from the Ukrainian dictionary (вагітності -> вагітність, ...), so a
      case change in the middle of the word (біль / болю) is found too. Empty for other languages.
    Two question words that are forms of the same word (вагітності, вагітність) make ONE entry."""
    groups: dict[object, tuple[set[str], set[str]]] = {}
    for w in re.findall(r"[^\W\d_]{4,}", text.lower()):
        if w in _STOP or w in skip:
            continue
        forms = uk_forms(w)
        key = forms if forms else _stem(w)
        stems, fs = groups.setdefault(key, (set(), set()))
        stems.add(_stem(w))
        fs |= forms
    return list(groups.values())


def keyword_stems(text: str, skip: set[str]) -> set[str]:
    """Content words of the question, cut to a stem (alkoholem -> alkohol, вагітності -> вагітнос)."""
    return {st for stems, _ in keyword_groups(text, skip) for st in stems}


def keyword_hits(text: str, groups: list[tuple[set[str], set[str]]]) -> int:
    """How many question words a passage contains: a stem anywhere in the text, or a dictionary form as a
    whole word. One point per question word, however often it appears.
    A 4-letter stem is the whole (uncut) word and matches inside longer words ("біль" inside "більше" = more),
    so when the dictionary knows the word, such short stems are skipped and its forms decide alone."""
    low = text.lower()
    words = set(re.findall(r"[^\W\d_]+", low))
    return sum(1 for stems, forms in groups
               if any(st in low for st in stems if len(st) > 4 or not forms) or bool(forms & words))


_CHILD = re.compile(r"dzieci|dziecka|niemowl|junior|baby|children|child|kids|infant|pediatric|дітей|дитяч|малюк|немовл"
                    r"|niños|infantil|pediátr|bebé|lactante|enfant|pédiatr|bébé|nourrisson|děti|dětsk|kojenc|batol", re.I)


def rerank(chunks: list[dict], question: str, skip: set[str], bonus: float = 0.12,
           prefer: set[str] | None = None) -> list[dict]:
    """Keyword bonus + a preference for the product the user actually named (its own group), and a small
    penalty for children's products unless the question is about children."""
    groups = keyword_groups(question, skip)
    about_kids = bool(_CHILD.search(question))
    for c in chunks:
        extra = bonus * min(keyword_hits(c["text"], groups), 3)
        if prefer and c.get("group") in prefer:
            extra += 0.04
        if not about_kids and _CHILD.search(c.get("drug", "")) and c.get("group") not in (prefer or ()):
            # adults' leaflet first, but a children's leaflet that answers the question stays in,
            # and a children's product the user explicitly named is never pushed down
            extra -= 0.05
        c["score"] = round(c["score"] + extra, 3)
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


def normalize_citations(text: str, n_sources: int | None = None) -> str:
    """Some models cite in their own style: 【2】, 【1†L1-L3】, [1, 2], [1-3] or [source 2]. Convert all of
    that to plain [n] markers, which the UI turns into clickable source badges. A range is expanded and
    clipped to the real sources (with 6 sources [1-20] -> [1]..[6], [4-2] -> [2][3][4]); a marker is never
    deleted as a whole, so a sentence does not silently lose its source."""
    cap = n_sources if n_sources else 30  # without a source count, still never explode on [1-9999]

    def expand(m):
        lo, hi = sorted((int(m[1]), int(m[2])))
        return "".join(f"[{i}]" for i in range(lo, min(hi, cap) + 1))
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[\s*(?:sources?|src)\s*(\d+)\s*\]", r"[\1]", text, flags=re.I)
    text = re.sub(r"\[(\d+)\s*[-–]\s*(\d+)\]", expand, text)
    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: "".join(f"[{n.strip()}]" for n in m[1].split(",")), text)
    text = re.sub(r"\s+(\[\d+\])", r"\1", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text)  # a marker that vanished leaves "alcohol ." behind


def uncited_sentences(text: str) -> int:
    """How many sentences / bullet lines of an answer carry no [n] marker -- an honesty signal for the logs
    and the UI (the text is kept: cutting the model's words could change the meaning)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ¿¡])|\n+", text)  # a capital letter after ". " = new sentence
    return sum(1 for p in parts if re.search(r"\w", p) and not re.search(r"\[\d+\]", p))


def translate_query(question: str) -> str:
    """Translate to English for retrieval. Guard against chatty models that ANSWER instead of translating:
    if the output is much longer than the question, it's not a translation -- search with the original."""
    out = llm.generate(TRANSLATE_PROMPT, f"<q>{question}</q>").strip().strip('"').splitlines()[0:1]
    out = out[0].strip() if out else ""
    return out if out and len(out) <= max(160, 3 * len(question)) else question


def fallback(question: str, out_lang: str) -> tuple[str, str]:
    """When the labels can't answer. Returns (text, mode):
    "general"   - a short general answer, clearly marked in the UI as NOT from the labels;
    "playful"   - a joke for absurd questions;
    "emergency" - a fixed, serious message (never generated, never a joke);
    "none"      - the plain "not found" message."""
    try:
        reply = llm.generate(TRIAGE_PROMPT.format(language=lang_name(out_lang)),
                             f"Question: {question}\n\nWrite any answer in {lang_name(out_lang)} "
                             f"(regardless of the question's language).").strip()
    except Exception:  # noqa: BLE001 -- the fallback is optional; never fail the request because of it
        return NOT_FOUND.get(out_lang, NOT_FOUND["en"]), "none"
    head = reply[:20].upper()
    if "EMERGENCY" in head:
        return EMERGENCY.get(out_lang, EMERGENCY["en"]), "emergency"
    for tag, mode, limit in (("GENERAL:", "general", 1500), ("JOKE:", "playful", 400)):
        if reply.upper().startswith(tag):
            text = normalize_citations(reply[len(tag):]).strip()
            text = re.sub(r"\[\d+\]", "", text)  # a general answer has no sources, so no citations
            if text and len(text) <= limit:
                return text, mode
    return NOT_FOUND.get(out_lang, NOT_FOUND["en"]), "none"


def ask(question: str, top_k: int | None = None, answer_language: str = "auto", market: str = "us") -> dict:
    """The question can be in any supported language; the MARKET decides which labels are searched,
    and the answer is written in `answer_language` ("auto" = same language as the question)."""
    if market not in MARKETS:
        raise ValueError(f"Market '{market}' is not available yet")
    top_k = top_k or config.TOP_K
    t_start = time.perf_counter()
    timings: dict[str, int] = {}

    def timed(name: str, fn):
        """Run fn() and add its duration (ms) to timings[name] -- a tiny per-stage profiler."""
        t = time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = timings.get(name, 0) + round((time.perf_counter() - t) * 1000)
    lang = detect_language(question)                                   # language of the QUESTION
    out_lang = lang if answer_language == "auto" else answer_language  # language of the ANSWER

    # 0) QUERY REWRITING (US only): the FDA labels and their embedding model are English, so a
    #    Ukrainian/Polish question is translated first. PL/UA use a multilingual model: no translation.
    if market == "us":
        search_q = question if lang == "en" else \
            timed("translate_ms", lambda: translate_query(question)) or question
        if config.US_FULL:  # every FDA-labelled product by name (brand names survive translation)
            corrections = []
            mentions = markets.detect("us", f"{question} {search_q}")
            drugs = sorted({g for m in mentions for g in m["groups"]})
            field, detected = "group", [m["asked"] for m in mentions]
        else:
            drugs, corrections = detect_drugs_fuzzy(search_q)
            mentions, field = [], "drug"
            detected = drugs
    else:
        search_q, corrections = question, []
        mentions = markets.detect(market, question)   # every product of the national register by name
        drugs = sorted({g for m in mentions for g in m["groups"]})
        field, detected = "group", [m["asked"] for m in mentions]
    base = {"detected_drugs": detected, "corrections": corrections, "language": lang, "answer_language": out_lang,
            "market": market, "search_query": search_q if (lang != "en" and market == "us") else None}
    def finish(result: dict) -> dict:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        print(json.dumps({"event": "ask", "lang": lang, "found": result["found"],
                          "uncited_sentences": result.get("uncited_sentences"), **timings}), flush=True)
        return {**result, "timings": timings}

    def not_found() -> dict:
        text, mode = timed("refusal_ms", lambda: fallback(question, out_lang))
        return finish({**base, "answer": text, "found": False, "mode": mode,
                       "playful": mode == "playful", "sources": [], "model": llm.last_model_used})

    # 1) RETRIEVE
    # Hybrid retrieval: take more candidates from the vector search, then re-rank them with a small
    # keyword bonus, so a passage that literally mentions "alkohol" beats one that is only "about ibuprofen".
    chunks = store.search(search_q, top_k=top_k * 5, drugs=drugs or None, timings=timings, market=market, field=field)
    names = {w for m in mentions for w in markets.norm(m["asked"]).split()} | {d.lower() for d in drugs}
    prefer = {m["main"] for m in mentions if m.get("main")}
    chunks = rerank(chunks, f"{question} {search_q}", names, prefer=prefer)[:top_k]
    relevant = [c for c in chunks if c["score"] >= config.MARKETS[market]["min_score"]]
    if not relevant:
        return not_found()

    # When the user names a product whose own leaflet isn't indexed (e.g. Ibuprom), the sources come from
    # another product of the same group (e.g. Nurofen). Tell the model and the UI so it's transparent.
    equivalents = []
    for m in mentions:
        reps = sorted({c["drug"] for c in relevant if c.get("group") in m["groups"]})
        if reps and not any(markets.first_word(r) == m["key"] for r in reps):
            inn = next(c for c in relevant if c.get("group") in m["groups"])
            equivalents.append({"asked": m["asked"], "asked_url": m["url"], "reps": reps,
                                "inn": inn.get("en") or inn.get("inn", ""), "form": inn.get("form", "")})

    # 2) AUGMENT + 3) GENERATE
    def generate(language: str) -> str:
        msg = f"Sources:\n{build_context(relevant)}\n\nQuestion: {question}"
        if lang != "en" and search_q != question:
            msg += f"\n(English version of the question: {search_q})"
        for e in equivalents:
            msg += SAME_SUBSTANCE_NOTE.format(asked=e["asked"], reps=", ".join(e["reps"]), inn=e["inn"])
        msg += f"\n\nWrite the answer in {lang_name(language)}."
        return timed("llm_ms", lambda: llm.generate(SYSTEM_PROMPT, msg)).strip()

    answer = generate(out_lang)
    if (not answer or "NOT_FOUND" in answer) and out_lang not in ("en", "auto"):
        # Some (smaller) models refuse when answer language != source language. Fall back to:
        # answer in English (same language as the sources), then translate that answer.
        english = generate("en")
        if english and "NOT_FOUND" not in english:
            answer = timed("llm_ms", lambda: llm.generate(
                TRANSLATE_ANSWER_PROMPT.format(language=lang_name(out_lang)), english)).strip()

    if not answer or "NOT_FOUND" in answer:
        return not_found()

    answer = normalize_citations(answer, len(relevant))
    # Drop citations that point to non-existent sources (a small hallucination guard),
    # then record which source numbers the model actually cited.
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(relevant) else "", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    uncited = uncited_sentences(answer)
    sources = [
        {"id": i, **{k: c.get(k, "") for k in ("drug", "section", "text", "url", "score", "effective_time",
                                               "inn", "form", "source")},
         "brands": drug_brands().get(c["drug"], []) if field == "drug" else [], "cited": i in cited}
        for i, c in enumerate(relevant, start=1)
    ]
    return finish({**base, "answer": answer, "found": True, "mode": "sourced", "sources": sources,
                   "equivalents": equivalents, "model": llm.last_model_used, "uncited_sentences": uncited})
