"""The RAG pipeline: question -> retrieve relevant label sections -> LLM answer with citations."""
import difflib
import json
import random
import re
import time
from functools import lru_cache

from . import config, llm, markets, observability, store

SYSTEM_PROMPT = """You are a drug-information assistant for pharmacists.
Answer ONLY using the numbered sources provided (excerpts from official drug labels / patient leaflets).
Rules:
- Cite every factual sentence with the source number in plain ASCII square brackets, e.g. [1] or [2][3].
  Cite each source separately: [2][3], never [2-3] or [2, 3]. Cite only the 1-3 sources that really
  contain the fact, not all of them. Never use any other citation style (no 【】, no line ranges).
- If the sources do not contain the answer, the answer part is exactly: "NOT_FOUND" and nothing else.
- Do not use outside knowledge. Do not guess doses, indications or interactions.
- Be concise: 2-6 sentences or a short bullet list. Use plain, professional language.
- The sources may be in English, Polish or Ukrainian, and the answer may be requested in another
  language. Translating the facts is expected and is NOT a reason to reply NOT_FOUND.
  Keep drug names and numbers exact.
- ALWAYS start the reply with one line saying how urgent the user's situation is, then the answer on the
  next line:
  RISK: EMERGENCY -> something has ALREADY happened that can be dangerous: much more than the allowed dose
    was taken, a child swallowed medicine by accident, poisoning, a severe allergic reaction, trouble
    breathing, chest pain, fainting, or any self-harm.
  RISK: CAUTION -> a dosing mistake or "what if" that is usually not an emergency: a missed, late or
    doubled dose, mixed-up times, "what happens if I take...".
  RISK: NONE -> everything else, including using a medicine the way a doctor said (e.g. crushing a
    tablet into food)."""

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
# Orange-level ("CAUTION") safety line shown under a leaflet answer. Fixed text, never generated.
SAFETY_NOTE = {
    "en": "If you feel unwell or are not sure what to do, contact your doctor or pharmacist. "
          "In an emergency call 112 (911 in the US).",
    "uk": "Якщо ви погано почуваєтеся або не впевнені, що робити, зверніться до лікаря чи фармацевта. "
          "У невідкладному випадку телефонуйте 112 (103 в Україні).",
    "pl": "Jeśli źle się czujesz lub nie wiesz, co zrobić, skontaktuj się z lekarzem lub farmaceutą. "
          "W nagłym wypadku dzwoń pod 112.",
    "cs": "Pokud se necítíte dobře nebo si nejste jistí, co dělat, obraťte se na lékaře nebo lékárníka. "
          "V případě nouze volejte 112 (nebo 155).",
    "es": "Si se encuentra mal o no sabe qué hacer, consulte a su médico o farmacéutico. "
          "En caso de emergencia, llame al 112.",
    "fr": "Si vous ne vous sentez pas bien ou avez un doute, contactez votre médecin ou votre pharmacien. "
          "En cas d'urgence, appelez le 112 (ou le 15, SAMU).",
}
# Red card when the LLM is DOWN and the keywords look like an emergency. The keywords can't judge the
# situation as well as the LLM, so the text is conditional ("if ...") and says the service is busy.
EMERGENCY_SOFT = {
    "en": "If someone has taken too much of a medicine, swallowed something harmful or has trouble breathing, "
          "call 112 (911 in the US) or a poison control centre now. Our AI service is busy right now; "
          "for other questions, please try again in a moment.",
    "uk": "Якщо хтось прийняв забагато ліків, проковтнув щось шкідливе або має проблеми з диханням, негайно "
          "телефонуйте 112 (103 в Україні) або в токсикологічний центр. Наш сервіс ШІ зараз перевантажений; "
          "з іншими питаннями спробуйте ще раз трохи пізніше.",
    "pl": "Jeśli ktoś przyjął za dużo leku, połknął coś szkodliwego lub ma trudności z oddychaniem, natychmiast "
          "zadzwoń pod 112 lub do ośrodka toksykologicznego. Nasza usługa AI jest teraz przeciążona; "
          "z innymi pytaniami spróbuj ponownie za chwilę.",
    "cs": "Pokud někdo užil příliš mnoho léku, spolkl něco škodlivého nebo má potíže s dýcháním, okamžitě "
          "volejte 112 (nebo 155) nebo Toxikologické informační středisko. Naše služba AI je teď přetížená; "
          "s dalšími dotazy to prosím zkuste za chvíli znovu.",
    "es": "Si alguien ha tomado demasiado de un medicamento, ha tragado algo dañino o tiene dificultad para "
          "respirar, llame ahora al 112 o al Servicio de Información Toxicológica. Nuestro servicio de IA está "
          "saturado en este momento; para otras preguntas, inténtelo de nuevo en un momento.",
    "fr": "Si quelqu'un a pris trop d'un médicament, a avalé quelque chose de dangereux ou a du mal à respirer, "
          "appelez immédiatement le 112 (ou le 15, SAMU) ou un centre antipoison. Notre service d'IA est "
          "surchargé en ce moment ; pour les autres questions, réessayez dans un instant.",
}

# Keyword check, used ONLY when the LLM is down (when it works, the LLM judges the situation).
# An emergency needs TWO signals: something was ALREADY taken (past tense, so "what if I take..." never
# matches) AND a sign of too much (whole pack, by accident, 5+ tablets...). A few words are enough alone,
# again only in the "already happened" form ("overdosed", not "overdose"). Hand-written: needs a check by
# native speakers (like S7).
_TAKEN = re.compile(r"""\b(?:
    took|swallowed|drank|ate|ingested|(?:has|have|had|'ve)\s+(?:taken|swallowed|drunk|eaten)|got\s+into
  | випи(?:в|ла|ли|ло)|проковтну(?:в|ла|ли|ло)|з['’ʼ]?ї(?:в|ла|ли|ло)|прийня(?:в|ла|ли|ло)
  | połkn(?:ął|ęła|ęło|ęli|ęły)\w*|wzi(?:ął|ęła|ęło|ęli|ęły)\w*|zjad(?:ł|ła|ło|li|ły)\w*|wypi(?:ł|ła|ło|li|ły)\w*
  | zaży(?:ł|ła|ło|li|ły)\w*|przyj(?:ął|ęła|ęło|ęli|ęły)\w*
  | spolkl\w*|snědl\w*|vypil\w*|vzal\w*|užil\w*|požil\w*
  | tom[éó]|(?:he|ha|han|has)\s+tomado|trag(?:ué|ó|aron)|tragad[oa]|beb(?:ió|í|ieron)|bebid[oa]|com(?:ió|ieron)|comid[oa]
  | ingiri[óo]|inger(?:í|ido)
  | (?:ai|a|as|avons|avez|ont)\s+(?:pris|bu)|avalée?s?|mangée?s?|ingérée?s?
)\b""", re.I | re.X)
_TOO_MUCH = re.compile(r"""(?:
    \btoo\s+(?:many|much)\b|\b(?:whole|entire|full)\s+(?:pack|packet|box|bottle|strip|blister|sheet)\b
  | \ball\s+(?:the|my|his|her|of\s+the)\s+(?:pills|tablets|capsules)\b
  | \bby\s+(?:mistake|accident)\b|\baccidentally\b
  | \b(?:[5-9]|[1-9]\d+)\s+(?:\w+\s+){0,2}?(?:pills?|tablets?|capsules?|таблет\w*|капсул\w*|tabletek|tabletki
        |kapsułek|pigułek|tablet|tablety|tabletky|kapslí|kapsle|comprimidos?|pastillas?|cápsulas?|comprimés?
        |gélules?|cachets?)\b
  | \b(?:за|за\s?надто\s)багато\b|\b(?:всю|цілу|весь|цілий)?\s*(?:пачку|упаковку|пляшку|блістер|флакон|коробку)\b
  | \bвипадково\b|\bпомилково\b
  | \b(?:za|zbyt)\s+(?:dużo|wiele)\b|\bcał(?:e|ą|y)\s+(?:opakowanie|paczkę|butelkę|blister|listek|fiolkę|pudełko)\b
  | \bprzez\s+pomyłkę\b|\bomyłkowo\b|\bprzypadk(?:iem|owo)\b
  | \bpříliš\s+(?:mnoho|moc)\b|\bcel(?:é|ou|ý)\s+(?:balení|krabičku|lahev|láhev|plato|tubu)\b|\bomylem\b|\bnáhodou\b
  | \bdemasiad[oa]s?\b|\btod[oa]\s+(?:la\s+caja|el\s+bl[ií]ster|el\s+frasco|la\s+botella|el\s+envase)\b
  | \bpor\s+(?:error|equivocación|accidente)\b|\bsin\s+querer\b|\baccidentalmente\b
  | \btrop\s+de\b|\btou(?:te|t)\s+(?:la\s+boîte|le\s+flacon|la\s+plaquette|la\s+bouteille|le\s+tube)\b
  | \bpar\s+(?:erreur|accident|mégarde)\b|\baccidentellement\b
)""", re.I | re.X)
_ALREADY_BAD = re.compile(r"""\b(?:
    overdosed|poisoned|unconscious|passed\s+out|not\s+breathing|stopped\s+breathing|can(?:no|')t\s+breathe
  | передозува(?:в|ла|ли|ло)|отруї(?:вся|лася|лися|лось|в|ла|ли)|непритомн\w*|без\s+свідомості|не\s+дихає
  | przedawkował\w*|zatru(?:ł|ła|li|ło)\w*|nieprzytomn\w*|nie\s+oddycha
  | předávkoval\w*|otrávil\w*|v\s+bezvědomí|nedýchá
  | intoxic(?:ó|ado|ada)|envenen(?:ó|ado|ada)|(?:tuvo|ha\s+tenido|he\s+tenido)\s+una\s+sobredosis|inconsciente|no\s+respira
  | (?:fait|faite)\s+une\s+(?:overdose|surdose)|intoxiquée?|empoisonnée?|inconsciente?|ne\s+respire\s+(?:plus|pas)
)\b""", re.I | re.X)


def looks_like_emergency(question: str) -> bool:
    """Keyword fallback for when the LLM can't judge (see the comment above)."""
    return bool(_ALREADY_BAD.search(question) or (_TAKEN.search(question) and _TOO_MUCH.search(question)))


RISK_LEVELS = ("none", "caution", "emergency")  # in order of urgency
_RISK_LINE = re.compile(r"^[^\w\n]*RISK[^\w\n]*:?[^\w\n]*(EMERGENCY|CAUTION|NONE)[^\w\n]*$\n?", re.I | re.M)


def split_risk(reply: str) -> tuple[str, str]:
    """("RISK: CAUTION\nanswer") -> ("caution", "answer"). No risk line -> "none" (older/forgetful model)."""
    found = [m.lower() for m in _RISK_LINE.findall(reply or "")]
    level = max(found, key=RISK_LEVELS.index) if found else "none"
    return level, _RISK_LINE.sub("", reply or "").strip()


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
    except Exception as e:  # noqa: BLE001 -- the fallback is optional; never fail the request because of it
        # Logged (was silent): the user now gets the plain "not found" only because the LLM failed, so this
        # answer must not be cached (app/cache.py) and counts as an issue in the metrics.
        observability.log_event("fallback_failed", error=f"{type(e).__name__}: {e}")
        if looks_like_emergency(question):  # the LLM can't judge: keywords decide (soft red card)
            return EMERGENCY_SOFT.get(out_lang, EMERGENCY_SOFT["en"]), "emergency"
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


REPAIR_NOTE = """

Your previous answer (below) had no valid citation. Rewrite it so that EVERY factual sentence cites
the number of the source it comes from, e.g. [1] or [2][3]. Keep only facts that are in the sources.
If the sources do not support the answer, reply exactly: NOT_FOUND
Previous answer:
{answer}"""


def _cite(answer: str, n_sources: int) -> tuple[str, set[int]]:
    """Normalise citation styles, drop numbers that point to no source, return (text, cited numbers)."""
    answer = normalize_citations(answer, n_sources)
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    return answer, {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


NUMBERS_NOTE = """

Your previous answer (below) contains numbers that are NOT in the source it cites: {problems}.
Rewrite it: copy every number (dose, strength, interval, duration, age, percentage) exactly as it is in
the source you cite, and cite that source. Do not convert units. If no source contains the number, leave
it out, or reply exactly: NOT_FOUND
Previous answer:
{answer}"""

# Number words that leaflets use instead of digits ("dwie tabletki", "half a tablet"). Only the SOURCE side
# uses them (so "2 tablets" in the answer matches "dwie tabletki"); in the answer only digits are checked.
_NUMBER_WORDS = {
    0.5: "half halves ½ pół połowa połowę połowy половина половину половини півтаблетки půl polovina polovinu "
         "media medio mitad moitié demi demie",
    1: "one once jeden jedna jedną jedno jednej jednego один одна одну одне одного однієї раз jednu un une uno una",
    2: "two twice dwa dwie dwóch dwukrotnie два дві двох двічі dva dvě dvou dvakrát dos deux",
    3: "three trzy trzech trzykrotnie три трьох тричі tři tří třikrát tres trois",
    4: "four cztery czterech чотири чотирьох čtyři čtyř cuatro quatre",
    5: "five pięć pięciu п'ять п’ять пʼять пяти п'яти pět cinco cinq",
    6: "six sześć sześciu шість шести šest seis",
    7: "seven siedem siedmiu сім семи sedm siete sept",
    8: "eight osiem ośmiu вісім восьми osm ocho huit",
    9: "nine dziewięć dziewięciu дев'ять дев’ять devět nueve neuf",
    10: "ten dziesięć dziesięciu десять десяти deset diez dix",
    12: "twelve dwanaście дванадцять dvanáct doce douze",
}
_WORD_VALUE = {w: float(v) for v, words in _NUMBER_WORDS.items() for w in words.split()}
# A number = digits, optionally with , or . parts and space-grouped thousands ("1 000"). Not glued to a
# letter or digit before it, so "B12" and "CYP3A4" are names, not numbers.
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:[ \u00a0\u202f]\d{3}(?!\d))*")
_LIST_NUMBER = re.compile(r"^\s*\d{1,2}[.)]\s+", re.M)  # "1. Take ..." is list numbering, not a fact

# Units, so "1 g" in the leaflet also accepts "1000 mg" (and 1000000 mcg) in the answer: a number WITH a unit
# is compared as an amount in the kind's base unit (mg, ml). unit -> (kind, size). Mass and volume never mix.
_UNITS = {"mcg": ("mass", 0.001), "µg": ("mass", 0.001), "μg": ("mass", 0.001), "ug": ("mass", 0.001),
          "мкг": ("mass", 0.001), "mg": ("mass", 1), "мг": ("mass", 1), "g": ("mass", 1000), "г": ("mass", 1000),
          "kg": ("mass", 1e6), "кг": ("mass", 1e6), "ml": ("volume", 1), "мл": ("volume", 1),
          "l": ("volume", 1000), "л": ("volume", 1000)}
# a number right before a unit; the unit must end there ("2 glasses" is not "2 g")
_AMOUNT = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)*(?:[ \u00a0\u202f]\d{3}(?!\d))*)\s?"
                     r"(mcg|µg|μg|ug|мкг|mg|мг|kg|кг|g|г|ml|мл|l|л)(?![^\W\d_])", re.I)


def _values(token: str) -> set[float]:
    """Possible values of one number as written: "2,5" -> 2.5; "1,000" -> 1000 or 1.0 (EN thousands or
    an EU decimal); "1 000" -> 1000."""
    t = re.sub(r"[ \u00a0\u202f]", "", token)
    out = set()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):
        out.add(float(re.sub(r"[.,]", "", t)))
    if re.fullmatch(r"\d+[.,]\d+", t):
        out.add(float(t.replace(",", ".")))
    if re.fullmatch(r"\d+", t):
        out.add(float(t))
    if not out:  # e.g. 1.2.3: its parts
        out = {float(p) for p in re.split(r"[.,]", t) if p}
    return out


def _numbers_in(text: str, with_words: bool = False) -> set[float]:
    found: set[float] = set()
    for token in _NUMBER.findall(text):
        found |= _values(token)
        found |= {float(p) for p in re.split(r"[ \u00a0\u202f]", token) if p.isdigit()}  # "1 500" -> 1, 500 too
    if with_words:  # the source side: also number words ("dwie" -> 2)
        found |= {_WORD_VALUE[w] for w in re.findall(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)?|½", text.lower()) if w in _WORD_VALUE}
    return found


def _amounts(text: str) -> tuple[set[tuple[str, float]], dict[str, set[float]]]:
    """(kind, amount in base unit) of every "number + unit", and per kind the raw numbers written with a
    unit of that kind: "500 µg" -> {("mass", 0.5)}, {"mass": {500}}."""
    amounts, raw = set(), {"mass": set(), "volume": set()}
    for token, unit in _AMOUNT.findall(text):
        kind, size = _UNITS[unit.lower()]
        for value in _values(token):
            amounts.add((kind, round(value * size, 6)))
            raw[kind].add(value)
    return amounts, raw


def wrong_numbers(answer: str, sources: list[dict], allowed: str = "") -> list[dict]:
    """Numbers of a cited sentence that are in none of the sources it cites. `allowed` = text whose numbers
    may be repeated (the user's question). Returns [{"number", "cited", "found_in"}] (found_in = sources
    that DO contain it: a wrong citation rather than a wrong number).
    A number WITH a unit in the answer is compared as an amount: "1000 mg" matches "1 g", but "500 mg" does
    not match "500 µg" and "1000 ml" does not match "1 g"."""
    texts = [f"{c.get('drug', '')} {c.get('section', '')} {c.get('text', '')}" for c in sources]
    per_source = [_numbers_in(t, with_words=True) for t in texts]
    per_amounts = [_amounts(t) for t in texts]
    ok_anyway = _numbers_in(allowed)

    def supported(token: str, unit: str | None, ids: list[int]) -> bool:
        whole = _values(token)
        if unit:
            kind, size = _UNITS[unit.lower()]
            mine = {(kind, round(v * size, 6)) for v in whole}
            for i in ids:
                amounts, raw = per_amounts[i - 1]
                if mine & amounts:
                    return True  # the same amount, in any unit
                if whole & per_source[i - 1] and not whole & raw[kind]:
                    return True  # the number is in the leaflet without a unit of this kind ("Nurofen 400")
            return bool(whole & ok_anyway)
        have = set().union(*(per_source[i - 1] for i in ids)) | ok_anyway
        parts = [p for p in re.split(r"[ \u00a0\u202f]", token) if p]
        return bool(whole & have) or all(_values(p) & have for p in parts)

    problems = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ¿¡])|\n+", _LIST_NUMBER.sub("", answer)):
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", part) if 1 <= int(n) <= len(sources)})
        if not cited:
            continue  # uncited sentences are counted by uncited_sentences()
        text = re.sub(r"\[\d+\]", " ", part)
        units = {m.start(1): m.group(2) for m in _AMOUNT.finditer(text)}  # number position -> its unit
        for m in _NUMBER.finditer(text):
            token, unit = m.group(0), units.get(m.start())
            if supported(token, unit, cited):
                continue
            problems.append({"number": token, "cited": cited,
                             "found_in": [i for i in range(1, len(sources) + 1)
                                          if i not in cited and supported(token, unit, [i])]})
    return problems


# ---------------------------------------------------------------- meaning check (the "judge")
JUDGE_PROMPT = """You check a drug-information answer against the leaflet excerpts it cites.
For each numbered claim decide, using ONLY the excerpts that claim cites:
SUPPORTED - the excerpt says this. A translation into another language, a paraphrase or a shorter
  summary counts as SUPPORTED.
CONTRADICTED - the excerpt says something different or the opposite (e.g. "can be used" vs "do not
  use", another dose, age, interval or duration).
NOT_IN_SOURCE - the excerpt does not contain this information at all.
Reply with one line per claim, exactly in this form and nothing else:
1: SUPPORTED
2: CONTRADICTED"""

MEANING_NOTE = """

A checker found that these sentences of your previous answer (below) do not match the source they cite:
{problems}
Rewrite the answer: say only what the cited sources say, with the same meaning (never turn "do not use"
into "can be used" or the reverse), and cite them. If the sources do not answer the question, reply
exactly: NOT_FOUND
Previous answer:
{answer}"""

# Topics where a wrong "yes/no" is most dangerous (decision 1b). Hand-written for 6 languages: needs a check
# by native speakers (like S9).
_RISKY = re.compile(r"""(?:
    pregnan|breast[-\s]?feed|nursing|lactat|\bchild|\bkids?\b|\bbab(?:y|ies)\b|infant|alcohol|\bdrunk\b|\bdrink
  | \bdriv(?:e|es|ing)\b|machiner|\bother\s+(?:medicine|medication|drug|pill)|interact|together\s+with|allerg
  | kidney|renal|liver|hepat
  | вагітн|годуванн\w*\s+груд|грудн\w*\s+вигодов|лактац|дит(?:и|я|ин)|діт(?:ей|ям|ьми)|немовл|малюк|алкогол|спирт
  | керуванн|за\s+кермом|водінн|механізм|інш\w+\s+лік|взаємоді|алерг|нирк|печінк
  | ciąż|karmi\w*\s+piersi|laktac|dzieck|dzieci|niemowl|alkohol|prowadz\w*\s+(?:samoch|pojazd|auto)|maszyn
  | inn\w*\s+lek|interakc|alergi|uczul|nerk|wątrob
  | těhot|kojen|kojenc|děti|dítě|dětí|alkohol|řízen|řídit|stroj|jin\w*\s+lék|alergi|ledvin|jater|játr
  | embaraz|lactancia|amamant|niñ[oa]|bebé|alcohol|conduc|máquinas|otros\s+medicamentos|interacci|alergi|riñ|hígado|hepát
  | grossesse|enceinte|allait|enfant|bébé|nourrisson|alcool|conduire|conduite|machines|autres\s+médicaments
  | interaction|allergi|\breins?\b|rénal|foie|hépat
)""", re.I | re.X)


def risky_topic(text: str) -> bool:
    return bool(_RISKY.search(text) or _CHILD.search(text))


def _cited_claims(answer: str, n_sources: int) -> list[tuple[str, list[int]]]:
    """(sentence without [n] markers, the source numbers it cites) for every cited sentence."""
    claims = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ¿¡])|\n+", answer):
        ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", part) if 1 <= int(n) <= n_sources})
        text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\[\d+\]", "", part)).strip(" -*•")
        if ids and re.search(r"\w", text):
            claims.append((text, ids))
    return claims


_VERDICT = re.compile(r"^\W*(?:claim\s*)?(\d+)\W*?[\s:.)\-–]*\W*(SUPPORTED|CONTRADICTED|NOT[\s_-]*IN[\s_-]*SOURCE)",
                      re.I | re.M)


def parse_verdicts(reply: str, n_claims: int) -> dict[int, str]:
    """"1: SUPPORTED\n2) contradicted" -> {1: "supported", 2: "contradicted"}; unreadable lines are skipped."""
    out = {}
    for num, verdict in _VERDICT.findall(reply or ""):
        if 1 <= int(num) <= n_claims:
            out[int(num)] = re.sub(r"[\s_-]+", "_", verdict.lower())
    return out


def judge(answer: str, sources: list[dict]) -> tuple[list[dict] | None, int]:
    """Ask the judge about every cited sentence. Returns (problems, unsupported_count); problems is None
    when the judge is down or its reply unreadable (decision 3a: the caller keeps the answer)."""
    claims = _cited_claims(answer, len(sources))
    if not claims:
        return [], 0
    used = sorted({i for _, ids in claims for i in ids})
    excerpts = "\n\n".join(f"[{i}] {sources[i - 1].get('drug', '')} — {sources[i - 1].get('section', '')}\n"
                            f"{sources[i - 1].get('text', '')}" for i in used)
    listing = "\n".join(f"{k}. {text} (cites {''.join(f'[{i}]' for i in ids)})" for k, (text, ids) in enumerate(claims, 1))
    try:
        reply = llm.strip_reasoning(llm.generate(JUDGE_PROMPT, f"Excerpts:\n{excerpts}\n\nClaims:\n{listing}"))
    except Exception as e:  # noqa: BLE001
        observability.log_event("judge_unavailable", error=f"{type(e).__name__}: {e}")
        return None, 0
    verdicts = parse_verdicts(reply, len(claims))
    if not verdicts:
        observability.log_event("judge_unreadable", reply=reply[:200])
        return None, 0
    problems, unsupported = [], 0
    for k, (text, ids) in enumerate(claims, 1):
        v = verdicts.get(k)  # a claim the judge skipped is not treated as wrong
        if v == "contradicted" or (v == "not_in_source" and config.JUDGE_STRICT):
            problems.append({"sentence": text, "cited": ids, "verdict": v})
        elif v == "not_in_source":
            unsupported += 1
    return problems, unsupported


def ask(question: str, top_k: int | None = None, answer_language: str = "auto", market: str = "us") -> dict:
    """The question can be in any supported language; the MARKET decides which labels are searched,
    and the answer is written in `answer_language` ("auto" = same language as the question).
    All LLM calls for one question share one time budget (llm.request_scope)."""
    with llm.request_scope():
        try:
            return _ask(question, top_k, answer_language, market)
        except Exception as e:  # noqa: BLE001
            if market not in MARKETS or not looks_like_emergency(question):
                raise  # a normal question: the API answers 503 "busy, try again"
            # The LLM (or search) is down and this looks like an emergency: never show just an error page.
            observability.log_event("emergency_without_llm", error=type(e).__name__)
            lang = detect_language(question)
            out_lang = lang if answer_language == "auto" else answer_language
            observability.note(language=lang, answer_language=out_lang, found=False, mode="emergency",
                               risk="emergency", llm_unavailable=True)
            return {"answer": EMERGENCY_SOFT.get(out_lang, EMERGENCY_SOFT["en"]), "found": False,
                    "mode": "emergency", "playful": False, "llm_unavailable": True, "sources": [],
                    "equivalents": [], "detected_drugs": [], "corrections": [], "language": lang,
                    "answer_language": out_lang, "market": market, "search_query": None, "model": "",
                    "timings": {}}


class _Emergency(Exception):
    """The LLM marked the user's situation RISK: EMERGENCY."""


def _ask(question: str, top_k: int | None, answer_language: str, market: str) -> dict:
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
    state = {"risk": "none"}  # how urgent the LLM judged the situation (the RISK line of its reply)

    def finish(result: dict) -> dict:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        if result["mode"] == "emergency" and result["answer"] in EMERGENCY_SOFT.values():
            result["llm_unavailable"] = True
        if state["risk"] == "caution" and result["mode"] != "emergency":
            result["safety_note"] = SAFETY_NOTE.get(out_lang, SAFETY_NOTE["en"])
        summary = {"language": lang, "answer_language": result.get("answer_language", out_lang),
                   "detected_drugs": result.get("detected_drugs", []), "found": result["found"],
                   "mode": result["mode"], "risk": state["risk"], "model": result.get("model", ""),
                   "meaning_check": result.get("meaning_check"),
                   "uncited_sentences": result.get("uncited_sentences"), "timings": dict(timings)}
        if result.get("llm_unavailable"):
            summary["llm_unavailable"] = True
        # Inside an API request these fields go into its ONE final log line; from scripts: a line of its own.
        if not observability.note(**summary):
            observability.log_event("ask", **summary)
        return {**result, "timings": timings}

    def not_found() -> dict:
        text, mode = timed("refusal_ms", lambda: fallback(question, out_lang))
        return finish({**base, "answer": text, "found": False, "mode": mode,
                       "playful": mode == "playful", "sources": [], "model": llm.model_used()})

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
    def generate(language: str, repair: str = "") -> str:
        msg = f"Sources:\n{build_context(relevant)}\n\nQuestion: {question}"
        if lang != "en" and search_q != question:
            msg += f"\n(English version of the question: {search_q})"
        for e in equivalents:
            msg += SAME_SUBSTANCE_NOTE.format(asked=e["asked"], reps=", ".join(e["reps"]), inn=e["inn"])
        msg += f"\n\nWrite the answer in {lang_name(language)}."
        msg += repair  # a repair note (missing citations / wrong numbers), or nothing
        level, text = split_risk(llm.strip_reasoning(timed("llm_ms", lambda: llm.generate(SYSTEM_PROMPT, msg))))
        if level == "emergency":
            raise _Emergency
        if level == "caution":
            state["risk"] = "caution"
        return text

    def refused(text: str) -> bool:
        return not text or "NOT_FOUND" in text

    try:
        written_in = out_lang
        answer = generate(out_lang)
        if refused(answer) and out_lang not in ("en", "auto"):
            # Some (smaller) models refuse when answer language != source language. Fall back to:
            # answer in English (same language as the sources), then translate that answer.
            english = generate("en")
            if not refused(english):
                try:
                    answer = split_risk(llm.strip_reasoning(timed("llm_ms", lambda: llm.generate(
                        TRANSLATE_ANSWER_PROMPT.format(language=lang_name(out_lang)), english))))[1]
                except Exception as e:  # noqa: BLE001 -- a cited English answer beats an error page
                    observability.log_event("translation_failed", error=f"{type(e).__name__}: {e}")
                    answer = ""
                if _cite(english, len(relevant))[1] and not _cite(answer, len(relevant))[1]:
                    answer, written_in = english, "en"  # the translation failed or lost every [n]

        if refused(answer):
            return not_found()

        # Drop citations that point to non-existent sources (a small hallucination guard),
        # then record which source numbers the model actually cited.
        answer, cited = _cite(answer, len(relevant))
        if not cited:
            # Violet = "confirmed by the leaflet", so an answer without a single valid [n] can't be shown as
            # violet. One repair attempt; if it still has no citation, it goes the "not found" way (orange).
            try:
                repaired = generate(written_in, repair=REPAIR_NOTE.format(answer=answer))
            except _Emergency:
                raise
            except Exception as e:  # noqa: BLE001
                observability.log_event("citation_repair_failed", error=f"{type(e).__name__}: {e}")
                repaired = ""
            answer, cited = _cite(repaired, len(relevant)) if not refused(repaired) else ("", set())
            if not cited:
                observability.log_event("uncited_answer_rejected", lang=lang)
                return not_found()

        # The citations must be TRUE: every number of a cited sentence is in the source it cites.
        # One repair attempt ("copy the numbers exactly"); still wrong -> not violet (decision "A").
        allowed = f"{question} {search_q}"
        problems = wrong_numbers(answer, relevant, allowed)
        if problems:
            note = ", ".join(f'"{p["number"]}" (you cited {"".join(f"[{i}]" for i in p["cited"])}'
                             + (f'; it is in {"".join(f"[{i}]" for i in p["found_in"])}' if p["found_in"] else "")
                             + ")" for p in problems)
            try:
                repaired = generate(written_in, repair=NUMBERS_NOTE.format(problems=note, answer=answer))
            except _Emergency:
                raise
            except Exception as e:  # noqa: BLE001
                observability.log_event("number_repair_failed", error=f"{type(e).__name__}: {e}")
                repaired = ""
            fixed, fixed_cited = _cite(repaired, len(relevant)) if not refused(repaired) else ("", set())
            if not fixed_cited or wrong_numbers(fixed, relevant, allowed):
                observability.log_event("wrong_numbers_rejected", lang=lang,
                                        numbers=[p["number"] for p in problems])
                return not_found()
            answer, cited = fixed, fixed_cited

        # The MEANING must match too (the "judge", a second LLM call): "safe in pregnancy [1]" while [1] says
        # "do not use". Only on risky topics (JUDGE_MODE). Judge down -> keep the answer (decision 3a).
        meaning, unsupported = "not_checked", 0
        if config.JUDGE_MODE == "always" or (config.JUDGE_MODE == "risky" and risky_topic(f"{question} {search_q} {answer}")):
            problems, unsupported = timed("judge_ms", lambda: judge(answer, relevant))
            meaning = "unavailable" if problems is None else "passed"
            if problems:
                listing = "\n".join(f'- "{p["sentence"]}" ({p["verdict"].replace("_", " ")}: '
                                     f'{"".join(f"[{i}]" for i in p["cited"])})' for p in problems)
                try:
                    repaired = generate(written_in, repair=MEANING_NOTE.format(problems=listing, answer=answer))
                except _Emergency:
                    raise
                except Exception as e:  # noqa: BLE001 -- here the answer is KNOWN to be wrong: no fail-open
                    observability.log_event("meaning_repair_failed", error=f"{type(e).__name__}: {e}")
                    repaired = ""
                fixed, fixed_cited = _cite(repaired, len(relevant)) if not refused(repaired) else ("", set())
                again, unsupported = (timed("judge_ms", lambda: judge(fixed, relevant))
                                      if fixed_cited and not wrong_numbers(fixed, relevant, allowed) else ([1], 0))
                if again:  # still wrong (or no citations / wrong numbers); None = judge down -> keep the fix
                    observability.log_event("wrong_meaning_rejected", lang=lang,
                                            sentences=[p["sentence"][:80] for p in problems])
                    return not_found()
                answer, cited, meaning = fixed, fixed_cited, "repaired"
    except _Emergency:
        # Red card: fixed text, never generated; no leaflet answer next to it ("don't wait, call now").
        state["risk"] = "emergency"
        return finish({**base, "answer": EMERGENCY.get(out_lang, EMERGENCY["en"]), "found": False,
                       "mode": "emergency", "playful": False, "sources": [], "model": llm.model_used()})

    uncited = uncited_sentences(answer)
    sources = [
        {"id": i, **{k: c.get(k, "") for k in ("drug", "section", "text", "url", "score", "effective_time",
                                               "inn", "form", "source")},
         "brands": drug_brands().get(c["drug"], []) if field == "drug" else [], "cited": i in cited}
        for i, c in enumerate(relevant, start=1)
    ]
    return finish({**base, "answer_language": written_in, "answer": answer, "found": True, "mode": "sourced",
                   "sources": sources, "meaning_check": meaning, "unsupported_sentences": unsupported,
                   "equivalents": equivalents, "model": llm.model_used(), "uncited_sentences": uncited})
