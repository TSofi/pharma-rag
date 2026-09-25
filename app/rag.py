"""The RAG pipeline: question -> retrieve relevant label sections -> LLM answer with citations."""
import difflib
import json
import random
import re
from functools import lru_cache

from . import config, llm, store

SYSTEM_PROMPT = """You are a drug-information assistant for pharmacists.
Answer ONLY using the numbered sources provided (excerpts from official FDA drug labels).
Rules:
- Cite every factual sentence with the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "NOT_FOUND" and nothing else.
- Do not use outside knowledge. Do not guess doses, indications or interactions.
- Be concise: 2-6 sentences or a short bullet list. Use plain, professional language.
- Write the answer in {language}. Keep drug names and numbers exactly as in the sources."""

TRANSLATE_PROMPT = """Translate the user's pharmacy question into English for searching US FDA drug labels.
Replace local/brand drug names with the US generic name when you are sure (e.g. paracetamol -> acetaminophen,
Nurofen -> ibuprofen). Output ONLY the English question, nothing else."""

LANGUAGES = {"en": "English", "uk": "Ukrainian", "pl": "Polish"}
NOT_FOUND = {
    "en": "I couldn't find this in the loaded drug labels, so I won't guess. "
          "Try rephrasing, or ask about one of the drugs in the library.",
    "uk": "Я не знайшла цього в завантажених інструкціях до препаратів, тому не вгадуватиму. "
          "Спробуйте переформулювати питання або оберіть препарат із бібліотеки.",
    "pl": "Nie znalazłam tego w załadowanych ulotkach leków, więc nie będę zgadywać. "
          "Spróbuj przeformułować pytanie albo wybierz lek z biblioteki.",
}
_PL_WORDS = {"jest", "czy", "jaka", "jaki", "jakie", "dawka", "dawkowanie", "lek", "leku", "można", "mozna",
             "się", "sie", "na", "przy", "ciąży", "ciazy", "skutki", "uboczne", "działa", "dla", "przeciwwskazania"}

NOT_FOUND_MSG = NOT_FOUND["en"]


def detect_language(text: str) -> str:
    """Cheap heuristic, no extra API call: Cyrillic -> Ukrainian, Polish letters/words -> Polish."""
    if re.search(r"[а-яіїєґА-ЯІЇЄҐ]", text):
        return "uk"
    words = set(re.findall(r"[a-ząćęłńóśźż]+", text.lower()))
    if re.search(r"[ąćęłńśźż]", text.lower()) or len(words & _PL_WORDS) >= 2:
        return "pl"
    return "en"


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


def library() -> list[dict]:
    """Drugs available in the knowledge base (for the UI)."""
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


def random_fact(seen: set[str] | None = None) -> dict:
    """A random fact the viewer hasn't seen yet in this session (by drug)."""
    facts = _facts()
    fresh = [f for f in facts if f["drug"] not in (seen or set())]
    return random.choice(fresh or facts) if facts else {}


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {c['drug']} — {c['section']}\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )


# One knowledge base per country ("market"). An EU country = EMA centrally authorised products
# (valid in every EU state) + that country's national register. Planned: at, pl, ua (see README).
MARKETS = {"us": "United States (FDA)"}


def ask(question: str, top_k: int | None = None, answer_language: str = "auto", market: str = "us") -> dict:
    """The question can be in any supported language; the MARKET decides which labels are searched,
    and the answer is written in `answer_language` ("auto" = same language as the question)."""
    if market not in MARKETS:
        raise ValueError(f"Market '{market}' is not available yet")
    top_k = top_k or config.TOP_K
    lang = detect_language(question)                                   # language of the QUESTION
    out_lang = lang if answer_language == "auto" else answer_language  # language of the ANSWER

    # 0) QUERY REWRITING: the labels and the embedding model are English, so a Ukrainian/Polish
    #    question is translated first. The answer is still written in the user's language.
    search_q = question if lang == "en" else llm.generate(TRANSLATE_PROMPT, question).strip() or question
    drugs, corrections = detect_drugs_fuzzy(search_q)
    base = {"detected_drugs": drugs, "corrections": corrections, "language": lang, "answer_language": out_lang,
            "market": market, "search_query": search_q if lang != "en" else None}
    not_found = {**base, "answer": NOT_FOUND[out_lang], "found": False, "sources": []}

    # 1) RETRIEVE
    chunks = store.search(search_q, top_k=top_k, drugs=drugs or None)
    relevant = [c for c in chunks if c["score"] >= config.MIN_SCORE]
    if not relevant:
        return not_found

    # 2) AUGMENT + 3) GENERATE
    user_msg = f"Sources:\n{build_context(relevant)}\n\nQuestion: {question}"
    if lang != "en":
        user_msg += f"\n(English version of the question: {search_q})"
    answer = llm.generate(SYSTEM_PROMPT.format(language=LANGUAGES[out_lang]), user_msg).strip()

    if not answer or "NOT_FOUND" in answer:
        return not_found

    # Drop citations that point to non-existent sources (a small hallucination guard),
    # then record which source numbers the model actually cited.
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(relevant) else "", answer)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    sources = [
        {"id": i, **{k: c[k] for k in ("drug", "section", "text", "url", "score", "effective_time")},
         "brands": drug_brands().get(c["drug"], []), "cited": i in cited}
        for i, c in enumerate(relevant, start=1)
    ]
    return {**base, "answer": answer, "found": True, "sources": sources, "model": llm.last_model_used}
