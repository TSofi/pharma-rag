"""The RAG pipeline: question -> retrieve relevant label sections -> LLM answer with citations."""
import difflib
import json
import random
import re
import time
from functools import lru_cache

from . import config, llm, store

SYSTEM_PROMPT = """You are a drug-information assistant for pharmacists.
Answer ONLY using the numbered sources provided (excerpts from official FDA drug labels).
Rules:
- Cite every factual sentence with the source number in plain ASCII square brackets, e.g. [1] or [2][3].
  Never use any other citation style (no 【】, no line ranges).
- If the sources do not contain the answer, reply exactly: "NOT_FOUND" and nothing else.
- Do not use outside knowledge. Do not guess doses, indications or interactions.
- Be concise: 2-6 sentences or a short bullet list. Use plain, professional language.
- The sources are in English; the user may want the answer in another language. Translating the
  facts is expected and is NOT a reason to reply NOT_FOUND. Keep drug names and numbers exact."""

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

LANGUAGES = {"en": "English", "uk": "Ukrainian", "pl": "Polish"}
NOT_FOUND = {
    "en": "I couldn't find this in the loaded drug labels, so I won't guess. "
          "Try rephrasing, or ask about one of the drugs in the library.",
    "uk": "Я не знайшла цього в завантажених інструкціях до препаратів, тому не вгадуватиму. "
          "Спробуйте переформулювати питання або оберіть препарат із бібліотеки.",
    "pl": "Nie znalazłam tego w załadowanych ulotkach leków, więc nie będę zgadywać. "
          "Spróbuj przeformułować pytanie albo wybierz lek z biblioteki.",
}
EMERGENCY = {
    "en": "This may be an emergency. Call your local emergency number (112 in the EU, 911 in the US) "
          "or a poison control centre now. Don't wait for an online answer.",
    "uk": "Це може бути невідкладний стан. Негайно телефонуйте на екстрений номер (112 в ЄС, 103 в Україні) "
          "або в токсикологічний центр. Не чекайте відповіді онлайн.",
    "pl": "To może być stan nagły. Zadzwoń natychmiast na numer alarmowy 112 lub do ośrodka toksykologicznego. "
          "Nie czekaj na odpowiedź online.",
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
# (valid in every EU state) + that country's national register. Planned: at, pl, ua (see README).
MARKETS = {"us": "United States (FDA)"}


def normalize_citations(text: str) -> str:
    """Some models cite in their own style: 【2】, 【1†L1-L3】, [1, 2] or [1-3]. Convert all of that
    to plain [n] markers, which the UI turns into clickable source badges."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[(\d+)\s*[-–]\s*(\d+)\]",
                  lambda m: "".join(f"[{i}]" for i in range(int(m[1]), int(m[2]) + 1) if int(m[2]) - int(m[1]) < 10), text)
    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: "".join(f"[{n.strip()}]" for n in m[1].split(",")), text)
    return re.sub(r"\s+(\[\d+\])", r"\1", text)


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
        reply = llm.generate(TRIAGE_PROMPT.format(language=LANGUAGES[out_lang]),
                             f"Question: {question}\n\nWrite any answer in {LANGUAGES[out_lang]} "
                             f"(regardless of the question's language).").strip()
    except Exception:  # noqa: BLE001 -- the fallback is optional; never fail the request because of it
        return NOT_FOUND[out_lang], "none"
    head = reply[:20].upper()
    if "EMERGENCY" in head:
        return EMERGENCY[out_lang], "emergency"
    for tag, mode, limit in (("GENERAL:", "general", 1500), ("JOKE:", "playful", 400)):
        if reply.upper().startswith(tag):
            text = normalize_citations(reply[len(tag):]).strip()
            text = re.sub(r"\[\d+\]", "", text)  # a general answer has no sources, so no citations
            if text and len(text) <= limit:
                return text, mode
    return NOT_FOUND[out_lang], "none"


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

    # 0) QUERY REWRITING: the labels and the embedding model are English, so a Ukrainian/Polish
    #    question is translated first. The answer is still written in the user's language.
    search_q = question if lang == "en" else \
        timed("translate_ms", lambda: translate_query(question)) or question
    drugs, corrections = detect_drugs_fuzzy(search_q)
    base = {"detected_drugs": drugs, "corrections": corrections, "language": lang, "answer_language": out_lang,
            "market": market, "search_query": search_q if lang != "en" else None}
    def finish(result: dict) -> dict:
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000)
        print(json.dumps({"event": "ask", "lang": lang, "found": result["found"], **timings}), flush=True)
        return {**result, "timings": timings}

    def not_found() -> dict:
        text, mode = timed("refusal_ms", lambda: fallback(question, out_lang))
        return finish({**base, "answer": text, "found": False, "mode": mode,
                       "playful": mode == "playful", "sources": [], "model": llm.last_model_used})

    # 1) RETRIEVE
    chunks = store.search(search_q, top_k=top_k, drugs=drugs or None, timings=timings)
    relevant = [c for c in chunks if c["score"] >= config.MIN_SCORE]
    if not relevant:
        return not_found()

    # 2) AUGMENT + 3) GENERATE
    def generate(language: str) -> str:
        msg = f"Sources:\n{build_context(relevant)}\n\nQuestion: {question}"
        if lang != "en":
            msg += f"\n(English version of the question: {search_q})"
        msg += f"\n\nWrite the answer in {LANGUAGES[language]}."
        return timed("llm_ms", lambda: llm.generate(SYSTEM_PROMPT, msg)).strip()

    answer = generate(out_lang)
    if (not answer or "NOT_FOUND" in answer) and out_lang != "en":
        # Some (smaller) models refuse when answer language != source language. Fall back to:
        # answer in English (same language as the sources), then translate that answer.
        english = generate("en")
        if english and "NOT_FOUND" not in english:
            answer = timed("llm_ms", lambda: llm.generate(
                TRANSLATE_ANSWER_PROMPT.format(language=LANGUAGES[out_lang]), english)).strip()

    if not answer or "NOT_FOUND" in answer:
        return not_found()

    answer = normalize_citations(answer)
    # Drop citations that point to non-existent sources (a small hallucination guard),
    # then record which source numbers the model actually cited.
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(relevant) else "", answer)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    sources = [
        {"id": i, **{k: c[k] for k in ("drug", "section", "text", "url", "score", "effective_time")},
         "brands": drug_brands().get(c["drug"], []), "cited": i in cited}
        for i, c in enumerate(relevant, start=1)
    ]
    return finish({**base, "answer": answer, "found": True, "mode": "sourced", "sources": sources, "model": llm.last_model_used})
