# Testing plan and status

How the test suite is organised, what is done, what is next, and the open suggestions found on the way.

## How to run

```bash
python -m pytest                      # all fast tests (no network, no model, no API key), ~3 s
python -m pytest tests/test_markets.py   # one group
python -m pytest -q --cov=app --cov=scripts.doc_text --cov-report=term-missing   # with line coverage
```

Counting: pytest counts *cases*, not functions. A function with `@pytest.mark.parametrize` over 6 languages
is 1 function but 6 cases. Both numbers are given below.

## Rules we follow

1. **One test = one mistake it catches.** The docstring starts with `Catches: ...`. If you cannot name
   the mistake, the test is probably not needed.
2. **Red first.** When a test is written for a bug, run it before the fix (it must fail), then fix, then
   run again (it must pass). A test that was never red proves nothing.
3. **Mutation check.** After a group is green, break the code on purpose in 3-10 ways (flip a condition,
   empty a list, change a constant) and confirm at least one test fails each time. If a mutation
   survives, add a test for it.
4. **Coverage is a map, not a score.** The "Missing" column tells you where to look. 100 % is not a goal;
   a line that only forwards a call does not need a test. A line that *decides* something does.
5. **Fast tests need nothing external.** No Qdrant, no embedding model, no LLM key. Those go in
   `tests/integration/` (group 10) and run by hand.

## Status

| # | Group | File | Functions | Cases | Status | App code changed |
|---|---|---|---|---|---|---|
| 0 | original suite | `test_rag.py` | 32 | 46 | done | `rag.py`: citation ranges clipped instead of deleted, `[source n]` accepted, `uncited_sentences` counter, explicit no-range rule in the prompt |
| 1 | leaflet parsing (PL/UA/CZ/ES/FR) | `test_doc_text.py` + `fixtures/leaflets/` | 22 | 45 | done | `doc_text.py`: QRD headings matched accent-blind (`fold_accents`) and with `4.` / `4 -` / `4)` numbering |
| 2 | six languages | `test_languages.py` | 16 | 61 | done | `rag.py`: NOT_FOUND + EMERGENCY texts for cs/es/fr; `data/facts.json`: text_cs/es/fr |
| 3 | product-name detection | `test_markets.py` | 16 | 45 | done | none |
| 4 | hybrid re-ranking | `test_rerank.py` | 25 | 90 | done | `rag.py`: named children's product not penalised; more child words in `_CHILD`; Ukrainian word forms from a dictionary (`uk_forms`, `keyword_groups`, `keyword_hits`); per-language stop list `_STOP_BY_LANG` |
| 5 | `ask()` edge branches | `test_ask_edges.py` | ~9 | | **next** | |
| 6 | LLM wrapper | `test_llm.py` | ~8 | | todo | |
| 7 | HTTP endpoints | `test_api.py` | ~8 | | todo | |
| 8 | embeddings + Qdrant config | `test_store.py` | ~5 | | todo | |
| 9 | chunk building / ingest | `test_ingest.py` | ~5 | | todo | |
| 10 | real retrieval (slow, manual) | `tests/integration/` | ~4 | | todo | |
| | **total now** | | **111** | **287** | | |

Line coverage after group 4: `rag.py` 95 %, `markets.py` 100 %, `doc_text.py` 95 %,
`main.py` 65 %, `feedback.py` 90 %, `llm.py` 18 %, `store.py` 20 %, `ingest_markets.py` 0 %.

## Remaining groups (detail)

### 5. `test_ask_edges.py` – branches of `rag.ask()`
- `US_FULL=True` path (production US path; the market file is not in the repo, so it is never exercised)
- triage LLM raises → mode `"none"` with the refusal text
- `GENERAL:` reply longer than 1500 chars → `"none"`; `JOKE:` longer than 400 → `"none"`
- two products in one question → both groups in the Qdrant filter
- `equivalents` is empty when the asked product's own leaflet is the one cited
- `translate_ms` / `embed_ms` appear in `timings`; `search_query` set for a non-English US question
- `rag.ask(top_k=...)` passes `top_k * 5` to the store

### 6. `test_llm.py` – `app/llm.py` (mock `httpx.post` and `time.sleep`)
- `_with_retries`: retries on 429/5xx/"overloaded", raises at once on 401, backoff 1-2-4 s
- circuit breaker: `_trip` blocks a provider for 300 s, `_available` after expiry
- provider order groq → gemini; a failing groq trips and gemini is used
- Groq model renamed (400 "model") → picks another model from `/models` and remembers it
- `last_model_used` set per provider; all providers failing raises the last error

### 7. `test_api.py` – `app/main.py` (FastAPI `TestClient`)
- `/api/ask`: 422 on question < 3 chars, `top_k` 99, unknown market / language; 502 when store or LLM raises
- `/api/health`: degrades to `"error: ..."` when Qdrant is down, still 200
- `/api/drugs?market=`, `/api/fact?lang=` + `seen=`
- `/api/feedback`: 6th request from one IP → 429; image > 4 MB → 400; not configured → 502
- startup warm-up never blocks (exception swallowed)

### 8. `test_store.py` – `app/store.py` (mock `QdrantClient`)
- `recreate_collection(compact=True)` sends `ScalarQuantization(INT8, always_ram=True)` and `on_disk=True`;
  `compact=False` sends neither; keyword index created per field
- `remote_embed`: 2-D token output → mean pooling; result is unit length; HTTP ≥ 400 raises
- `embed_query`: remote failure falls back to the local model
- `search`: filter uses the given `field` (`drug` vs `group`); timings filled

### 9. `test_ingest.py` – `scripts/ingest_markets.py`, `scripts/ingest.py`
- `build_chunks` skips `SKIP_SECTIONS`; keeps group/inn/en/form/url/source/market on every chunk
- embedded text has the prefix `"Product (inn; en) - Section: ..."` (the index was built with it)
- `point_id` is deterministic (same chunk → same UUID; re-run overwrites, not duplicates)
- `popular_groups` keeps the top-N plus the starter list

### 10. `tests/integration/` – slow, run by hand (`pytest -m integration`)
- build a local Qdrant from ~20 PL + CZ leaflets with the real multilingual model
- Polish question finds a Czech passage; "alkoholem" query ranks the alcohol passage first
- int8 vs float32 query vector: cosine ≈ 1.0; score of a correct hit ≥ `MIN_SCORE_MULTI`
- plus: extend `eval/questions.jsonl` with ~30 PL/UA/CZ/ES/FR and cross-language questions

## Open suggestions (found while testing, not changed)

| # | Where | What | Why it matters | Effort |
|---|---|---|---|---|
| S4 | `markets.detect` | prefix rule: a word with the same 6-letter stem and ≤ 3 extra letters matches ("nuroferon" → Nurofen) | rare false hit; watch the logs | – |
| S5 | `tests/test_rag.py::pl_market` | uses a `json.loads` monkeypatch; `test_markets.py::registers` writes real files | migrate the old fixture for consistency | 10 min |
| S6 | `data/markets/*.json` | not in the tarball although the README says "committed" | `US_FULL` and the market library cannot be tested on real data | – |
| S7 | `rag.EMERGENCY` cs/es/fr | translations written by the assistant | red safety card: a native check is worth 2 minutes | 2 min |
| S8 | `data/facts.json` cs/es/fr | same | loading-screen text only | 5 min |

## Fixed so far (with the test that pins it)

| Fix | Test |
|---|---|
| NOT_FOUND / EMERGENCY texts for cs, es, fr | `test_languages.py::test_refusal_and_emergency_texts_exist_in_every_language` |
| facts translated to cs, es, fr | `test_rag.py::test_facts_are_translated_into_every_ui_language` |
| children's product named by the user is not penalised in rerank | `test_rerank.py::test_named_children_product_is_not_penalised` |
| `_CHILD` recognises baby/junior/niemowl/малюк/pediátr/bébé/kojenc/… | `test_rerank.py::test_more_child_words_are_recognised` |
| Ukrainian grammatical cases (вагітність / вагітності, біль / болю, ніч / ночі): question words are looked up in the `pymorphy3` Ukrainian dictionary (+21 MB RAM, ~1 ms per question) and every form of the word counts as a whole-word hit; the 2-letter stem rule is kept as a safety net (алкоголь → алкогольних) and is the only rule for unknown words (brand names) and other languages; for known words with an uncut 4-letter stem (`біль`) the dictionary decides alone, so `біль` no longer hits `більше`. Was S2 / an `xfail`. |
| EU headings printed in CAPITALS without accents (`4. QUELS SONT LES EFFETS INDESIRABLES`, `COMO TOMAR`) were not recognised, so the side-effects text was glued to the dosage section. Headings are now matched on an accent-folded copy of the text (same length, so the original text is cut and saved untouched); `4 -`, `4 –`, `4)` and `4:` numbering accepted. **Needs a re-ingest of fr/es (and cz/pl to be safe) to reach the index.** | `test_doc_text.py::test_headings_in_capitals_with_or_without_accents_are_found`, `::test_heading_number_separator_variants_are_accepted`, `::test_accent_folding_keeps_section_text_intact` |
| Citation ranges: `[1-20]` (10+ sources) was deleted together with its marker, leaving the sentence unsourced. Now every range is expanded and clipped to the real number of sources (`[1-20]` with 6 sources → `[1]..[6]`, `[4-2]` → `[2][3][4]`, cap 30 without a count), `[source 2]`/`[src 2]` accepted, a vanished marker leaves no stray space, the prompt forbids ranges explicitly and asks for 1-3 real sources per fact, and the response/log carries `uncited_sentences` (sentences or bullets without any `[n]`). Was S3. Equivalent mutation noted: `ask()` passing no source count gives the same result thanks to the older invalid-number guard. | `test_rag.py::test_citation_ranges_are_never_silently_deleted`, `::test_answer_with_a_wide_range_keeps_valid_citations_end_to_end`, `::test_uncited_sentences_are_counted`, `::test_prompt_forbids_citation_ranges_explicitly` |
| Stop words: `during`, `podczas`, `під час`, `během`, `durante`, `pendant`, `this/ten/цей/tento/este/ce`, generic medicine nouns (`medicine`, `lek`, `препарат`, `lék`, `medicamento`, `médicament`) were keywords, so "take during meals" earned the same bonus as a pregnancy passage. `_STOP` is now built from a readable per-language list (`_STOP_BY_LANG`, ~50-70 words each, hand-curated: no medically meaningful words, form words like tablet/syrup kept). Was S1. | `test_rerank.py::test_function_words_and_generic_medicine_words_are_not_keywords`, `::test_during_meals_passage_gets_no_bonus_for_a_pregnancy_question`, `::test_medically_meaningful_words_are_never_stop_words` | `test_rerank.py::test_ukrainian_case_change_in_the_middle_of_the_word_still_matches` (+6 tests around it) |
