# To do after the bug fixes (things only you can run)

## 1. Ukrainian grammatical cases (bug #1) – new library

`requirements.txt` now has two extra packages (`pymorphy3`, `pymorphy3-dicts-uk`).

- Locally: `pip install -r requirements.txt`
- Render: nothing to do, the Docker image is rebuilt on the next deploy (+14 MB disk, +21 MB RAM).
- No re-ingest needed: this fix works at question time only.

## 2. French / Spanish headings in CAPITALS (bug #2) – re-ingest needed

The fix is in the *parser* (`scripts/doc_text.py`). Leaflets already in Qdrant were split with the old
parser, so their side-effects text is still filed under "How to take it" until you re-index.

1. Make sure `.env` has `QDRANT_URL` and `QDRANT_API_KEY` (your Qdrant Cloud), otherwise the script
   writes to a local folder, not to the live database.
2. Make sure the downloaded leaflets still exist: `data/raw/fr/docs.jsonl` and `data/raw/es/docs.jsonl`.
   If not: `python -m scripts.fetch_eu` (downloads ~5k leaflets, takes a while).
3. Quick check first (a few minutes, 30 leaflets):
   ```bash
   python -m scripts.ingest_markets --markets fr --limit 30
   python -m scripts.debug_ask fr "Quels sont les effets indésirables de l'ibuprofène ?"
   ```
   The sources should now show the section "Possible side effects".
4. Full rebuild, one country at a time, at a quiet time (the market is empty/partial while it runs):
   ```bash
   python -m scripts.ingest_markets --markets fr     # ~65k chunks, roughly 1–2 h on a laptop CPU
   python -m scripts.ingest_markets --markets es     # ~81k chunks, roughly 1–2 h
   ```
   Optional, to be safe: `--markets cz` and `--markets pl` (same accent-blind rule applies to them).
5. Do NOT use `--append` or `--resume` for this: they keep the old, wrongly split chunks.
   The default run deletes and rebuilds the whole collection, so there are no duplicates.

## How to verify everything still works

```bash
python -m pytest          # expected: all green (236 tests after fix #2)
```

---

# Summary of today's four fixes

## Files changed
| File | What |
|---|---|
| `app/rag.py` | bugs 1, 3, 4 (Ukrainian word forms, citation ranges + `uncited_sentences`, per-language stop list, prompt rule) |
| `scripts/doc_text.py` | bug 2 (accent-blind headings, `4 -` / `4)` numbering) |
| `requirements.txt` | bug 1 (`pymorphy3`, `pymorphy3-dicts-uk`) |
| `tests/test_rerank.py` | tests for bugs 1 and 4 (old `xfail` removed) |
| `tests/test_doc_text.py` | tests for bug 2 |
| `tests/test_rag.py` | tests for bug 3 |
| `tests/TESTING.md` | counts, S1/S2/S3 moved to "Fixed so far" |
| `TODO_after_fixes.md` | this file (new) |

## Which test checks which bug

**Bug 1 – Ukrainian grammatical cases (вагітність / вагітності, біль / болю)** – `tests/test_rerank.py`
- `test_ukrainian_case_change_in_the_middle_of_the_word_still_matches` – 7 word pairs, both directions
- `test_ukrainian_forms_are_matched_as_whole_words` – "біль" must not hit "більше"
- `test_unknown_ukrainian_word_falls_back_to_the_old_stem_rule` – brand names / typos
- `test_ambiguous_ukrainian_word_takes_forms_of_every_reading` – "солі" -> сіль and сол
- `test_old_stem_rule_is_kept_for_known_ukrainian_words` – "алкоголь" still hits "алкогольних"
- `test_one_point_per_question_word_however_many_forms_appear`
- `test_ukrainian_pregnancy_passage_wins_on_equal_vector_score` – end-to-end rerank
- `test_non_ukrainian_stems_are_unchanged_by_the_dictionary`

**Bug 2 – French/Spanish headings in CAPITALS without accents** – `tests/test_doc_text.py`
- `test_headings_in_capitals_with_or_without_accents_are_found` – PL/CZ/ES/FR x with/without accents
- `test_heading_number_separator_variants_are_accepted` – `4.` `4 -` `4 –` `4)` `4 `
- `test_accent_folding_keeps_section_text_intact` – saved text keeps its accents

**Bug 3 – citation ranges `[1-20]` deleted** – `tests/test_rag.py`
- `test_citation_ranges_are_never_silently_deleted` – clipping, reversed range, cap, `[source 2]`
- `test_answer_with_a_wide_range_keeps_valid_citations_end_to_end`
- `test_uncited_sentences_are_counted`
- `test_prompt_forbids_citation_ranges_explicitly`

**Bug 4 – "during" etc. counted as keywords** – `tests/test_rerank.py`
- `test_function_words_and_generic_medicine_words_are_not_keywords` – 7 languages
- `test_during_meals_passage_gets_no_bonus_for_a_pregnancy_question`
- `test_medically_meaningful_words_are_never_stop_words` – 34 medical words guard

## Run only today's tests (79 cases)
```bash
python -m pytest -q -k "ukrainian or headings_in_capitals or heading_number_separator or accent_folding or citation_ranges or wide_range or uncited_sentences or forbids_citation or function_words or during_meals or medically_meaningful"
```
Per bug:
```bash
python -m pytest -q -k "ukrainian"                                                        # bug 1
python -m pytest -q -k "headings_in_capitals or heading_number_separator or accent_folding"  # bug 2
python -m pytest -q -k "citation_ranges or wide_range or uncited_sentences or forbids_citation"  # bug 3
python -m pytest -q -k "function_words or during_meals or medically_meaningful"           # bug 4
python -m pytest -q                                                                       # everything: 287
```
