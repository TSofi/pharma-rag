# Testing plan and status

How the test suite is organised, what is done, what is next, and the open suggestions found on the way.

## How to run

```bash
python -m pytest                      # all fast tests (no network, no model, no API key), ~3 s
python -m pytest tests/test_markets.py   # one group
python -m pytest -q --cov=app --cov=scripts.doc_text --cov-report=term-missing   # with line coverage
python -m pytest tests/test_llm.py tests/test_llm_answers.py tests/test_emergency.py   # only the LLM-reliability work (116)
python -m pytest tests/test_citation_truth.py tests/test_citation_units.py   # "citations are true" (22 + 22)
python -m pytest tests/test_meaning_check.py   # the meaning check / judge (51)
python -m pytest tests/test_api_errors.py   # HTTP errors: exact numbers + one JSON shape (49)
python -m pytest tests/test_api_input.py    # input checks: codes + hints, question cleaning, no letters (79)
python -m pytest tests/test_api_fields.py   # unknown JSON fields -> 400 (strict) + frontend stays in sync (17)
python -m pytest tests/test_api_limits.py   # questions per visitor: 10/min, 200/day -> 429 (29)
python -m pytest tests/test_api_report.py   # report form spam checks: time trap, same text, caps, links, real images (49)
python -m pytest -k citation        # only tests whose name contains a word
python -m pytest --lf               # only the tests that failed last time
python -m pytest -v > out.txt       # every test name + result, saved to a file
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
| 6 | LLM wrapper + what the user gets on LLM failure | `test_llm.py` (40 cases), `test_llm_answers.py` (17 cases) | 42 | 57 | done, **frozen** | `llm.py`: time limits, empty/cut-off answers, probe, typed `LLMUnavailable`; `rag.py`: citation repair → orange, translation guards; `main.py`: 503 + Retry-After |
| 6b | emergency levels 🔴🟠⚪ (LLM `RISK:` line; keywords when the LLM is down) | `test_emergency.py` | 18 | 59 | done, **frozen** | `rag.py`: `RISK:` line in `SYSTEM_PROMPT`, `split_risk`, `SAFETY_NOTE`, `EMERGENCY_SOFT`, `looks_like_emergency`; `app.js`/`style.css`: safety line + retry button |
| 6c | citations are TRUE: every number of a cited sentence is in the cited source | `test_citation_truth.py` | 13 | 22 | done | `rag.py`: `wrong_numbers` + `NUMBERS_NOTE` (one repair, then orange); test data: 1 line in `test_emergency.py` `CHUNKS[0]`, 1 line in `test_rag.py::mocked_pipeline` (the fake leaflets did not contain the numbers the fake answers cite) |
| 6c-2 | units in the number check: "1 g" = "1000 mg", but "500 µg" ≠ "500 mg" | `test_citation_units.py` | 5 | 22 | done | `rag.py`: `_UNITS`, `_AMOUNT`, `_amounts`; a number WITH a unit is compared as an amount (mass/volume, base mg/ml) |
| 6d | MEANING of a cited sentence matches its source (LLM judge, risky topics only) | `test_meaning_check.py` | 18 | 51 | done | `rag.py`: `JUDGE_PROMPT`, `MEANING_NOTE`, `risky_topic`, `judge`, `parse_verdicts`; `config.py`/`.env.example`: `JUDGE_MODE`, `JUDGE_STRICT`; test logic: 1 line in `test_languages.py` (it assumed the LAST LLM call is the answer; now the judge can come after it) |
| 7a | HTTP errors: exact numbers (400/404/405/413/429/500/502/503) + one JSON shape | `test_api_errors.py` | 24 | 49 | done | new `errors.py` (validation → 400, body size limit → 413, crash → JSON 500 with id, 404/405 for `/api/...`, `looks_busy`); `main.py`: `errors.install(app)` before CORS, `/api/ask` 502 without internal details / busy → 503, `/api/feedback` not set up → 503, 429 + Retry-After, bad image → 400, mail failure → 502 without details |
| 7b | input checks: language/country codes (+ hints for mix-ups), question cleaned, empty / no letters → 400 | `test_api_input.py` | 16 | 79 | done | new `inputs.py` (`AnswerLanguage`, `UiLanguage`, `Market`, `Question`, `clean_text`); `main.py`: fields use these types instead of `Literal` / `min_length`; `errors.py`: messages of our own checks (`app_*`) |
| 7c | unknown JSON fields → 400 (strict); the frontend's fields are read from `app.js` and must be known | `test_api_fields.py` | 8 | 17 | done | `main.py`: `model_config = ConfigDict(extra="forbid")` on `AskRequest` and `FeedbackRequest` |
| 7d | limit on questions per visitor (10/min, 200/day → 429); real visitor = last address in the proxy note; report form uses it too | `test_api_limits.py` | 18 | 29 | done, **needs a check on Render** (see report) | new `limits.py` (`client_ip`, `visitor_key`, `mask`, `RateLimit`, `limit_ask`); `main.py`: `limits.limit_ask(request)` in `/api/ask`, report form counts by `limits.client_ip`; `config.py` + `.env.example`: `ASK_LIMIT_PER_MINUTE`, `ASK_LIMIT_PER_DAY`; `tests/conftest.py`: counters reset before every test |
| 7e | report form spam checks without accounts: time trap (< 3 s or no timer = bot), same visitor + same text within 24 h, 5/hour per visitor, 90 e-mails/day in total, message must have letters and ≤ 2 links, contact cleaned, screenshot must really be an image, counters forget old visitors | `test_api_report.py` | 25 | 49 | done | new `spam.py`; `inputs.py`: `ReportMessage`, `Contact`, `clean_message`; `feedback.py`: `is_real_image`, old `_recent`/`allowed` removed; `limits.py`: `RateLimit` forgets visitors with nothing left in the window; `main.py`: new field `form_ms`, report endpoint uses `spam`; `config.py` + `.env.example`: `FEEDBACK_LIMIT_PER_HOUR`, `FEEDBACK_LIMIT_PER_DAY`; `frontend/app.js`: timer started when the form opens, `form_ms` sent |
| 7 | other HTTP endpoints (health, drugs, fact, warm-up) | `test_api.py` | ~5 | | todo | |
| 8 | embeddings + Qdrant config | `test_store.py` | ~5 | | todo | |
| 9 | chunk building / ingest | `test_ingest.py` | ~5 | | todo | |
| 10 | real retrieval (slow, manual) | `tests/integration/` | ~4 | | todo | |
| | **total now** | | **298** | **721** | | |

Line coverage after group 6: `rag.py` 95 %, `markets.py` 100 %, `doc_text.py` 95 %, `llm.py` 88 % (was 18 %),
`main.py` 72 %, `feedback.py` 90 %, `store.py` 20 %, `ingest_markets.py` 0 %. In `llm.py` the missing lines are
mainly the Anthropic provider (not used in production, see S13).

**Frozen files.** `test_llm.py` (40 cases), `test_llm_answers.py` (17) and `test_emergency.py` (59) are done:
do not add to or change them. New work goes into new files, and these three must still show
40 + 17 + 59 = 116 passed after every change to the app code. One agreed exception (group 6c): one line of
fake leaflet text in `test_emergency.py` (`CHUNKS[0]` now says "Adults: 1 tablet; never within 4 hours of
the next dose"), because the number check correctly found that two fake answers cited a passage without
their number (the re-ranker puts that passage at [1] or [2]). Test logic and the count (59) unchanged.

## Report: LLM reliability (groups 6 – 6d)

**Goal.** Answers come only from official leaflets and every sentence cites its passage. Check what happens
when the LLM misbehaves (Groq timeout / 429, Gemini also down, no citation, wrong citation…), cover it with
tests and fix what is broken.

**In short, the new tests cover:** (1) the LLM being slow, rate-limited, offline or returning garbage,
(2) what the user sees then, (3) emergencies, (4) whether citations are TRUE: present, with the right numbers
and units, and the right meaning. All run without internet or API keys (fake Groq / Gemini / clock), ~3 s.

| New test file | Cases | What it checks |
|---|---|---|
| `test_llm.py` | 40 | Groq / Gemini layer: retries, timeouts, fallback, circuit breaker, bad replies |
| `test_llm_answers.py` | 17 | what the user gets on LLM trouble: busy message, orange for uncited, translation, model label |
| `test_emergency.py` | 59 | emergency levels 🔴🟠⚪, with the LLM working and with the LLM down, 6 languages |
| `test_citation_truth.py` | 22 | every number of a cited sentence is in the cited source |
| `test_citation_units.py` | 22 | the same with units (1 g = 1000 mg, but 500 µg ≠ 500 mg) |
| `test_meaning_check.py` | 51 | an LLM judge checks the meaning of cited sentences on risky topics |
| **total** | **211** | whole suite: 498 passed, 0 xfailed |

### `test_llm.py` (40) – Groq / Gemini layer
Found (reproduced with fakes before fixing) → fixed:
- Both providers timing out: the user waited **120 s**, then saw a raw error → one Groq try ≤ 12 s
  (`GROQ_TIMEOUT_S`), all LLM calls of one question ≤ 30 s (`LLM_DEADLINE_S`).
- After a total outage **every question failed instantly for 5 minutes**, even if the providers were back
  after 10 s → one test call ("probe") every 10 s, site recovers in ~10 s.
- An **empty / cut-off / safety-blocked** answer was treated as "the leaflet doesn't say" → now it counts as
  a failure and the next provider is asked.
- One **too-long question** (400 "context length") permanently switched Groq to another model and benched
  Groq for everyone → model swap only on "model not found / decommissioned"; a bad request doesn't bench.
- "15004 tokens" was read as error 500 (text search) → the status code is read properly.
- The backup-model lookup had no timeout → 10 s.

### `test_llm_answers.py` (17) – what the user sees
- All providers down → **503 "The AI service is busy right now, try again in 10 s"** + Retry-After; the UI
  shows its "busy, retry" message (before: "Something went wrong" + raw SDK text).
- An answer with **no citation was shown violet** ("confirmed by the leaflet") → one repair attempt, still
  none → **orange**. Same for an answer citing only non-existent sources ([7] of 2).
- `<think>…</think>` notes of reasoning models were shown → removed.
- Translation step failed → a good cited English answer was thrown away (502) → the English answer is shown.
  Translation lost every [n] → the English answer is shown.
- The model name in the UI could come from another user's request → tracked per question.

### `test_emergency.py` (59) – 🔴 / 🟠 / ⚪
- The red card appeared **only when the search found nothing**, so "I took 5 × 500 mg" got a normal violet
  answer → every answer now starts with a hidden `RISK:` line (no extra LLM call): 🔴 fixed "call 112 now"
  card, 🟠 leaflet answer + fixed safety line (`safety_note`), ⚪ normal.
- With the LLM down, an emergency question got an error page → keyword check (something ALREADY taken +
  a sign of too much, or "overdosed"…) → soft red card with a retry button; ordinary questions → "busy".
- False alarms tested too ("doctor said crush the pill into food", "what if I take two", "forgot my pill").

### `test_citation_truth.py` (22) – numbers in citations
- A citation was only checked to EXIST: "take 3 tablets [1]" was violet even if [1] says 2 → every number of
  a cited sentence must be in the source it cites (no extra LLM call) → one repair naming the number (and the
  source that really has it) → still wrong → orange.
- No false alarms for 2,5 = 2.5, 1 000 = 1000, мг vs mg, "dwie" = 2, ranges 6-8, list numbers, B12.
- The check immediately found 2 fake answers in our own tests citing a passage without their number.
- Checked by hand on the real PL/UA fixture leaflets: correct answers pass (also translated); an invented
  "1200 mg", a wrong maximum "5 tablets", "2400 mg" are caught.

### `test_citation_units.py` (22) – units
- "1 g" vs "1000 mg" was counted as wrong → amounts are compared in mg / ml (mcg, µg, мкг, mg, мг, g, г,
  kg, кг, ml, мл, l, л). Also catches what the number-only check missed: "500 mg" for "500 µg",
  "1200 g" for "1200 мг". Mass never matches volume; "2 glasses" is not "2 g".

### `test_meaning_check.py` (51) – meaning (LLM judge)
- "Safe in pregnancy [1]" while [1] says "do not use" was violet → on risky topics (pregnancy,
  breastfeeding, children, alcohol, driving, other medicines, allergy, kidney/liver – in the question OR the
  answer) a second LLM call judges each cited sentence against ONLY its cited sources: SUPPORTED /
  CONTRADICTED / NOT_IN_SOURCE → a problem → one repair → checked again → still wrong → orange.
- Settings: `JUDGE_MODE=risky|always|off`, `JUDGE_STRICT=true|false`; judge down → answer stays violet,
  logged (`meaning_check: "unavailable"`). The judge prompt says translation / paraphrase = SUPPORTED.

### Changes to existing tests (test data / 1 line each, no count changed)
- `test_emergency.py` (frozen, agreed): 1 line of fake leaflet text (`CHUNKS[0]`), because the number check
  correctly found that two fake answers cited a passage without their number.
- `test_rag.py::mocked_pipeline`: 1 line of fake leaflet text ("type 2", "2.5 mg" were missing).
- `test_languages.py`: 1 line – it assumed the LAST LLM call is the answer; now the judge can come after it.

### How it was verified
- **Red first:** every bug was reproduced with a failing test (`xfail(strict)` or a red run) before the fix.
- **Mutation checks:** the new code was broken on purpose about 80 times; every mutation that survived either
  got a stronger test or was shown to be equivalent (same behaviour, e.g. a double check).
- Real leaflet text (fixtures) for the number/unit check; a simulated outage through the real API.

### Still open (for real-API testing and later)
- **Real LLM behaviour is not tested** (all providers are fakes): does the model write the `RISK:` line and
  pick the right level, does it cite, how accurate is the judge (how many wrong answers it catches, how many
  correct ones it turns orange – especially translated answers). Watch the logs: `risk`, `meaning_check`,
  `uncited_answer_rejected`, `wrong_numbers_rejected`, `wrong_meaning_rejected`.
- **Internet fully offline** (Qdrant / embeddings down too): the user now gets a clean 502 "search service is
  not responding" (group 7a; was a raw "ConnectError: ..."). LLM alone offline → "busy" (503) works;
  emergency questions still get the red card. Retries for Qdrant/embeddings: group 8.
- Native check of the texts and keyword lists written by the assistant (S9, S19).
- Topics not yet "risky": contraindications, overdose, boxed warning (S20).
- Keyword limits while the LLM is down (S10); frontend checked only by eye (S15).

## Report: API errors (group 7a, `test_api_errors.py`)

**Goal.** Every error of the HTTP API has an exact number and the same JSON shape, so the frontend can always
show it and the user knows what to do. Internal details (exception text, Qdrant URL, keys) never leave the server.

```bash
python -m pytest tests/test_api_errors.py        # only this batch: 49 cases, ~2 s
python -m pytest tests/test_api_errors.py -v     # with every test name
```

Error shape: `{"detail": "<readable text>", "error": "<CODE>", "field": "<name>" (400 only), "request_id" (500 only)}`.
`detail` is always a string (before: 422 gave a list and the UI showed "[object Object]").

| Status | When | Code(s) | Tests |
|---|---|---|---|
| 400 | broken JSON; body missing / not an object; wrong field (missing, too short/long, not text, out of range, not in the list); bad `?lang=` / `?market=` | `INVALID_JSON`, `INVALID_BODY`, `MISSING_FIELD`, `TOO_SHORT`, `TOO_LONG`, `WRONG_TYPE`, `OUT_OF_RANGE`, `INVALID_VALUE`; report: `INVALID_IMAGE` | `test_broken_json_is_400_with_a_clear_message`, `test_body_that_is_not_a_json_object_is_400`, `test_wrong_field_is_400_naming_the_field`, `test_bad_query_parameter_is_400`, `test_error_never_sends_the_input_back`, `test_a_screenshot_that_is_not_an_image_is_400` |
| 404 / 405 | unknown `/api/...` address / right address, wrong method (`Allow` names the right one) | `NOT_FOUND`, `METHOD_NOT_ALLOWED` | `test_unknown_api_address_is_a_json_404`, `test_unknown_api_address_is_404_whatever_the_method`, `test_wrong_method_is_405_and_says_which_one_is_right` |
| 413 | body > 16 KB (`/api/feedback`: > 6.5 MB), also when streamed without Content-Length | `TOO_LARGE` | `test_too_big_question_request_is_413_and_never_reaches_the_pipeline`, `test_too_big_report_is_413_with_a_short_answer`, `test_size_limit_also_works_without_a_content_length_header`, `test_a_normal_4_mb_screenshot_is_not_blocked_by_the_size_limit` |
| 429 | 6th report from one IP within an hour, `Retry-After: 3600` | `RATE_LIMITED` | `test_too_many_reports_is_429_with_a_wait_time` |
| 500 | a bug (any unexpected exception): short message + `request_id`; full traceback + id in the log | `INTERNAL_ERROR` | `test_a_crash_is_a_json_500_without_details_and_the_details_go_to_the_log` |
| 502 | search pipeline or mail service failed: generic text, details in the log | `UPSTREAM_ERROR` | `test_search_failure_is_502_without_internal_details`, `test_mail_service_failure_is_502_without_details` |
| 503 | LLM busy (unchanged text) / another service busy ("503", "429", "overloaded"… in its error) / report form not set up | `LLM_UNAVAILABLE`, `SERVICE_BUSY`, `FEEDBACK_UNAVAILABLE` | `test_llm_busy_stays_a_503_the_ui_recognises`, `test_a_busy_service_is_a_clean_503_the_ui_offers_to_retry`, `test_a_number_inside_another_number_is_not_a_busy_signal`, `test_report_form_not_set_up_is_503` |
| all | error answers keep the CORS header (otherwise the browser hides them: "API unreachable"); a valid question and the website are unchanged | | `test_error_answers_keep_the_cors_header`, `test_a_valid_question_still_reaches_the_pipeline_unchanged`, `test_the_website_itself_is_still_served` |

**Found on the way.**
- The first version turned a raw "503 overloaded" into a generic 502; the frozen
  `test_llm_answers.py::test_outage_reaches_the_browser_as_an_error_status` caught it (the UI must show "busy, try
  again"). Fixed in the app (`errors.looks_busy` → clean 503 `SERVICE_BUSY`), the frozen test is unchanged.
- Non-GET requests to `/api/...` fell through to the static frontend: `DELETE /api/ask` answered "405, use GET",
  `POST /api/nothing-here` answered 405 instead of 404. Red test first, then fixed in `errors._http_error`.

**How it was verified.** Red first (37 of 40 failed before the code; the 3 green ones guard normal behaviour).
12 deliberate mutations (no 400 handler, no field name, size counted only from Content-Length, limit too low
for a 4 MB screenshot, crash details shown, errors not installed, installed after CORS, no 405, busy pattern
without word boundaries, no Retry-After, not-set-up not 503, + 1 harmless control): every real one was caught.
Checked on a real `uvicorn` server with `curl` (broken JSON, empty question, bad language, 30 KB body with and
without Content-Length, GET /api/ask, report not set up, website).

**Changes to existing tests:** none. Frozen files: 116 passed.

## Report: input checks (group 7b, `test_api_input.py`)

**Goal.** Stop requests that can't be answered BEFORE they cost a search + LLM call, and tell the user exactly
what to fix. Decision: do NOT block questions by language (a German or Chinese question still passes).

```bash
python -m pytest tests/test_api_input.py        # only this batch: 79 cases, ~3 s
python -m pytest tests/test_api_input.py -v     # with every test name
```

| What | Rule | Error | Tests |
|---|---|---|---|
| Language / country codes (`answer_language`, `market`, `?lang=`, `?market=`) | known code in any case / with spaces is accepted and cleaned (`"EN"` → `en`); unknown → list of allowed codes; known mix-up → hint: language `ua`→`uk`, `cz`→`cs`, `gb`/`us`→`en`; country `uk`→`ua`, `cs`→`cz`, `en`/`usa`→`us`; a hint only if that code is allowed there (`?lang=auto` gets none) | 400 `INVALID_VALUE` (+ `WRONG_TYPE` for a number / null / list) | `test_every_supported_answer_language_still_works`, `test_every_supported_country_still_works`, `test_code_in_capitals_or_with_spaces_is_accepted_and_cleaned`, `test_a_mixed_up_code_gets_a_hint`, `test_an_unknown_code_lists_the_allowed_ones_and_gives_no_false_hint`, `test_a_code_that_is_not_text_is_400`, `test_codes_in_the_address_are_cleaned_too`, `test_bad_codes_in_the_address_are_400_with_the_same_hints`, `test_a_hint_never_points_to_a_code_that_is_not_allowed_there`, `test_code_lists_match_the_rest_of_the_app` |
| Question cleaning | accents joined (e + ´ → é), invisible characters (zero-width, BOM) and broken characters (lone `\ud800`) removed, control characters (new line, tab, NUL) → space, spaces collapsed; the CLEANED text goes to the pipeline | – | `test_the_question_is_cleaned_before_it_reaches_the_pipeline`, `test_a_broken_character_is_removed_not_a_crash` |
| Question checks (after cleaning) | not empty; 3–500 characters; at least one letter in any alphabet | 400 `EMPTY`, `TOO_SHORT`, `TOO_LONG`, `NO_LETTERS` | `test_an_empty_question_is_400`, `test_a_question_with_no_letters_is_400`, `test_length_is_counted_after_cleaning`, `test_real_questions_in_any_language_still_pass_unchanged` |

**How it was verified.** Red first (43 of the new cases failed before the code). 15 deliberate mutations
(invisible / broken / control characters kept, no accent joining, no letter check, only Latin letters count,
length before cleaning, uncleaned text to the pipeline, "EN" refused, `ua` hint missing, hint not checked,
empty reported as too short, our messages not used, facts accepting `auto`, `/api/drugs` not checked): 14 caught
at once; the survivor (a hint pointing to a code not allowed there) was impossible with today's lists, so
`test_a_hint_never_points_to_a_code_that_is_not_allowed_there` was added and now catches it. Checked on a real
`uvicorn` server with `curl`. Swagger (`/docs`) still shows the code drop-downs and the 3–500 length.

**Changes to existing tests (agreed rule: 1 line each, logic unchanged).**
- `test_api_errors.py::test_wrong_field_is_400_naming_the_field[empty]`: `""` now gives `EMPTY` ("question is
  empty") instead of `TOO_SHORT`, a clearer message. Count unchanged (49).
- `test_languages.py::test_api_accepts_exactly_the_supported_languages`: reads the language list from
  `app/inputs.py` instead of the removed FastAPI `Literal` (+ its imports). Same check, count unchanged.

## Report: unknown fields (group 7c, `test_api_fields.py`)

**Goal.** A typo in a field name (`answerLanguage` instead of `answer_language`) was silently ignored: the answer
came in the wrong language and nobody knew why. Now the JSON body may contain ONLY the known fields (strict).
Decision: strict, because the owner deploys the frontend and the API together.

```bash
python -m pytest tests/test_api_fields.py        # only this batch: 17 cases, <1 s
python -m pytest tests/test_api_fields.py -v     # with every test name
```

| What | Rule | Tests |
|---|---|---|
| `/api/ask` | only `question`, `top_k`, `answer_language`, `market`; anything else → 400 `UNKNOWN_FIELD` naming it, nothing reaches the pipeline. A misspelled REQUIRED field (`Question`) → 400 `MISSING_FIELD` "'question' is required." | `test_unknown_field_in_a_question_is_400_naming_it`, `test_misspelled_required_field_is_400_and_never_reaches_the_pipeline` |
| `/api/feedback` | only `message`, `contact`, `context`, `image`, `website` (the hidden anti-bot field); anything else → 400, nothing mailed | `test_unknown_field_in_a_report_is_400_and_nothing_is_mailed`, `test_the_hidden_anti_bot_field_still_works` |
| Frontend in sync | the field names inside `JSON.stringify({...})` in `frontend/app.js` are read by the test and must all be known to the server, and exactly that request still gets 200 | `test_the_frontend_sends_only_fields_the_server_knows`, `test_exactly_what_the_frontend_sends_still_works` |
| Address parameters | extras in the address (`?v=2`, `?utm_source=...`) are still ignored (browsers/tools add them) | `test_extra_address_parameters_are_still_ignored` |
| Field lists | the accepted fields are listed in the test: adding/removing one is a conscious change (update the test and `app.js` together) | `test_the_accepted_fields_are_exactly_these` |

**How it was verified.** Red first (10 of 17 failed before the code; the 7 green ones guard what must keep
working). 7 deliberate mutations, all caught: questions not strict, reports not strict, honeypot field removed,
no "not a known field" message, `app.js` sending a new field in a question, in a report, a field added to the
server silently.

**When you add a field later:** add it to the request class in `main.py`, to `app.js`, and to `ASK_FIELDS` /
`FEEDBACK_FIELDS` in `test_api_fields.py`. If you forget the server side, `test_the_frontend_sends_only_fields_the_server_knows`
fails before deploy (otherwise every request from the website would get 400).

**Changes to existing tests:** none.

## Report: question limit per visitor (group 7d, `test_api_limits.py`)

**Goal.** Every question costs up to 4-5 LLM calls on the free Groq/Gemini keys. Without a limit one script could
use up the quota, and every real user would see "busy". Now: at most **10 questions a minute and 200 a day per
visitor** → HTTP 429 "too many questions, try again in N s". Only `/api/ask` is limited (website, health ping,
drug list, facts are not).

```bash
python -m pytest tests/test_api_limits.py        # only this batch: 29 cases, ~3-8 s
python -m pytest tests/test_api_limits.py -v     # with every test name
```

**Settings (`.env` locally, "Environment" on Render), no code change needed:**

| Setting | Default | Meaning |
|---|---|---|
| `ASK_LIMIT_PER_MINUTE` | 10 | questions per visitor per minute; `0` = no minute limit |
| `ASK_LIMIT_PER_DAY` | 200 | questions per visitor per 24 h; `0` = no daily limit |

Both `0` = the limit is off. Use this if real users are ever blocked by mistake (see "check on Render").

| What | Rule | Tests |
|---|---|---|
| Who is one visitor | the LAST address in the `X-Forwarded-For` note (Render's proxy adds the real one at the end; a bot can only add fakes at the start); no note or garbage → the direct caller; IPv6 counted per /64 block (one connection owns a whole block) | `test_the_visitor_is_the_last_address_in_the_proxy_note`, `test_ipv6_addresses_of_one_connection_count_as_one_visitor`, `test_fake_addresses_at_the_start_of_the_note_do_not_help_a_bot` |
| The limits | 11th question in a minute → 429 with the exact `Retry-After`; 201st in a day → 429 "try again in N h"; a new minute/day → allowed again; visitors counted separately; refused and invalid (400) requests are not counted | `test_the_11th_question_in_a_minute_is_a_429_the_ui_shows_as_busy`, `test_after_a_minute_the_visitor_can_ask_again`, `test_the_wait_time_is_exact`, `test_the_201st_question_in_a_day_is_a_429_with_hours_to_wait`, `test_visitors_have_separate_counters`, `test_blocked_attempts_do_not_extend_the_wait`, `test_invalid_requests_do_not_use_up_the_limit`, `test_other_addresses_are_not_limited` |
| What the user sees | the 429 text contains "429", which the UI's own "busy" pattern (read from `app.js` by the test) shows as "busy, try again" + retry button | `test_the_11th_question_in_a_minute_is_a_429_the_ui_shows_as_busy` |
| Settings | numbers come from the settings, `0` = off, defaults 10 / 200 documented in `.env.example` | `test_the_limits_come_from_the_settings_and_0_switches_them_off`, `test_the_settings_are_documented_with_safe_defaults` |
| Log | one line per blocked visitor per minute (not per refused request): `{"event": "rate_limited", "visitor": "203.0.113.x", "rule": ..., "retry_after_s": ..., "proxy_note_entries": N}`; the address is always partly hidden | `test_a_block_is_logged_once_a_minute_with_a_hidden_address`, `test_the_log_never_shows_a_full_address` |
| Memory + threads | at most 50,000 visitors remembered (the least recently seen are forgotten first); requests at the same moment can't slip past the limit (lock) | `test_memory_stays_bounded_with_many_visitors`, `test_parallel_requests_cannot_slip_past_the_limit` |
| Report form | its 5-per-hour limit now counts the real visitor too. Before, it used `request.client.host`, which behind Render's proxy is probably the proxy, so all visitors may have shared one counter | `test_report_limit_counts_each_visitor_separately` |

**⚠️ To check on Render (not done yet: the owner decided to implement first, check later).**
The rule "last address in the note = the visitor" was tested locally only. If Render has ONE MORE middleman
in front of its proxy, the last address would be that middleman, and many real users would share one counter.
How to check, about 5 minutes, no code needed:
1. Deploy, open the live site on a laptop, and ask 11 questions quickly (or temporarily set
   `ASK_LIMIT_PER_MINUTE=3` on Render and ask 4). The last one must show "busy, try again".
2. Right away, ask a question from a phone **with Wi-Fi off** (mobile data = a different address).
   It must work normally. **If the phone is ALSO blocked**, the address is wrong: set both limits to `0` on
   Render (limit off) and fix `limits.client_ip`.
3. In the Render logs, find the `rate_limited` line: `visitor` should look like the laptop's address from
   whatismyip.com (last part hidden), and `proxy_note_entries` tells how many addresses the note had.
4. Later, now and then: if the `rate_limited` lines always show the same few `visitor` values from different
   people, that's the middleman and the limit needs fixing.

**Thoughts / decisions.**
- Why 10 / 200: a person reads each answer (1-3 questions a minute), so 10 leaves room. 200 a day is far above
  normal use, and still safe for a school or office sharing one address. A script is stopped within a minute.
- Counters are in memory: they reset when Render restarts the server (the free plan sleeps when idle). A
  determined attacker gets a fresh day after a restart. Good enough for a free project; a database
  (e.g. Redis) would be the next step if that ever matters.
- Questions that fail later (502/503) still count: the limit is checked before the search. With 10 a minute
  that doesn't hurt real users.
- The rule relies on the app being reachable ONLY through Render's proxy (true on Render). Run elsewhere
  without a proxy, a visitor could write the note himself: then set the limits to 0 or adapt `client_ip`.

**How it was verified.** Red first (the test file couldn't even load before `limits.py` existed). 14 deliberate
mutations: 12 caught at once. 2 survived and the tests were made stronger: "refused attempts counted" (the test
retried 30 times at the same instant: now it retries every 3 s over the whole minute) and "no lock" (threads
rarely collide: now the test makes Python switch threads as often as possible, 20 rounds; caught 5 of 5 runs,
the correct code passes 3 of 3). Checked on a real `uvicorn` server with `curl`: 4th question with
`ASK_LIMIT_PER_MINUTE=3` → 429 "try again in 60 s"; a fake address at the start of the note doesn't help;
another visitor still gets through; one `rate_limited` log line with `203.0.113.x`.

**Changes to existing tests.** `tests/conftest.py`: one new shared setup step, `fresh_rate_limits`, that empties
the question counters before every test. All tests come from the same test address, so without it the 11th
question of the whole run got "429" (37 tests in `test_api_input.py` failed). No test was changed.

## Report: report form spam checks (group 7e, `test_api_report.py`)

**Goal.** Protect the "Report a problem" form (it e-mails the owner) from spam WITHOUT accounts, keys or outside
services (a captcha like Cloudflare Turnstile was considered and postponed: only if spam still gets through).

```bash
python -m pytest tests/test_api_report.py        # only this batch: 49 cases, ~4 s
python -m pytest tests/test_api_report.py -v     # with every test name
```

**One rule for what happens:**
- clear **bot** signs → a fake `{"ok": true}`: no e-mail, the bot learns nothing, one log line
  `{"event": "report_dropped", "reason": "honeypot" | "too_fast" | "duplicate", "visitor": "203.0.113.x"}`;
- things a **real person** might do → a clear **400** they can fix;
- **too many** → **429** with `Retry-After`.

Order of the checks in `/api/feedback`: mail set up? (503) → input checks (400) → bot signs (fake ok) →
limits (429) → send (400 bad screenshot / 502 mail service) → remember the sent e-mail.

| # | Check | Result | Tests |
|---|---|---|---|
| 1 | **Time trap:** `app.js` starts a timer when the form opens and sends `form_ms`; under 3 s or missing = bot | fake ok | `test_a_report_written_in_3_s_or_more_is_mailed`, `test_a_report_sent_too_fast_or_without_the_timer_gets_a_fake_ok`, `test_a_broken_timer_value_is_400`, `test_the_website_sends_how_long_the_form_was_open` |
| – | Hidden field `website` filled (honeypot, existed before) | fake ok | `test_the_hidden_field_still_gives_a_fake_ok_and_is_logged` |
| 2 | **Daily total:** at most `FEEDBACK_LIMIT_PER_DAY` (90) e-mails a day for all visitors together; only e-mails really sent count (not bots, not mail failures) | 429 "try again tomorrow" | `test_the_daily_total_stops_at_the_cap_for_everyone`, `test_only_e_mails_really_sent_count_towards_the_daily_cap` |
| – | **Per visitor:** at most `FEEDBACK_LIMIT_PER_HOUR` (5) an hour (existed; now a setting) | 429 "…in N min" | `test_the_6th_report_in_an_hour_from_one_visitor_is_429` |
| – | Settings: `0` = that limit is off; defaults 5 / 90 documented in `.env.example` | | `test_the_limits_come_from_the_settings_and_0_switches_them_off`, `test_the_settings_are_documented_with_the_agreed_defaults` |
| 3 | **Message:** cleaned (invisible / control characters, spaces; line breaks KEPT, at most one empty line), then 5–3000 characters, at least one letter, at most 2 links (`http://`, `https://`, `www.`) | 400 `EMPTY` / `TOO_SHORT` / `NO_LETTERS` / `TOO_MANY_LINKS` | `test_an_empty_message_is_400`, `test_a_message_with_no_letters_is_400`, `test_a_message_that_is_too_short_after_cleaning_is_400`, `test_more_than_2_links_is_400_with_a_clear_message`, `test_the_message_is_cleaned_but_keeps_its_line_breaks` |
| 3b | **Same text:** the SAME visitor + the same text (ignoring spaces / capitals) within 24 h = double click or bot. Another person with the same words is always sent. A report that failed to send is not remembered. Only a hash is kept, never the text | fake ok | `test_the_same_visitor_sending_the_same_text_again_gets_a_fake_ok`, `test_the_same_text_with_other_spaces_or_capitals_is_still_the_same`, `test_two_people_writing_the_same_words_both_get_through`, `test_the_same_text_is_accepted_again_after_24_hours`, `test_a_report_that_failed_to_send_is_not_remembered` |
| 4 | **Contact:** one line: line breaks and hidden characters removed (can't pretend to be extra e-mail lines); `reply_to` only for a valid address | cleaned | `test_line_breaks_and_hidden_characters_are_removed_from_the_contact` |
| 5 | **Screenshot:** the file's first bytes must match its type (PNG / JPEG / GIF / WEBP "magic numbers"), not just its name | 400 `INVALID_IMAGE` | `test_a_real_screenshot_is_attached`, `test_a_file_that_only_claims_to_be_an_image_is_400` |
| 6 | **Server memory** (RAM, not the mailbox): the per-visitor counter forgets visitors after their hour and never keeps more than 50,000; the same-text memory forgets after 24 h and keeps at most 20,000 | – | `test_the_report_counter_forgets_old_visitors`, `test_the_same_text_memory_is_bounded_and_forgets_after_a_day` |

**Thoughts / decisions.**
- Daily cap 90: Resend's free plan (as far as we know; check resend.com/pricing) allows 100 e-mails a day and
  3,000 a month; 90 × 31 = 2,790 stays under both. The counter is in memory and resets when Render restarts the
  server, so on a rare day the real total can be a bit higher. Fine for this project.
- "Same text" was changed on the owner's advice: blocking the same words for EVERYONE would lose a real second
  report ("wrong answer, please check"), so it is per visitor only.
- The time trap treats a report WITHOUT `form_ms` as a bot (otherwise bots posting straight to the API skip it).
  Only risk: a page opened before a deploy (old `app.js`) gets a fake "ok" once. Very rare.
- Bots that read the page's JavaScript could send a fake `form_ms`; that's why there are several layers (limits,
  daily cap, same text). If spam still gets through: Cloudflare Turnstile (free, needs an account + 2 keys).

**How it was verified.** Red first (the file couldn't load before `spam.py` existed). 21 deliberate mutations,
all caught: missing timer allowed, time trap 0.3 s, no same-text check, same text blocking OTHER people,
spaces/capitals dodging it, no daily cap, sent e-mails not counted, counted BEFORE sending, daily total never
resetting, same-text memory unbounded, old visitors never forgotten, no real image check, any RIFF file
accepted as WEBP, no link check, only 1 link allowed, line breaks lost, contact not cleaned, message without
letters accepted, `app.js` timer not started, drops not logged, the 429 text not saying "an hour". Checked on a
real `uvicorn` server with `curl` (too fast / no timer → fake ok + log line, `?!?!` → 400, 3 links → 400,
fake PNG → 400, a real report reaches the mail step).

**Changes to existing tests (no test added or removed, counts unchanged):**
- The old report counter `feedback._recent` is gone → `feedback._recent.clear()` became `spam.reset()`:
  `test_rag.py` (1 + import), `test_api_errors.py` (3 + import), `test_api_fields.py` (2 + import),
  `test_api_limits.py` (2 + import). `tests/conftest.py`: the shared `fresh_rate_limits` step also calls `spam.reset()`.
- Reports that must be mailed now carry the page's timer `"form_ms": 20_000` (otherwise they look like bots):
  `test_rag.py::test_feedback_is_mailed_without_exposing_the_address` (2 requests),
  `test_api_errors.py` (4 tests), `test_api_fields.py::test_exactly_what_the_frontend_sends_still_works`.
- `test_api_errors.py::test_too_many_reports_is_429_with_a_wait_time` and
  `test_api_limits.py::test_report_limit_counts_each_visitor_separately`: the 5 reports now have different texts
  (the same text twice is now a duplicate).
- `test_api_errors.py::test_a_normal_4_mb_screenshot_is_not_blocked_by_the_size_limit`: the 4 MB file now starts
  like a real PNG (it was all zeros; the new check refuses that, correctly).
- `test_api_fields.py`: `FEEDBACK_FIELDS` + `form_ms`, on purpose (that test exists to make new fields a conscious change).

## Remaining groups (detail)

### 5. `test_ask_edges.py` – branches of `rag.ask()`
- `US_FULL=True` path (production US path; the market file is not in the repo, so it is never exercised)
- triage LLM raises → mode `"none"` with the refusal text
- `GENERAL:` reply longer than 1500 chars → `"none"`; `JOKE:` longer than 400 → `"none"`
- two products in one question → both groups in the Qdrant filter
- `equivalents` is empty when the asked product's own leaflet is the one cited
- `translate_ms` / `embed_ms` appear in `timings`; `search_query` set for a non-English US question
- `rag.ask(top_k=...)` passes `top_k * 5` to the store

### 6. LLM reliability – done (`test_llm.py`, `test_llm_answers.py`, `test_emergency.py`)
No network: Groq is a fake `httpx.post/get`, Gemini a fake `_gemini` / `google.genai.Client`, time a fake
clock (a "30 s timeout" costs nothing and waits are measured exactly).
- `test_llm.py` (provider layer): retry + backoff on 429/5xx/overloaded, no retry on 401/403/400, a timeout
  moves on at once; Groq request shape; retired Groq model replaced and remembered, but NOT on a
  context-length 400; empty / `null` / cut-off Groq answer and empty / blocked Gemini answer → next
  provider; Gemini backup + discovered models (listed once, with a timeout); groq → gemini order, circuit
  breaker and its expiry; one Groq try ≤ 15 s, all tries ≤ 30 s; all down → `LLMUnavailable` naming every
  provider; probe after a total outage; a bad request doesn't bench Groq for everyone.
- `test_llm_answers.py` (what the user gets): outage = error, never a fake "not found"; 503 + Retry-After +
  the text the UI shows as "busy, try again"; triage outage → polite refusal; model label per question;
  no citation / only fake `[7]` → orange after one repair attempt; partly cited stays violet; NOT_FOUND
  variants; `<think>` removed; failed or citation-losing translation → cited English answer.
- `test_emergency.py` (part 6): the three levels with the LLM working (RISK line), with the LLM down
  (keywords, 11 emergency + 12 ordinary questions in 6 languages, leaflet found or not), the US-market
  translation path, fixed texts in every language, HTTP 200 red card vs 503.

Walkthrough of the outage, measured with the fake clock: both providers timing out → "busy" after 30 s
(was 120 s); during an outage each new question tries one untested backup model (≤ 30 s), then everyone
gets "busy" at once except one probe every 10 s; providers back → answers again within ~10 s (was 5 min).

### 6c. citations are true – done (`test_citation_truth.py`)
Before: a `[n]` was only checked to EXIST and point to a real source, so "take 3 tablets [1]" was violet
even if [1] says 2. Now (no extra LLM call) every NUMBER in a cited sentence must be in one of the sources
that sentence cites; the user's own numbers from the question may be repeated. On a mismatch: one repair
attempt that names each bad number and where it really is ("12" (you cited [1]; it is in [2])); still wrong
→ not violet (not-found path → orange). Decision "A" (strict), agreed.
- No false alarms for: 2,5 = 2.5; 1 000 = 1000 = 1,000; "мг" vs "mg"; number words in the leaflet
  ("dwie", "half" → 2, 0.5); ranges 6-8 / 6–8 / "6 to 8"; list numbering "1)" / "1."; names (B12, CYP2C9);
  uncited sentences (they are counted by `uncited_sentences` instead).
- Checked by hand on the real fixture leaflets (pl/ua): correct answers pass, also in another language;
  an invented "1200 mg", a wrong maximum "5 tablets", "2400 mg" are caught.
- Units (6c-2, `test_citation_units.py`): mcg/µg/мкг, mg/мг, g/г, kg/кг, ml/мл, l/л; "2 glasses" is not
  "2 g"; mass never matches volume; "100 mg/5 ml" works; float noise (2.01 g = 2010 mg) handled. A number
  without a unit in the answer is compared as a plain number (no conversion: stricter).

### 6d. meaning check – done (`test_meaning_check.py`)
A second LLM call (the judge, same provider chain and 30 s budget) gets every cited sentence with ONLY the
sources it cites and answers SUPPORTED / CONTRADICTED / NOT_IN_SOURCE per sentence. Order in `ask()`:
citations exist → numbers match (6c) → meaning matches (6d). Decisions, as settings:
- 1b `JUDGE_MODE=risky` (default): only if the question OR the answer touches pregnancy, breastfeeding,
  children, alcohol, driving, other medicines, allergy, kidney/liver (`risky_topic`, 6 languages);
  `always` / `off` possible. On `eval/questions.jsonl` 4 of 16 questions trigger it (answers add more).
- 2a `JUDGE_STRICT=true` (default): NOT_IN_SOURCE counts like CONTRADICTED; `false` = only counted in
  `unsupported_sentences`. The prompt says translation / paraphrase / summary = SUPPORTED.
- 3a judge down, too slow or unreadable → the answer stays violet, `meaning_check: "unavailable"`.
- A problem → one repair naming the sentence → numbers checked again → judged again → still wrong (or the
  repair call fails: the answer is KNOWN to be wrong, so no fail-open) → orange. `RISK: EMERGENCY` in the
  repair → red card. Response + log: `meaning_check` = passed / repaired / not_checked / unavailable.
- The judge is scripted in the tests: its real accuracy needs real-API testing (e.g. ~30 correct + ~30
  deliberately wrong answers: how many does it catch, how many correct ones does it reject?).

### 8-real. Real LLM check (part 8, by hand, needs API keys) – todo
Goes into `tests/integration/` (not run by `python -m pytest`). One real call per provider; the answer has a
valid `[n]`; the RISK level of the 4 example questions ("5 x 500 mg" → EMERGENCY, "forgot lunch pill" →
CAUTION, "doctor said crush it" → NONE, "what is the dose" → NONE). Until then: ask these 4 on the live app.

### 7. `test_api.py` – `app/main.py` (FastAPI `TestClient`)
- ~~`/api/ask`: 422 on question < 3 chars, `top_k` 99, unknown market / language; 502 when store or LLM raises~~
  **done in 7a** (now 400 with a readable message; 502 without internal details)
- ~~`/api/feedback`: 6th request from one IP → 429; image > 4 MB → 400; not configured → 502~~ **done in 7a**
  (not configured is now 503)
- ~~7b: bad language/country codes with a hint (`ua` → `uk`), questions with no letters → 400~~ **done in 7b**
- ~~7c: unknown fields → 400 (`extra="forbid"`), e.g. a typo `answerLanguage`~~ **done in 7c**
- ~~point 4: limit on questions per visitor (429)~~ **done in 7d**; ⚠️ the visitor address rule still needs the check on Render (see "Report: question limit")
- ~~point 5: report form spam checks without accounts (time trap, daily cap, links, duplicates, real image check)~~ **done in 7e**
- `/api/health`: degrades to `"error: ..."` when Qdrant is down, still 200
- `/api/drugs?market=`, `/api/fact?lang=` + `seen=`
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
| S9 | `rag.SAFETY_NOTE`, `rag.EMERGENCY_SOFT`, keyword lists `_TAKEN` / `_TOO_MUCH` / `_ALREADY_BAD` (uk/pl/cs/es/fr) | written by the assistant | safety texts + the only emergency check while the LLM is down: a native check per language | 15 min |
| S10 | `looks_like_emergency` | "by accident/by mistake" fires for 1-2 pills ("accidentally took my pill twice"); a child with 3 tablets is missed (number rule starts at 5) | only while the LLM is down; the red text is conditional | decide |
| S11 | `SYSTEM_PROMPT` RISK line | the model's judgement of the 3 levels is not tested (LLM is scripted in tests) | needs part 8: real calls with the 4 example questions | 30 min |
| S12 | ~~`rag._cite`~~ | ~~a citation is checked to exist, not to be TRUE~~ | **done in group 6c** (numbers) | – |
| S13 | `llm._anthropic` | not tested (not used in production; `LLM_PROVIDER` has no `anthropic`) | test it before switching it on | 15 min |
| S14 | `llm._discover_backup_models` | a failed lookup is cached as `[]` until restart | no Gemini backups after a network blip at the first lookup | 10 min |
| S15 | frontend | no JS tests: "busy" message + retry button, orange card for uncited answers, 🟠 safety line, red card with retry | check by eye once after deploy | 5 min |
| S17 | ~~`wrong_numbers`~~ | ~~only numbers are checked, not meaning~~ | **done in 6d** (LLM judge on risky topics) | – |
| S19 | `rag._RISKY` | topic words for 6 languages written by the assistant | decides when the judge runs: a native check per language | 15 min |
| S20 | `rag._RISKY` | "contraindications", "overdose", "boxed warning" questions are NOT risky topics yet | a wrong yes/no there matters too; adding them = more judge calls | decide |
| S21 | judge | its accuracy is unknown (scripted in tests); watch `meaning_check` and `wrong_meaning_rejected` in the logs | too strict = good answers turn orange; too lenient = misses | real-API test |
| S18 | ~~`wrong_numbers`~~ | ~~"1 g" vs "1000 mg" counted as wrong~~ | **done in 6c-2**; also catches unit mistakes the number-only check missed ("1200 g" for "1200 мг", "500 mg" for "500 µg") | – |
| S22 | `limits.limit_ask` | an EMERGENCY question from a visitor who hit the question limit gets "429 busy", not the red card | rare (10 questions in a minute first), but this is a medical app: when the LLM is down the app already shows the fixed red card (`EMERGENCY_SOFT`) for emergency words; the same could be done above the limit (no LLM call, no cost) | decide |
| S23 | Render | the "last address in the note = visitor" rule is not confirmed on the live server | if wrong, real users share one counter and can block each other | 5 min, see "Report: question limit" |
| S16 | whole app | the `RISK:` line changed `SYSTEM_PROMPT` for every answer | ask 3-4 normal questions on the live app: still violet with citations? | 5 min |

## Fixed so far (with the test that pins it)

| Fix | Test |
|---|---|
| Report form spam checks: time trap (`form_ms` from `app.js`), same visitor + same text within 24 h, 90 e-mails/day in total (only real sends count), 5/hour per visitor as a setting, message must have letters and ≤ 2 links (was: `"      "` e-mailed), contact one line, screenshot must really be an image (was: only its name checked), counters bounded and forgetting old visitors (was: growing forever) | `test_api_report.py` (all 49, see "Report: report form spam checks") |
| Limit on questions per visitor: 10/min, 200/day (settings, 0 = off) → 429 shown as "busy, try again"; visitor = last address in the proxy note (fakes at the start don't help), IPv6 per /64; refused and invalid requests not counted; memory bounded; thread-safe; the report form's limit now counts real visitors, not Render's proxy | `test_api_limits.py` (all 29, see "Report: question limit") |
| Unknown JSON fields → 400 `UNKNOWN_FIELD` naming the field (a typo like `answerLanguage` was silently ignored); the frontend's request fields are checked against the server's in a test | `test_api_fields.py` (all 17, see "Report: unknown fields") |
| Input checks: question cleaned (invisible / control / broken characters, spaces) before it reaches the pipeline; empty or letter-free question → 400 `EMPTY` / `NO_LETTERS` (was sent to the LLM); length counted after cleaning; codes accepted in any case, unknown ones → 400 with a hint for mix-ups (`ua`/`uk`, `cz`/`cs`) | `test_api_input.py` (all 79, see "Report: input checks") |
| API errors: exact numbers + one JSON shape (`detail` string, `error` code); 422 → 400 with a readable message naming the field; input never echoed; body size limit → 413; crash → JSON 500 with `request_id`; no internal details in 502; busy service → 503; 404/405 right for `/api/...`; CORS header on errors | `test_api_errors.py` (all 49, see "Report: API errors") |
| Meaning check: on risky topics an LLM judge checks each cited sentence against ONLY its cited sources; CONTRADICTED (and NOT_IN_SOURCE when strict) → one repair → re-checked (numbers + judge) → still wrong → orange; judge down → stays violet, logged | `test_meaning_check.py::test_contradicted_sentence_gets_one_repair_and_is_judged_again`, `::test_still_contradicted_after_the_repair_goes_orange`, `::test_judge_down_or_unreadable_keeps_the_answer_violet`, `::test_not_in_source_follows_the_strict_setting` |
| Units: a number with a unit is compared as an amount, so "1 g" = "1000 mg" = "1000000 mcg" (also мг/г/мкг/мл/л), while "500 mg" for "500 µg" and "1200 g" for "1200 мг" are now caught | `test_citation_units.py::test_the_same_amount_in_another_unit_is_not_a_wrong_number`, `::test_a_different_amount_is_still_caught` |
| Citations must be TRUE for numbers: a number of a cited sentence missing from the cited source(s) → one repair attempt naming the number (and the source that really has it) → still wrong → orange, never violet | `test_citation_truth.py::test_wrong_dose_with_a_valid_looking_citation_is_never_violet`, `::test_right_number_but_wrong_source_is_caught`, `::test_wrong_number_gets_one_repair_attempt_that_can_fix_it`, `::test_still_wrong_after_the_repair_goes_orange` |
| Emergency check on EVERY question: the answer starts with `RISK: EMERGENCY/CAUTION/NONE` (no extra LLM call); EMERGENCY → fixed red card even when the leaflet was found (before: only when nothing was found); CAUTION → violet answer + fixed `safety_note`; the RISK line is never shown | `test_emergency.py::test_clear_emergency_with_a_leaflet_found_shows_the_red_card`, `::test_possible_risk_gets_the_leaflet_answer_plus_a_safety_note`, `::test_risk_line_is_never_shown_to_the_user` |
| LLM down + keywords (already taken + too much, or "overdosed"…) → soft red card `EMERGENCY_SOFT` with HTTP 200 and a retry button; ordinary questions → 503 "busy" | `test_emergency.py::test_llm_down_and_clear_emergency_words_give_the_soft_red_card`, `::test_llm_down_and_an_ordinary_question_is_just_busy`, `::test_api_llm_down_emergency_is_a_200_red_card_and_ordinary_question_a_503` |
| LLM time limits: one Groq try ≤ `GROQ_TIMEOUT_S` (12 s), all LLM calls of one question ≤ `LLM_DEADLINE_S` (30 s); was up to 120 s+ | `test_llm.py::test_hanging_groq_costs_at_most_15_s_before_gemini_answers`, `::test_total_outage_is_bounded_by_one_deadline` |
| All providers down → `llm.LLMUnavailable` → HTTP 503 + `Retry-After`, clean "busy" text the UI recognises (raw errors only in the log) | `test_llm.py::test_total_outage_raises_one_typed_error_naming_every_provider`, `test_llm_answers.py::test_total_outage_is_a_503_with_retry_after_and_a_clean_message` |
| After a total outage one probe call every 10 s (`PROBE_INTERVAL_S`) instead of 5 minutes of instant failures | `test_llm.py::test_after_a_total_outage_the_next_request_still_probes_a_provider` |
| Empty / safety-blocked / cut-off (`finish_reason=length`, `MAX_TOKENS`) answer = `EmptyAnswer` → next provider, not "not in the leaflet" | `test_llm.py::test_empty_groq_answer_falls_through_to_gemini`, `::test_truncated_groq_answer_is_not_returned_as_complete`, `::test_blocked_or_empty_gemini_answer_moves_to_the_next_model` |
| Groq model swap only on `model_not_found`/`decommissioned`; a request-specific 400/413/422 no longer benches the provider for everyone | `test_llm.py::test_context_length_error_does_not_swap_the_production_model`, `::test_a_bad_request_does_not_bench_groq_for_everyone` |
| HTTP status read from the error (SDK attribute / leading code / stand-alone number), not as a substring (`15004` ≠ 500) | `test_llm.py::test_a_status_code_is_not_found_inside_another_number` |
| Model discovery has a 10 s timeout | `test_llm.py::test_model_discovery_is_time_limited` |
| Answer with no valid `[n]` → one repair attempt → still none → not-found path (orange general answer), never violet | `test_llm_answers.py::test_answer_without_any_citation_is_shown_orange_not_violet`, `::test_uncited_answer_gets_one_repair_attempt`, `::test_answer_citing_only_nonexistent_sources_is_shown_orange` |
| `<think>…</think>` removed from every reply | `test_llm_answers.py::test_reasoning_trace_is_never_shown_to_the_user` |
| Translation step fails or drops every `[n]` → the cited English answer is shown (`answer_language: "en"`) | `test_llm_answers.py::test_failed_translation_returns_the_cited_english_answer`, `::test_translation_that_drops_the_citations_is_not_used` |
| Model label per question (`llm.request_scope` / `model_used()`), not a global shared by all users | `test_llm_answers.py::test_model_label_is_never_taken_from_a_previous_request` |
| NOT_FOUND / EMERGENCY texts for cs, es, fr | `test_languages.py::test_refusal_and_emergency_texts_exist_in_every_language` |
| facts translated to cs, es, fr | `test_rag.py::test_facts_are_translated_into_every_ui_language` |
| children's product named by the user is not penalised in rerank | `test_rerank.py::test_named_children_product_is_not_penalised` |
| `_CHILD` recognises baby/junior/niemowl/малюк/pediátr/bébé/kojenc/… | `test_rerank.py::test_more_child_words_are_recognised` |
| Ukrainian grammatical cases (вагітність / вагітності, біль / болю, ніч / ночі): question words are looked up in the `pymorphy3` Ukrainian dictionary (+21 MB RAM, ~1 ms per question) and every form of the word counts as a whole-word hit; the 2-letter stem rule is kept as a safety net (алкоголь → алкогольних) and is the only rule for unknown words (brand names) and other languages; for known words with an uncut 4-letter stem (`біль`) the dictionary decides alone, so `біль` no longer hits `більше`. Was S2 / an `xfail`. |
| EU headings printed in CAPITALS without accents (`4. QUELS SONT LES EFFETS INDESIRABLES`, `COMO TOMAR`) were not recognised, so the side-effects text was glued to the dosage section. Headings are now matched on an accent-folded copy of the text (same length, so the original text is cut and saved untouched); `4 -`, `4 –`, `4)` and `4:` numbering accepted. **Needs a re-ingest of fr/es (and cz/pl to be safe) to reach the index.** | `test_doc_text.py::test_headings_in_capitals_with_or_without_accents_are_found`, `::test_heading_number_separator_variants_are_accepted`, `::test_accent_folding_keeps_section_text_intact` |
| Citation ranges: `[1-20]` (10+ sources) was deleted together with its marker, leaving the sentence unsourced. Now every range is expanded and clipped to the real number of sources (`[1-20]` with 6 sources → `[1]..[6]`, `[4-2]` → `[2][3][4]`, cap 30 without a count), `[source 2]`/`[src 2]` accepted, a vanished marker leaves no stray space, the prompt forbids ranges explicitly and asks for 1-3 real sources per fact, and the response/log carries `uncited_sentences` (sentences or bullets without any `[n]`). Was S3. Equivalent mutation noted: `ask()` passing no source count gives the same result thanks to the older invalid-number guard. | `test_rag.py::test_citation_ranges_are_never_silently_deleted`, `::test_answer_with_a_wide_range_keeps_valid_citations_end_to_end`, `::test_uncited_sentences_are_counted`, `::test_prompt_forbids_citation_ranges_explicitly` |
| Stop words: `during`, `podczas`, `під час`, `během`, `durante`, `pendant`, `this/ten/цей/tento/este/ce`, generic medicine nouns (`medicine`, `lek`, `препарат`, `lék`, `medicamento`, `médicament`) were keywords, so "take during meals" earned the same bonus as a pregnancy passage. `_STOP` is now built from a readable per-language list (`_STOP_BY_LANG`, ~50-70 words each, hand-curated: no medically meaningful words, form words like tablet/syrup kept). Was S1. | `test_rerank.py::test_function_words_and_generic_medicine_words_are_not_keywords`, `::test_during_meals_passage_gets_no_bonus_for_a_pregnancy_question`, `::test_medically_meaningful_words_are_never_stop_words` | `test_rerank.py::test_ukrainian_case_change_in_the_middle_of_the_word_still_matches` (+6 tests around it) |
