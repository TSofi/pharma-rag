# Roadmap

Planned work for PharmaRAG, roughly in order. Hours are rough estimates.

## 1. More countries (~10 h)
Open data is already confirmed for these registers. Each one needs a fetcher, a leaflet parser and an ingest run.
- [ ] Austria
- [ ] Italy
- [ ] Ireland
- [ ] Sweden
- [ ] Latvia
- [ ] Estonia

## 2. Better coverage in existing countries (~6 h)
- [ ] United States: index the remaining ~75% of openFDA label groups (now only the most common ~25%).
- [ ] EMA leaflets for Czechia, Spain and France, not only Poland (EMA blocks bulk downloads, so this needs a slower, polite fetcher).
- [ ] Scanned PDF leaflets without a text layer: add OCR instead of skipping them.
- [ ] Generics: show when products with the same substance differ (excipients, strengths) instead of using one leaflet per group.

## 3. Website (~4 h)
- [ ] Privacy page: what is logged and what is not (question text is not logged by default).
- [ ] Clearer "general answer, not from the leaflet" label on mobile.
- [ ] Show the country and leaflet date next to every source.

## 4. Running it in production (~3 h)
- [ ] Turn on the per-IP question limits on Render (the code is ready, the limits are off by default).
- [ ] Small dashboard for `/api/metrics` (answer time per stage, cache hits, errors).

## 5. Tests and docs (~3 h)
- [ ] Tests for every new country parser.
- [ ] Update the retrieval check in `eval/` with questions for the new countries.
- [ ] Short "add a new country" guide in the README.
