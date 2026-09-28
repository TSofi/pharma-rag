"""EU-wide (centrally authorised) medicines from EMA, added to every EU market in its own language.

Medicines like Eliquis, Ozempic or Jardiance are authorised once for the whole EU by the European
Medicines Agency, so national registers often only link to EMA. EMA publishes the product information
(SmPC + patient leaflet) in every EU language; we keep the patient leaflet part.

The list of medicines (with the exact page slug of each one) comes from EMA's JSON data file.

Run:   python -m scripts.fetch_ema                    (pl, cz, es, fr; resumable)
       python -m scripts.fetch_ema --markets cz --limit 20   (quick test)
Then:  python -m scripts.ingest_markets --markets pl,cz,es,fr --docs docs_ema.jsonl --append
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from scripts.doc_text import any_text, leaflet_part
from scripts.fetch_markets import HEADERS, OUT, RAW

LANG = {"pl": "pl", "cz": "cs", "es": "es", "fr": "fr"}
JSON_URLS = ["https://www.ema.europa.eu/en/documents/report/medicines-output-medicines_json-report_en.json",
             "https://www.ema.europa.eu/en/media/67423"]


def medicines(client: httpx.Client) -> list[dict]:
    cache = RAW / "eu" / "ema_medicines.json"
    if not cache.exists():
        for url in JSON_URLS:
            r = client.get(url)
            if r.status_code == 200 and r.content[:1] in (b"[", b"{"):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(r.content)
                break
        else:
            raise SystemExit("Could not download EMA's medicines JSON")
    data = json.loads(cache.read_text("utf-8"))
    rows = data.get("data", data) if isinstance(data, dict) else data
    out = []
    for m in rows:
        if str(m.get("category", "")).lower() != "human" or "authorised" != str(m.get("medicine_status", "")).lower():
            continue
        url = m.get("medicine_url") or ""
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        if slug:
            out.append({"name": m["name_of_medicine"].strip(), "slug": slug,
                        "inn": (m.get("international_non_proprietary_name_common_name") or m.get("active_substance") or "").strip(),
                        "atc": m.get("atc_code_human") or ""})
    return out


def run(market: str, meds: list[dict], client: httpx.Client, limit: int | None, workers: int) -> None:
    lang = LANG[market]
    idx_path = OUT / f"{market}.json"
    idx = json.loads(idx_path.read_text("utf-8"))
    docs_file = RAW / market / "docs_ema.jsonl"
    done = {json.loads(l)["group"] for l in docs_file.read_text("utf-8").splitlines()} if docs_file.exists() else set()
    # Poland already has some EMA leaflets from the first run (docs.jsonl): don't index those twice.
    main_docs = RAW / market / "docs.jsonl"
    have = {json.loads(l)["name"].lower() for l in main_docs.read_text("utf-8").splitlines()
            if '"source": "EMA"' in l} if main_docs.exists() else set()
    known = {p[0].lower() for p in idx["products"]}
    todo = []
    for m in meds:
        gid = f"{market}:ema-{m['slug']}"
        # link shown to users = the PDF in the market's language
        page = f"https://www.ema.europa.eu/{lang}/documents/product-information/{m['slug']}-epar-product-information_{lang}.pdf"
        idx["groups"].setdefault(gid, {"inn": m["inn"], "en": m["inn"], "atc": m["atc"], "form": "eu",
                                       "cands": [{"name": m["name"], "url": page, "source": "EMA"}]})
        if m["name"].lower() not in known:
            idx["products"].append([m["name"], gid, page, True])
        if gid not in done and m["name"].lower() not in have:
            todo.append((gid, m, page))
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), "utf-8")
    todo = todo[:limit]
    print(f"  {len(meds)} EU medicines, {len(done)} already downloaded, {len(todo)} to go")
    lock, stats = threading.Lock(), {"ok": 0, "failed": 0}

    def fetch(gid, m, page):
        pdf = page
        try:
            r = client.get(pdf)
            text = leaflet_part(any_text(r.content, pdf), market) if r.status_code == 200 else ""
        except Exception:  # noqa: BLE001
            text = ""
        with lock:
            if len(text) > 500:
                with docs_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"group": gid, "name": m["name"], "url": page, "source": "EMA", "inn": m["inn"],
                                        "en": m["inn"], "form": "eu", "text": text}, ensure_ascii=False) + "\n")
                stats["ok"] += 1
            else:
                stats["failed"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, *t) for t in todo]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 10 == 0 or i == len(futures):
                eta = (time.time() - t0) / i * (len(futures) - i) / 60
                print(f"\r  {market}: {i}/{len(futures)}  ok={stats['ok']} failed={stats['failed']}  ~{eta:.0f} min left ",
                      end="", flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pl,cz,es,fr")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    client = httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True)
    meds = medicines(client)
    for mk in args.markets.split(","):
        print(f"\n== {mk}")
        run(mk, meds, client, args.limit, args.workers)
