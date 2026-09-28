"""United States market from the COMPLETE openFDA drug-label database (~150k labels).

Same idea as scripts/fetch_markets.py: every product (brand) name becomes searchable, labels are
grouped by generic name + route (oral, topical, ...), and one label per group is indexed (the one with
the most complete sections, preferring well-known brands).

The bulk files are streamed one by one (~100 MB zipped each) and parsed with ijson, so memory stays low;
nothing big is kept on disk.

Run:   python -m scripts.fetch_us                 (~20-40 min, depends on the connection)
       python -m scripts.fetch_us --partitions 1  (quick test: only the first bulk file)
Writes data/markets/us.json (product index, committed) and data/raw/us/docs.jsonl (label sections).
"""
import argparse
import json
import re
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
import ijson

from scripts.fetch_labels import DRUGS, SECTIONS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "us"
OUT = ROOT / "data" / "markets" / "us.json"
MAX_SECTION_CHARS = 5000
# Sections worth indexing (the others duplicate these or are rarely asked about).
KEEP = [k for k in SECTIONS if k not in ("mechanism_of_action", "precautions")]
HINTS = {b.lower() for brands in DRUGS.values() for b in brands}


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.lower()).strip("-")[:70]


def is_homeopathic(rec: dict) -> bool:
    head = " ".join((rec.get("indications_and_usage") or rec.get("purpose") or [""])[:1])[:400].lower()
    return "homeopathic" in head or " hpus" in head


def partitions() -> list[str]:
    idx = httpx.get("https://api.fda.gov/download.json", timeout=60).json()
    return [p["file"] for p in idx["results"]["drug"]["label"]["partitions"]]


def main(limit: int | None) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict] = {}        # gid -> meta + best label so far
    products: dict[tuple, list] = {}     # (name, gid) -> [name, gid, url, rx]
    files = partitions()[:limit]
    t0, n_labels = time.time(), 0
    for i, url in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {url.rsplit('/', 1)[-1]}  downloading…", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "part.zip"
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r, zpath.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as jf:
                for rec in ijson.items(jf, "results.item"):
                    n_labels += 1
                    ofda = rec.get("openfda") or {}
                    if not ofda.get("generic_name") or is_homeopathic(rec):
                        continue
                    generic = re.sub(r"\s+", " ", ofda["generic_name"][0]).strip().lower()
                    route = (ofda.get("route") or ["other"])[0].lower()
                    gid = f"us:{slug(generic)}:{slug(route)}"
                    set_id = rec.get("set_id", "")
                    link = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
                    rx = "PRESCRIPTION" in " ".join(ofda.get("product_type", [])).upper()
                    brands = [re.sub(r"\s+", " ", b).strip() for b in ofda.get("brand_name", [])][:3]
                    for b in brands or [generic.title()]:
                        products.setdefault((b.lower(), gid), [b, gid, link, rx])
                    sections = [{"section": title, "text": " ".join(rec.get(k, [])).strip()[:MAX_SECTION_CHARS]}
                                for k, title in SECTIONS.items() if k in KEEP and rec.get(k)]
                    if not sections:
                        continue
                    brand = brands[0] if brands else generic.title()
                    score = (brand.lower() in HINTS, len(sections), rec.get("effective_time", ""))
                    g = groups.setdefault(gid, {"inn": generic.title(), "en": generic.title(), "form": route,
                                                "score": None})
                    if g["score"] is None or score > g["score"]:
                        g.update(score=score, name=brand, url=link, effective_time=rec.get("effective_time", ""),
                                 sections=sections)
        el = (time.time() - t0) / 60
        print(f"      {n_labels} labels read, {len(groups)} groups, {len(products)} product names  "
              f"(~{el / i * (len(files) - i):.0f} min left)", flush=True)

    with (RAW / "docs.jsonl").open("w", encoding="utf-8") as f:
        for gid, g in groups.items():
            if g.get("sections"):
                f.write(json.dumps({"group": gid, "name": g["name"], "url": g["url"], "source": "FDA",
                                    "inn": g["inn"], "en": g["en"], "form": g["form"],
                                    "effective_time": g["effective_time"], "sections": g["sections"]},
                                   ensure_ascii=False) + "\n")
    index = {"market": "us",
             "groups": {gid: {"inn": g["inn"], "en": g["en"], "atc": "", "form": g["form"],
                              "cands": [{"name": g["name"], "url": g["url"], "source": "FDA"}] if g.get("sections") else []}
                        for gid, g in groups.items()},
             "products": list(products.values())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"\nDone: {n_labels} labels -> {len(groups)} groups, {len(products)} product names "
          f"-> data/markets/{OUT.name} + data/raw/us/docs.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=None, help="only the first N bulk files (testing)")
    main(ap.parse_args().partitions)
