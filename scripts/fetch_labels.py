"""Step 1 of the data pipeline: download official drug labels from the openFDA API.

openFDA (https://open.fda.gov) exposes FDA Structured Product Labels as JSON, already split
into sections (Indications, Dosage, Contraindications, ...). Public domain, free, no key needed.

Output: data/labels.jsonl  (one JSON record per drug)
Run:    python scripts/fetch_labels.py
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "labels.jsonl"
# data/drugs.txt lines look like "sertraline: Zoloft". openFDA's own brand_name field is
# unreliable (often just the salt, e.g. "Sertraline Hydrochloride"), so we curate brands ourselves.
DRUGS: dict[str, list[str]] = {}
for _line in (ROOT / "data" / "drugs.txt").read_text(encoding="utf-8").splitlines():
    if _line.strip() and not _line.startswith("#"):
        _name, _, _brands = _line.partition(":")
        DRUGS[_name.strip().lower()] = [b.strip() for b in _brands.split(",") if b.strip()]

# openFDA field -> human-readable section title (Rx labels + OTC "Drug Facts" labels)
SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "purpose": "Purpose",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "do_not_use": "Do Not Use",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "ask_doctor": "Ask a Doctor Before Use",
    "ask_doctor_or_pharmacist": "Ask a Doctor or Pharmacist Before Use",
    "stop_use": "Stop Use and Ask a Doctor If",
    "drug_interactions": "Drug Interactions",
    "adverse_reactions": "Adverse Reactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "overdosage": "Overdosage",
    "mechanism_of_action": "Mechanism of Action",
}
MAX_SECTION_CHARS = 8000  # very long sections (big adverse-reaction tables) are truncated


def query(drug: str) -> list[dict]:
    search = f'openfda.generic_name:"{drug}"'
    url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode({"search": search, "limit": 20})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r).get("results", [])
    except urllib.error.HTTPError as e:
        if e.code == 404:  # openFDA returns 404 when nothing matches
            return []
        raise


def pick_best(drug: str, results: list[dict]) -> dict | None:
    """Several manufacturers publish labels for the same drug. Prefer single-ingredient
    products (generic name == the drug, no combos) with the most sections filled in."""

    def score(rec):
        generics = [g.lower() for g in rec.get("openfda", {}).get("generic_name", [])]
        single = any(g.split()[0] == drug.split()[0] and " and " not in g and "," not in g for g in generics)
        n_sections = sum(1 for k in SECTIONS if rec.get(k))
        return (single, n_sections, rec.get("effective_time", ""))

    results = [r for r in results if r.get("openfda", {}).get("generic_name")]
    return max(results, key=score) if results else None


def to_record(drug: str, rec: dict, brands: list[str]) -> dict:
    ofda = rec.get("openfda", {})
    set_id = rec.get("set_id", "")
    sections = []
    for key, title in SECTIONS.items():
        text = " ".join(rec.get(key, [])).strip()
        if text:
            sections.append({"section": title, "text": text[:MAX_SECTION_CHARS]})
    return {
        "drug": drug,
        "brands": brands,
        "generic_name": (ofda.get("generic_name") or [drug])[0].title(),
        "manufacturer": (ofda.get("manufacturer_name") or [""])[0],
        "set_id": set_id,
        "effective_time": rec.get("effective_time", ""),  # label version date, YYYYMMDD
        "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
        "sections": sections,
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    ok = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for drug, brands in DRUGS.items():
            best = pick_best(drug, query(drug))
            if not best:
                print(f"  [--] {drug}: not found")
                continue
            rec = to_record(drug, best, brands)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            print(f"  [ok] {drug:<22} {len(rec['sections']):>2} sections  brands={rec['brands'][:2]}")
            time.sleep(0.3)  # be polite: openFDA allows ~240 requests/min without a key
    print(f"\nSaved {ok}/{len(DRUGS)} drug labels -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
