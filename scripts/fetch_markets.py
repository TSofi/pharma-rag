"""Poland / Ukraine markets from the COMPLETE national registers.

Every product in the register becomes searchable by name. Products are grouped by active substance +
form (tablets, syrup, gel, ...) because generics of one substance share almost word-for-word the same
leaflet. For each group ONE leaflet is downloaded and indexed (preferring the original brand); every
other product of the group points to it, and the answer says so ("Ibuprom -> leaflet of Nurofen,
same substance and form"), with a link to the asked product's own leaflet too.

  PL  Rejestr Produktów Leczniczych (URPL, e-Zdrowie) + EMA product information for EU-wide authorisations
  UA  Державний реєстр лікарських засобів (MOZ, CC BY 4.0)

Needs data/raw/pl_rpl.csv and data/raw/ua_reestr.csv (python -m scripts.probe_markets).
Run:   python -m scripts.fetch_markets              (resumable: re-run to continue after a break)
       python -m scripts.fetch_markets --markets pl --limit 50   (quick test)
Writes data/markets/<m>.json (product index, committed) and data/raw/<m>/docs.jsonl (leaflet texts).
"""
import argparse
import csv
import io
import json
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from scripts.doc_text import any_text, leaflet_part

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "markets"
HEADERS = {"User-Agent": "PharmaRAG portfolio project (educational)"}

# Brand names to prefer as the group's representative leaflet (usually the original product).
HINTS = set("""glucophage zestril lipitor sortis norvasc losec controloc euthyrox letrox zocor cozaar betaloc
dilatrend aldactone verospiron zoloft lexapro cipralex prozac wellbutrin xanax stilnox neurontin tramal
nurofen aleve apap panadol amoxil sumamed cipronex encorton ventolin singulair zyrtec claritine warfin
eliquis plavix januvia jardiance milurit metex viagra omnic proscar voltaren diclac laremid imodium
polopiryna aspirin furagin no-spa pyralgina otrivin flavamed acc nimesil ketonal concor tritace crestor
aerius xyzal smecta espumisan kreon theraflu gripex fervex
глюкофаж норваск омез лосек еутирокс l-тироксин зокор лозап беталок верошпірон золофт ципралекс ксанакс
нейронтин трамал нурофен аспірин вольтарен диклак імодіум фурагін но-шпа анальгін отривін ацц амброксол
німесил кетонал конкор тритаце крестор еріус смекта еспумізан креон терафлю фервекс варфарин плавікс
еліквіс янувія джардинс вентолін сингуляр зиртек кларитин""".split())

PLACEHOLDER_INN = re.compile(r"^(comb drug|mono|produkt złożony|homeopatyczny.*|без мнн|.*combinations.*|)$", re.I)

BUCKETS = [  # first match wins; PL + UA keywords
    ("eye/ear", r"oczu|uszu|oczn|очн|вушн"),
    ("nasal", r"do nosa|nosow|назальн|нос"),
    ("inhalation", r"inhal|інгал"),
    ("injection", r"wstrzyk|infuz|ін.єкц|інфуз|ліофіл|концентрат"),
    ("rectal/vaginal", r"czop|doodbyt|dopochw|globul|супозит|ректал|вагінал|песар"),
    ("skin", r"krem|maść|żel|skór|plast|emulsj|pianka|крем|мазь|гель|нашкір|пластир|емульс|шкір"),
    ("oral liquid", r"syrop|doustn|płyn|zawiesin|krople|сироп|оральн|сусп|краплі|розчин|еліксир"),
    ("oral solid", r"tablet|kaps|draż|pastyl|granul|proszek|таблет|капсул|драже|льодян|гранул|порошок|пастил"),
    ("herbal", r"zioła|трава|листя|квітки|кореневищ|плоди|збір|настойк"),
]


def bucket(form: str) -> str:
    f = form.lower()
    return next((name for name, pat in BUCKETS if re.search(pat, f)), "other")


def slug(s: str) -> str:
    return re.sub(r"[^\w]+", "-", s.lower()).strip("-")[:60]


def read_csv(path: Path, encoding: str, delimiter: str) -> list[dict]:
    text = path.read_bytes().decode(encoding, errors="replace")
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def num(s: str) -> int:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 10**9


def hint_rank(name: str) -> int:
    first = re.split(r"[\s®,]+", name.lower().strip())[0]
    return 0 if first in HINTS else 1


def english_names() -> dict[str, str]:
    """ATC code -> English substance name, from the Ukrainian register (its INN column is in English)."""
    out = {}
    for r in read_csv(RAW / "ua_reestr.csv", "cp1251", ";"):
        inn, atc = r["Міжнародне непатентоване найменування"].strip(), r["Код АТС 1"].strip()
        if atc and inn and not PLACEHOLDER_INN.match(inn):
            out.setdefault(atc, inn)
    return out


# ------------------------------------------------------------------ Poland
def build_pl(en: dict[str, str]) -> tuple[dict, list]:
    groups, products = {}, []
    for r in read_csv(RAW / "pl_rpl.csv", "utf-8-sig", "|"):
        if r["Rodzaj preparatu"] != "Ludzki":
            continue
        name, inn, atc = r["Nazwa Produktu Leczniczego"].strip(), r["Nazwa powszechnie stosowana"].strip(), r["Kod ATC"].strip()
        form = r["Postać farmaceutyczna"]
        key_inn = name.split()[0] if PLACEHOLDER_INN.match(inn) else inn  # combinations: group per brand
        gid = f"pl:{slug(key_inn)}:{bucket(form)}"
        own_url = next((r[c] for c in ("Ulotka", "Etykieto-ulotka", "Ulotka importu równoległego")
                        if (r[c] or "").startswith("http")), "")
        rx = "Rp" in r["Opakowanie"] and "OTC" not in r["Opakowanie"]
        g = groups.setdefault(gid, {"inn": inn, "en": en.get(atc, ""), "atc": atc, "form": bucket(form), "cands": []})
        if not g["en"] and atc in en:
            g["en"] = en[atc]
        leaflet = r["Ulotka"] if (r["Ulotka"] or "").startswith("http") else r["Etykieto-ulotka"]
        if r["Typ procedury"] == "CEN":  # EU-wide authorisation: leaflet published by EMA, in Polish
            leaflet = f"https://www.ema.europa.eu/pl/documents/product-information/{slug(name.replace('®', ''))}-epar-product-information_pl.pdf"
            own_url = own_url or leaflet
            g["cands"].append(((hint_rank(name), 1, len(name)), name, leaflet, "EMA"))
        elif r["Typ procedury"] != "IR" and (leaflet or "").startswith("http"):
            g["cands"].append(((hint_rank(name), 0, num(r["Numer pozwolenia"])), name, leaflet, "URPL"))
        products.append([name, gid, own_url, rx])
    return groups, products


# ------------------------------------------------------------------ Ukraine
def build_ua() -> tuple[dict, list]:
    groups, products = {}, []
    for r in read_csv(RAW / "ua_reestr.csv", "cp1251", ";"):
        form, name = r["Форма випуску"], r["Торгівельне найменування"].strip()
        if "субстанц" in (form + r["Фармакотерапевтична група"]).lower() or r["Дострокове припинення"] == "Так":
            continue
        inn, atc = r["Міжнародне непатентоване найменування"].strip(), r["Код АТС 1"].strip()
        key_inn = re.split(r"[\s®]+", name)[0] if PLACEHOLDER_INN.match(inn) else inn
        gid = f"ua:{slug(key_inn)}:{bucket(form)}"
        url = r["URL інструкції"] if (r["URL інструкції"] or "").startswith("http") else ""
        rx = "без рецепта" not in r["Умови відпуску"].lower()
        products.append([name, gid, url, rx])
        g = groups.setdefault(gid, {"inn": inn, "en": "" if PLACEHOLDER_INN.match(inn) else inn, "atc": atc,
                                    "form": bucket(form), "cands": []})
        reg = r["Номер Реєстраційного посвідчення"]
        if url and "in bulk" not in form.lower():
            g["cands"].append(((hint_rank(name), 0, num(reg.split("/")[1] if "/" in reg else "")), name, url, "MOZ"))
    return groups, products


def finalize(market: str, groups: dict, products: list) -> dict:
    """Pick the best leaflet per group (plus 2 fallbacks in case a download fails)."""
    for g in groups.values():
        # candidates: (rank, name, fetch-url, source[, public link shown to users])
        g["cands"] = [{"name": c[1], "url": c[2], "source": c[3], **({"link": c[4]} if len(c) > 4 else {})}
                      for c in sorted(g.pop("cands"))[:3]]
    index = {"market": market, "groups": groups, "products": products}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{market}.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")), "utf-8")
    with_leaflet = sum(1 for g in groups.values() if g["cands"])
    print(f"  {len(products)} products -> {len(groups)} groups ({with_leaflet} with a leaflet to download)")
    return index


# ------------------------------------------------------------------ download + extract text
def download(market: str, index: dict, limit: int | None, workers: int, zip_path: Path | None = None) -> None:
    """Fetch + extract one leaflet per group. URLs starting with "zip:" are read from a local bulk ZIP."""
    folder = RAW / market
    folder.mkdir(parents=True, exist_ok=True)
    docs_file = folder / "docs.jsonl"
    done = set()
    if docs_file.exists():
        for line in docs_file.read_text("utf-8").splitlines():
            try:
                done.add(json.loads(line)["group"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [(gid, g) for gid, g in index["groups"].items() if g["cands"] and gid not in done][:limit]
    print(f"  {len(done)} already downloaded, {len(todo)} to go")
    lock, stats = threading.Lock(), {"ok": 0, "failed": 0}
    client = httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True)
    bulk = zipfile.ZipFile(zip_path) if zip_path else None

    def get(url: str) -> bytes:
        if url.startswith("zip:"):
            with lock:  # a ZipFile isn't safe to read from several threads at once
                return bulk.read(url[4:])
        r = client.get(url)
        return r.content if r.status_code == 200 else b""

    def fetch(gid: str, g: dict) -> None:
        for cand in g["cands"]:  # fall back to the next product if a leaflet is missing / unreadable
            try:
                text = any_text(get(cand["url"]), cand["url"])
                if cand["source"] == "EMA":
                    text = leaflet_part(text)  # the EMA PDF also holds the long SmPC; keep the leaflet
            except Exception:  # noqa: BLE001
                text = ""
            if len(text) > 500:
                rec = {"group": gid, "name": cand["name"], "url": cand.get("link", cand["url"]), "source": cand["source"],
                       "inn": g["inn"], "en": g["en"], "form": g["form"], "text": text}
                with lock:
                    with docs_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["ok"] += 1
                return
            time.sleep(0.2)
        with lock:
            stats["failed"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:  # a few parallel downloads; polite to public servers
        futures = [pool.submit(fetch, gid, g) for gid, g in todo]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 10 == 0 or i == len(futures):
                eta = (time.time() - t0) / i * (len(futures) - i) / 60
                print(f"\r  {market}: {i}/{len(futures)}  ok={stats['ok']} failed={stats['failed']}  ~{eta:.0f} min left ",
                      end="", flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pl,ua")
    ap.add_argument("--limit", type=int, default=None, help="download only N leaflets (testing)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    en = english_names()
    for m in args.markets.split(","):
        print(f"\n== {m}")
        groups, products = build_pl(en) if m == "pl" else build_ua()
        download(m, finalize(m, groups, products), args.limit, args.workers)
