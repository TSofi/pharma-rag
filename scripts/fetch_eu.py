"""More EU markets from open national registers (same approach as scripts/fetch_markets.py):
every product becomes searchable by name; one leaflet per group (active substance + form) is indexed.

  cz  SÚKL open data: DLP product database + the monthly ZIP with all leaflets (PIL), ~3 GB, downloaded once
  es  AEMPS CIMA REST API: every medicine with its leaflet (prospecto) as HTML
  fr  ANSM BDPM open data (Licence Ouverte): product + composition files, one notice page per product

Run:   python -m scripts.fetch_eu --markets es,fr,cz      (resumable, like fetch_markets)
       python -m scripts.fetch_eu --markets es --limit 20  (quick test)
"""
import argparse
import csv
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import httpx

from scripts.fetch_markets import HEADERS, HINTS, PLACEHOLDER_INN, RAW, download, english_names, finalize, slug

EU = RAW / "eu"
BUCKETS_EU = [  # first match wins; Czech (SÚKL gives English form names), Spanish, French keywords
    ("eye/ear", r"eye|ear |ocular|otic|colir|oftálm|ótic|collyre|ophtalm|auricul"),
    ("nasal", r"nasal|nasal"),
    ("inhalation", r"inhal"),
    ("injection", r"inject|infus|perfus|inyect|concentrate for|concentrado"),
    ("rectal/vaginal", r"suppos|supositor|rectal|vagin|pessar|ovule|óvulo"),
    ("skin", r"cream|ointment|gel|cutaneous|transdermal|patch|crema|pomada|parche|tópic|crème|pommade|cutan|transderm"),
    ("oral liquid", r"syrup|oral solution|oral suspension|oral drops|drops|jarabe|solución oral|suspensión oral|gotas|"
                    r"bebible|sirop|buvable|gouttes"),
    ("oral solid", r"tablet|capsul|granul|powder|lozenge|comprim|cápsul|polvo|sobre|pastill|gélule|poudre|sachet"),
]
HINTS_EU = HINTS | set("""ibalgin paralen nurofen brufen aspirin panadol ibuprofen voltaren
    gelocatil dalsy enantyum espidifen frenadol nolotil voltadol doliprane dafalgan efferalgan advil spasfon
    smecta gaviscon imodium""".split())


def bucket_eu(form: str) -> str:
    f = form.lower()
    return next((name for name, pat in BUCKETS_EU if re.search(pat, f)), "other")


def rank(name: str, *extra) -> tuple:
    first = re.split(r"[\s®,]+", name.lower().strip())[0]
    return (0 if first in HINTS_EU else 1, *extra, len(name))


def add(groups, products, market, key_inn, en, form, name, url, rx, cand=None):
    gid = f"{market}:{slug(key_inn)}:{form}"
    products.setdefault((name.lower(), gid), [name, gid, url, rx])
    g = groups.setdefault(gid, {"inn": key_inn, "en": en, "atc": "", "form": form, "cands": []})
    if cand:
        g["cands"].append(cand)


# ------------------------------------------------------------------ Czech Republic
def cz_files() -> tuple[bytes, Path]:
    c = httpx.Client(headers=HEADERS, timeout=600, follow_redirects=True)
    page = c.get("https://opendata.sukl.cz/?q=katalog%2Fpil-pribalove-informace-product-information-leaflet").text
    pil_url = re.search(r"https://opendata\.sukl\.cz/soubory/SOD\d+/PIL\d+\.zip", page).group(0)
    pil = EU / pil_url.rsplit("/", 1)[-1]
    if not pil.exists():  # ~3 GB, downloaded once (then re-used)
        print(f"  downloading {pil_url} (~3 GB, once)…", flush=True)
        tmp = pil.with_suffix(".part")
        with c.stream("GET", pil_url) as r, tmp.open("wb") as f:
            total, done = int(r.headers.get("content-length", 0)), 0
            for chunk in r.iter_bytes(1 << 22):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        tmp.rename(pil)
        print()
    return (EU / "cz_dlp.zip").read_bytes(), pil


def build_cz(en: dict[str, str]) -> tuple[dict, list]:
    dlp, pil_zip = cz_files()
    z = zipfile.ZipFile(io.BytesIO(dlp))

    def table(name, enc="cp1250"):
        f = next(n for n in z.namelist() if n.lower().startswith(name))
        return list(csv.DictReader(io.StringIO(z.read(f).decode(enc, errors="replace")), delimiter=";"))

    forms = {r["FORMA"]: r.get("NAZEV_EN") or r["NAZEV"] for r in table("dlp_formy")}
    # English substance names by ATC code: SÚKL's own table (7-character codes = single substances)
    en = {**{r["ATC"]: r["NAZEV_EN"].capitalize() for r in table("dlp_atc") if len(r["ATC"]) == 7 and r.get("NAZEV_EN")},
          **en}
    docs = {r["KOD_SUKL"]: r["PIL"] for r in table("dlp_nazvydokumentu", "utf-8-sig") if r.get("PIL")}
    in_zip = set(zipfile.ZipFile(pil_zip).namelist())
    by_base = {n.rsplit("/", 1)[-1]: n for n in in_zip}
    groups, products = {}, {}
    seen_pil = set()
    for r in table("dlp_lecivepripravky"):
        name = r["NAZEV"].strip()
        atc = r["ATC_WHO"].strip()
        form = bucket_eu(forms.get(r["FORMA"], r["FORMA"]) + " " + r["CESTA"])
        key = en.get(atc) or (atc if atc else name.split()[0])
        link = f"https://prehledy.sukl.cz/prehled_leciv.html#/detail-reg/{r['KOD_SUKL']}"
        pil = docs.get(r["KOD_SUKL"], "")
        cand = None
        if pil in by_base and (pil, key, form) not in seen_pil:
            seen_pil.add((pil, key, form))
            cand = (rank(name), name, f"zip:{by_base[pil]}", "SUKL", link)
        add(groups, products, "cz", key, en.get(atc, ""), form, name.title(), link, r["VYDEJ"] != "V", cand)
    return groups, list(products.values())


# ------------------------------------------------------------------ Spain
def build_es(_en) -> tuple[dict, list]:
    c = httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True)
    groups, products, page, total = {}, {}, 1, None
    while True:
        data = c.get("https://cima.aemps.es/cima/rest/medicamentos", params={"pagina": page}).json()
        total = total or data.get("totalFilas", 0)
        for m in data.get("resultados", []):
            name = m["nombre"]
            short = re.split(r"\s+\d", name, 1)[0].strip().title()
            vtm = (m.get("vtm") or {}).get("nombre") or short
            vias = " ".join(v["nombre"] for v in m.get("viasAdministracion") or [])
            form = bucket_eu(((m.get("formaFarmaceuticaSimplificada") or {}).get("nombre", "")) + " " + vias)
            doc = next((d for d in m.get("docs", []) if d.get("tipo") == 2), None)
            link = (doc or {}).get("urlHtml") or f"https://cima.aemps.es/cima/publico/detalle.html?nregistro={m['nregistro']}"
            cand = ((rank(short, not m.get("comerc")), short, doc["urlHtml"], "AEMPS", link)
                    if doc and doc.get("urlHtml") else None)
            add(groups, products, "es", vtm.lower(), "", form, short, link, bool(m.get("receta")), cand)
        print(f"\r  es: page {page}, {len(products)} products", end="", flush=True)
        if page * data.get("tamanioPagina", 200) >= total or not data.get("resultados"):
            break
        page += 1
    print()
    return groups, list(products.values())


# ------------------------------------------------------------------ France
def build_fr(_en) -> tuple[dict, list]:
    c = httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True)

    def tsv(name):
        path = EU / name
        if not path.exists():
            path.write_bytes(c.get(f"https://base-donnees-publique.medicaments.gouv.fr/download/file/{name}").content)
        return [line.split("\t") for line in path.read_bytes().decode("latin-1").splitlines() if line.strip()]

    subst = defaultdict(set)
    for row in tsv("CIS_COMPO_bdpm.txt"):
        if len(row) > 6 and row[6].strip() == "SA":
            subst[row[0]].add(row[3].strip().lower())
    groups, products = {}, {}
    for row in tsv("CIS_bdpm.txt"):
        cis, full, forme, voies, statut, _, comm = (x.strip() for x in row[:7])
        subs = sorted(subst.get(cis, []))
        if "active" not in statut.lower() or any("homéopath" in s or "homeopath" in s for s in subs):
            continue
        name = full.split(",")[0].strip()
        short = re.split(r"\s+\d", name, 1)[0].strip().title()
        key = " + ".join(subs) if subs and not PLACEHOLDER_INN.match(" ".join(subs)) else short
        form = bucket_eu(forme + " " + voies)
        url = f"https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid={cis}&typedoc=N"
        cand = (rank(short, "commercialis" not in comm.lower()), short, url, "ANSM", url)
        add(groups, products, "fr", key, "", form, short, url, True, cand)
    return groups, list(products.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="es,fr,cz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    EU.mkdir(parents=True, exist_ok=True)
    en = english_names()
    for m in args.markets.split(","):
        print(f"\n== {m}")
        groups, products = {"cz": build_cz, "es": build_es, "fr": build_fr}[m](en)
        index = finalize(m, groups, products)
        pil_zip = next(EU.glob("PIL*.zip"), None) if m == "cz" else None
        download(m, index, args.limit, args.workers, zip_path=pil_zip)
