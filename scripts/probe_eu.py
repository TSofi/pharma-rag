"""Look at the structure of more European registers before writing their importers.

  CZ  SÚKL open data: DLP product database (CSV in a ZIP) + the monthly ZIP of all leaflets (PIL)
  ES  AEMPS CIMA REST API (leaflets as HTML, split into sections)
  FR  ANSM BDPM open data (tab-separated files) + one notice page per product
  IE  HPRA XML product listing (CC BY 4.0) with leaflet PDFs
  LV  ZVA register export (XML) + leaflet PDFs

Run:  python -m scripts.probe_eu     -> small files in data/raw/eu/, a report printed (copy it to Claude).
"""
import io
import json
import re
import zipfile
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "eu"
H = {"User-Agent": "PharmaRAG portfolio project (educational)"}
c = httpx.Client(headers=H, timeout=120, follow_redirects=True)


def head(r: httpx.Response) -> str:
    return f"{r.status_code} {r.headers.get('content-type', '')[:40]} {len(r.content) / 1e6:.2f} MB"


def show_table(name: str, raw: bytes, n_cols: int = 40) -> None:
    for enc in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()
    delim = max(";|,\t", key=lambda d: lines[0].count(d)) if lines else ";"
    cols = lines[0].split(delim) if lines else []
    first = lines[1].split(delim) if len(lines) > 1 else []
    print(f"   {name}: {len(lines) - 1} rows, enc={enc}, delim={delim!r}")
    for i, col in enumerate(cols[:n_cols]):
        print(f"     [{i}] {col[:40]} = {(first[i] if i < len(first) else '')[:70]}")


def section(title: str):
    print(f"\n== {title}")


def cz():
    section("CZ SÚKL")
    page = c.get("https://opendata.sukl.cz/?q=katalog%2Fdatabaze-lecivych-pripravku-dlp").text
    m = re.search(r"https://opendata\.sukl\.cz/soubory/SOD\d+/DLP\d+\.zip", page)
    url = m.group(0) if m else "https://opendata.sukl.cz/soubory/SOD20260925/DLP20260925.zip"
    r = c.get(url)
    print(" ", url, head(r))
    (RAW / "cz_dlp.zip").write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        print("   files:", ", ".join(names[:40]))
        for want in ("dlp_lecivepripravky", "dlp_nazvydokumentu", "dlp_slozeni", "dlp_formy"):
            f = next((n for n in names if want in n.lower()), None)
            if f:
                show_table(f, z.read(f))
    pil = re.search(r"https://opendata\.sukl\.cz/soubory/SOD\d+/PIL\d+\.zip",
                    c.get("https://opendata.sukl.cz/?q=katalog%2Fpil-pribalove-informace-product-information-leaflet").text)
    print("   PIL zip:", pil.group(0) if pil else "not found")


def es():
    section("ES CIMA")
    r = c.get("https://cima.aemps.es/cima/rest/medicamentos", params={"pagina": 1})
    print("  medicamentos?pagina=1", head(r))
    data = r.json()
    print("   totalFilas:", data.get("totalFilas"), "tamanioPagina:", data.get("tamanioPagina"))
    first = (data.get("resultados") or [{}])[0]
    print("   first:", json.dumps(first, ensure_ascii=False)[:1500])
    n = first.get("nregistro")
    if n:
        r = c.get("https://cima.aemps.es/cima/rest/docSegmentado/secciones/2", params={"nregistro": n})
        print("  secciones prospecto", head(r), r.text[:600])
        r = c.get(f"https://cima.aemps.es/cima/dochtml/p/{n}/Prospecto.html")
        print("  Prospecto.html", head(r))


def fr():
    section("FR BDPM")
    for f in ("CIS_bdpm.txt", "CIS_COMPO_bdpm.txt", "CIS_CPD_bdpm.txt"):
        r = c.get(f"https://base-donnees-publique.medicaments.gouv.fr/download/file/{f}")
        if r.status_code != 200:
            r = c.get(f"https://base-donnees-publique.medicaments.gouv.fr/telechargement.php?fichier={f}")
        print(" ", f, head(r))
        if r.status_code == 200:
            (RAW / f).write_bytes(r.content)
            show_table(f, r.content, 14)
    try:
        cis = (RAW / "CIS_bdpm.txt").read_bytes().decode("latin-1").split("\t", 1)[0]
        for u in (f"https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid={cis}&typedoc=N",
                  f"https://base-donnees-publique.medicaments.gouv.fr/medicament/{cis}/extrait"):
            r = c.get(u)
            print("  notice", u, head(r), "| has 'Notice':", "notice" in r.text.lower())
    except FileNotFoundError:
        pass


def ie():
    section("IE HPRA")
    r = c.get("https://assets.hpra.ie/products/xml/latestHumanlist.xml")
    print("  latestHumanlist.xml", head(r))
    (RAW / "ie_human.xml").write_bytes(r.content)
    text = r.text
    first = re.search(r"<(\w+)>\s*<(\w+)", text[:3000])
    print("   start:", re.sub(r"\s+", " ", text[:2500]))
    print("   PIL links found:", len(re.findall(r"PIL[^<\"]*\.pdf", text)))


def lv():
    section("LV ZVA")
    r = c.get("https://dati.zva.gov.lv/zalu-registrs/export/")
    print("  export page", head(r))
    links = re.findall(r'href="([^"]+\.(?:xml|zip|csv))"', r.text)
    print("   links:", links[:10])


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for fn in (cz, es, fr, ie, lv):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- keep probing the other countries
            print(f"   FAILED: {type(e).__name__}: {e}")
