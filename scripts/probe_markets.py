"""Step 1 of adding the Poland / Ukraine / Austria markets: download each official register's
product list and print its structure, so the ingestion script can be written against the real columns.

Sources (each country market = national register + EMA for centrally authorised EU medicines):
  PL  Rejestr Produktów Leczniczych (URPL / e-Zdrowie), leaflets via /api/rpl/medicinal-products/{id}/leaflet
  UA  Державний реєстр лікарських засобів (MOZ, CC BY 4.0 on data.gov.ua)
  AT  Arzneispezialitätenregister (BASG, data.gv.at)
  EU  EMA medicines list (EPAR product information exists in Polish and German)

Run:  python -m scripts.probe_markets      -> files in data/raw/, a short report printed.
"""
import csv
import io
import json
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "PharmaRAG portfolio project (educational)"}

SOURCES = {
    "pl_rpl.csv": ["https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl/medicinal-products/public-pl-report/get-old-csv",
                   "https://rejestry.ezdrowie.gov.pl/api/rpl/medicinal-products/public-pl-report/get-old-csv"],
    "ua_reestr.csv": ["http://www.drlz.com.ua/ibp/zvity.nsf/all/zvit/$file/reestr.csv"],
    "at_ckan.json": ["https://www.data.gv.at/katalog/api/3/action/package_show?id=arzneispezialitaetenregister"],
    "ema_medicines.xlsx": ["https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"],
}


def download(name: str, urls: list[str]) -> Path | None:
    for url in urls:
        try:
            r = httpx.get(url, headers=HEADERS, timeout=120, follow_redirects=True)
            print(f"  {r.status_code}  {len(r.content) / 1e6:.1f} MB  {url}")
            if r.status_code == 200 and r.content:
                path = RAW / name
                path.write_bytes(r.content)
                return path
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {type(e).__name__}: {e}  {url}")
    return None


def show_csv(path: Path) -> None:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251", "cp1250", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    dialect = csv.Sniffer().sniff(text[:5000], delimiters=";,\t|")
    rows = list(csv.reader(io.StringIO(text), dialect))
    print(f"  encoding={enc} delimiter={dialect.delimiter!r} rows={len(rows) - 1}")
    for i, col in enumerate(rows[0]):
        sample = next((r[i] for r in rows[1:50] if len(r) > i and r[i].strip()), "")
        print(f"   [{i}] {col[:60]}  =  {sample[:90]}")


def show_ckan(path: Path) -> None:
    res = json.loads(path.read_text("utf-8")).get("result", {})
    print(f"  license={res.get('license_id')}")
    for r in res.get("resources", []):
        print(f"   - {r.get('format')}  {r.get('name')}  {r.get('url')}")


def show_xlsx(path: Path) -> None:
    try:
        import openpyxl
    except ImportError:
        print("  (pip install openpyxl to inspect; file saved)")
        return
    ws = openpyxl.load_workbook(path, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    for _ in range(12):  # EMA puts a title block above the header row
        header = next(rows)
        if header and sum(v is not None for v in header) > 5:
            break
    first = next(rows)
    for i, col in enumerate(header):
        print(f"   [{i}] {str(col)[:60]}  =  {str(first[i])[:90] if i < len(first) else ''}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, urls in SOURCES.items():
        print(f"\n== {name}")
        path = download(name, urls)
        if not path:
            continue
        try:
            {"csv": show_csv, "json": show_ckan, "xlsx": show_xlsx}[name.rsplit(".", 1)[1]](path)
        except Exception as e:  # noqa: BLE001
            print(f"  could not parse: {e}")
