"""Index the Polish / Ukrainian leaflets: sections -> chunks -> multilingual vectors -> Qdrant.

Reads data/raw/<market>/docs.jsonl (from scripts.fetch_markets) and writes the collection straight to
Qdrant Cloud when QDRANT_URL is set in .env (otherwise to the local ./qdrant_data folder).

Run:   python -m scripts.ingest_markets                 (both markets, rebuilds the collections)
       python -m scripts.ingest_markets --markets pl --limit 30   (quick test on 30 leaflets)
       python -m scripts.ingest_markets --resume        (continue after an interruption)
"""
import argparse
import json
import time
import uuid

from app import config, store
from scripts.doc_text import QRD, sections_pl, sections_ua
from scripts.ingest import chunk_text

BATCH = 128
# Long pharmacology / packaging sections rarely answer patient or pharmacist questions but would take
# ~20% of the storage, so they are left out.
SKIP_SECTIONS = {"Pharmacological properties", "Package", "Shelf life", "Manufacturer"}


def popular_groups(market: str, top: int) -> set[str]:
    """The `top` groups with the most products on the market (many brands/generics = a commonly used drug)."""
    idx = json.loads((config.ROOT / "data" / "markets" / f"{market}.json").read_text("utf-8"))
    count: dict[str, int] = {}
    for _name, gid, _url, _rx in idx["products"]:
        count[gid] = count.get(gid, 0) + 1
    keep = set(sorted(count, key=count.get, reverse=True)[:top])
    # Patented drugs have few manufacturers (Eliquis, Ozempic…), so also keep the curated starter list.
    from scripts.fetch_labels import DRUGS
    return keep | {gid for gid in idx["groups"] if any(f":{g.split()[0]}" in gid for g in DRUGS)}


def build_chunks(market: str, limit: int | None, top: int | None = None, docs: str = "docs.jsonl") -> list[dict]:
    # EU leaflets share one template (QRD); Ukraine has its own; US docs arrive already split into sections.
    split = sections_ua if market == "ua" else (lambda t: sections_pl(t, market if market in QRD else "pl"))
    out = []
    keep = popular_groups(market, top) if top else None
    with open(config.ROOT / "data" / "raw" / market / docs, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                break
            d = json.loads(line)
            if keep is not None and d["group"] not in keep:
                continue
            sections = [(x["section"], x["text"]) for x in d["sections"]] if "sections" in d else split(d["text"])
            for section, text in sections:
                if section in SKIP_SECTIONS:
                    continue
                for i, piece in enumerate(chunk_text(text)):
                    out.append({"group": d["group"], "drug": d["name"], "inn": d["inn"], "en": d["en"],
                                "form": d["form"], "section": section, "text": piece, "chunk_index": i,
                                "url": d["url"], "source": d["source"], "market": market,
                                "effective_time": d.get("effective_time", "")})
    return out


def point_id(c: dict) -> str:
    # Deterministic id: re-running (or resuming) overwrites the same points instead of duplicating them.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['group']}|{c['section']}|{c['chunk_index']}"))


def ingest(market: str, limit: int | None, resume: bool, top: int | None = None,
           docs: str = "docs.jsonl", append: bool = False) -> None:
    """append=True adds to an existing collection (e.g. EMA leaflets) instead of rebuilding it."""
    m = config.MARKETS[market]
    chunks = build_chunks(market, limit, top, docs)
    target = config.QDRANT_URL or config.QDRANT_LOCAL_PATH
    print(f"\n== {market}: {len(chunks)} chunks from {len({c['group'] for c in chunks})} leaflets -> {target}")
    # Contextual embedding: product + substance + section travel with every chunk.
    texts = [f"{c['drug']} ({c['inn']}{'; ' + c['en'] if c['en'] else ''}) - {c['section']}: {c['text']}"
             for c in chunks]
    client = store.get_client()
    start_at = 0
    if append and client.collection_exists(m["collection"]):
        pass  # keep what's there; deterministic point ids make re-runs overwrite, not duplicate
    elif resume and client.collection_exists(m["collection"]):
        start_at = max(0, client.count(m["collection"]).count - BATCH)  # redo the last batch to be safe
        print(f"  resuming at chunk {start_at}")
    else:
        dim = len(store.embed_passages(texts[:1], m["model"])[0])
        store.recreate_collection(dim, m["collection"], keyword_fields=("group",), compact=True)
    t0 = time.time()
    for s in range(start_at, len(chunks), BATCH):
        batch = chunks[s:s + BATCH]
        vectors = store.embed_passages(texts[s:s + BATCH], m["model"])
        store.upsert([point_id(c) for c in batch], vectors, batch, m["collection"])
        done = s + len(batch)
        eta = (time.time() - t0) / max(1, done - start_at) * (len(chunks) - done) / 60
        print(f"\r  {done}/{len(chunks)}  ~{eta:.0f} min left ", end="", flush=True)
    print(f"\n  done in {(time.time() - t0) / 60:.1f} min")
    if not limit:  # record which groups are really searchable (the UI library lists only those)
        path = config.ROOT / "data" / "markets" / f"{market}.json"
        idx = json.loads(path.read_text("utf-8"))
        reps = {c["group"]: c["drug"] for c in chunks}
        for gid, g in idx["groups"].items():
            g["indexed"] = gid in reps or (append and g.get("indexed", False))
            if gid in reps:
                g["rep"] = reps[gid]
        path.write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), "utf-8")
        print(f"  {len(reps)} searchable groups written to {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pl,ua", help="pl, ua and/or us")
    ap.add_argument("--limit", type=int, default=None, help="only the first N leaflets (testing)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--top", type=int, default=None, help="only the N most common drugs (by number of products)")
    ap.add_argument("--docs", default="docs.jsonl", help="which file in data/raw/<market>/ to index")
    ap.add_argument("--append", action="store_true", help="add to the existing collection instead of rebuilding it")
    args = ap.parse_args()
    for mk in args.markets.split(","):
        ingest(mk, args.limit, args.resume, args.top, args.docs, args.append)
    store.get_client().close()
