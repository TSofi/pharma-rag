"""Step 2 of the data pipeline: chunk -> embed -> store in Qdrant.

Reads data/labels.jsonl, splits each label section into overlapping chunks (~900 chars),
turns each chunk into a vector with fastembed and upserts it into Qdrant together with
its metadata (drug, section, source URL). Re-running it rebuilds the collection from scratch.

Local:  python scripts/ingest.py            (writes to ./qdrant_data)
Cloud:  set QDRANT_URL + QDRANT_API_KEY in .env, then run the same command.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, store  # noqa: E402

CHUNK_SIZE = 900
OVERLAP = 150
BATCH = 64


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split on sentence boundaries, pack sentences into ~size-char chunks, and carry the last
    ~overlap chars into the next chunk so a fact split across a boundary isn't lost."""
    sentences = re.split(r"(?<=[.;:])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] if overlap else ""
        cur = f"{cur} {s}".strip()
        while len(cur) > size * 1.5:  # a single huge "sentence" (e.g. a table): hard split
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks() -> list[dict]:
    out = []
    with open(config.DATA_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for sec in rec["sections"]:
                for i, piece in enumerate(chunk_text(sec["text"])):
                    out.append({
                        "drug": rec["drug"],
                        "brands": rec["brands"],
                        "section": sec["section"],
                        "text": piece,
                        "chunk_index": i,
                        "url": rec["url"],
                        "set_id": rec["set_id"],
                        "effective_time": rec["effective_time"],
                    })
    return out


def main():
    if not config.DATA_FILE.exists():
        sys.exit("data/labels.jsonl not found - run scripts/fetch_labels.py first")
    chunks = build_chunks()
    target = config.QDRANT_URL or config.QDRANT_LOCAL_PATH
    print(f"{len(chunks)} chunks from {config.DATA_FILE.name} -> Qdrant ({target})")

    # "Contextual" embedding: prepend drug + section so a chunk like "Take 500 mg twice daily"
    # still carries WHICH drug and WHICH section it belongs to.
    texts = [f"{c['drug']} ({', '.join(c['brands'][:2])}) - {c['section']}: {c['text']}" for c in chunks]

    t0 = time.time()
    first = store.embed_passages(texts[:1])
    store.recreate_collection(dim=len(first[0]))
    for start in range(0, len(chunks), BATCH):
        batch_texts = texts[start:start + BATCH]
        vectors = store.embed_passages(batch_texts)
        ids = list(range(start, start + len(batch_texts)))
        store.upsert(ids, vectors, chunks[start:start + BATCH])
        print(f"  {min(start + BATCH, len(chunks))}/{len(chunks)}", end="\r")
    store.get_client().close()
    print(f"\nDone in {time.time() - t0:.0f}s. Collection '{config.COLLECTION}' is ready.")


if __name__ == "__main__":
    main()
