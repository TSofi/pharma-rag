"""Copy the local vector index (./qdrant_data) to Qdrant Cloud, without re-computing embeddings.

The vectors were already computed by scripts/ingest.py, so we just read them (with their payloads)
from the local embedded Qdrant and upsert them into the cloud cluster. Takes seconds, not minutes.

Needs QDRANT_URL and QDRANT_API_KEY in .env.
Run:  python scripts/upload_to_cloud.py
"""
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

BATCH = 256


def main():
    if not config.QDRANT_URL:
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY in .env first.")

    local = QdrantClient(path=config.QDRANT_LOCAL_PATH)
    cloud = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=60)

    info = local.get_collection(config.COLLECTION)
    dim = info.config.params.vectors.size
    total = local.count(config.COLLECTION).count
    print(f"Local collection '{config.COLLECTION}': {total} vectors, {dim} dimensions")

    if cloud.collection_exists(config.COLLECTION):
        cloud.delete_collection(config.COLLECTION)
    cloud.create_collection(
        config.COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    # Keyword index on "drug" makes the drug-name filter fast on the server.
    cloud.create_payload_index(config.COLLECTION, "drug", models.PayloadSchemaType.KEYWORD)

    offset, sent = None, 0
    while True:
        points, offset = local.scroll(config.COLLECTION, limit=BATCH, offset=offset,
                                      with_payload=True, with_vectors=True)
        if not points:
            break
        cloud.upsert(config.COLLECTION, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
        ])
        sent += len(points)
        print(f"  uploaded {sent}/{total}", end="\r")
        if offset is None:
            break

    local.close()
    print(f"\nDone. Qdrant Cloud now has {cloud.count(config.COLLECTION).count} vectors.")


if __name__ == "__main__":
    main()
