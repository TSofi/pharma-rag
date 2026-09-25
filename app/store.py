"""Embeddings + vector database.

- Embeddings: text -> vector of 384 numbers that captures its *meaning*.
  We use fastembed (small ONNX model, runs on CPU, free, no API key).
- Qdrant: stores the vectors together with a "payload" (drug name, section, source URL,
  original text) and finds the vectors closest to a query vector (cosine similarity).
"""
import time
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from . import config


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    # Downloaded once (~130 MB) and cached; lru_cache keeps one instance in memory.
    return TextEmbedding(model_name=config.EMBED_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in get_embedder().passage_embed(texts)]


def embed_query(text: str) -> list[float]:
    # Queries and passages are embedded slightly differently for BGE models.
    return next(iter(get_embedder().query_embed(text))).tolist()


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    return QdrantClient(path=config.QDRANT_LOCAL_PATH)  # embedded local mode


def recreate_collection(dim: int) -> None:
    client = get_client()
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Index on drug name so we can filter by drug quickly (metadata filtering).
    # (Only meaningful on a Qdrant server; local mode ignores indexes.)
    if config.QDRANT_URL:
        client.create_payload_index(config.COLLECTION, "drug", models.PayloadSchemaType.KEYWORD)


def upsert(ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    get_client().upsert(
        config.COLLECTION,
        points=[models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)],
    )


def search(query: str, top_k: int, drugs: list[str] | None = None, timings: dict | None = None) -> list[dict]:
    """Return the top_k most similar chunks. If `drugs` is given, only search those drugs.
    If a `timings` dict is passed, embedding and vector-search durations (ms) are recorded in it."""
    t0 = time.perf_counter()
    vector = embed_query(query)
    t1 = time.perf_counter()
    flt = None
    if drugs:
        flt = models.Filter(must=[models.FieldCondition(key="drug", match=models.MatchAny(any=drugs))])
    res = get_client().query_points(config.COLLECTION, query=vector, limit=top_k, query_filter=flt, with_payload=True)
    if timings is not None:
        timings["embed_ms"] = round((t1 - t0) * 1000)
        timings["vector_search_ms"] = round((time.perf_counter() - t1) * 1000)
    return [{**p.payload, "score": round(p.score, 3)} for p in res.points]

