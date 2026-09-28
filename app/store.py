"""Embeddings + vector database.

- Embeddings: text -> vector of 384 numbers that captures its *meaning*.
  We use fastembed (small ONNX models, run on CPU, free, no API key). English FDA labels use a
  compact English model; Polish/Ukrainian leaflets use a multilingual one.
- Qdrant: stores the vectors together with a "payload" (drug name, section, source URL,
  original text) and finds the vectors closest to a query vector (cosine similarity).
  One collection per market (US, PL, UA).
"""
import time
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from . import config


@lru_cache(maxsize=2)
def get_embedder(model: str | None = None) -> TextEmbedding:
    # Downloaded once and cached; lru_cache keeps one instance per model in memory.
    return TextEmbedding(model_name=model or config.EMBED_MODEL)


def embed_passages(texts: list[str], model: str | None = None) -> list[list[float]]:
    return [v.tolist() for v in get_embedder(model).passage_embed(texts)]


def embed_query(text: str, model: str | None = None) -> list[float]:
    # Queries and passages are embedded slightly differently for BGE models.
    return next(iter(get_embedder(model).query_embed(text))).tolist()


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=60)
    return QdrantClient(path=config.QDRANT_LOCAL_PATH)  # embedded local mode


def recreate_collection(dim: int, name: str | None = None, keyword_fields: tuple[str, ...] = ("drug",),
                        compact: bool = False) -> None:
    """compact=True (large market collections): full vectors + payload live on disk, and a 4x smaller
    int8 copy of the vectors stays in RAM for fast search. That's what lets ~150k chunks fit into the
    free 1 GB Qdrant Cloud cluster."""
    name = name or config.COLLECTION
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    kwargs = {}
    if compact and config.QDRANT_URL:
        kwargs = dict(
            on_disk_payload=True,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
        )
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=compact),
        **kwargs,
    )
    # Keyword index so we can filter by drug / group quickly (metadata filtering).
    # (Only meaningful on a Qdrant server; local mode ignores indexes.)
    if config.QDRANT_URL:
        for field in keyword_fields:
            client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def upsert(ids: list, vectors: list[list[float]], payloads: list[dict], name: str | None = None) -> None:
    get_client().upsert(
        name or config.COLLECTION,
        points=[models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)],
    )


def search(query: str, top_k: int, drugs: list[str] | None = None, timings: dict | None = None,
           market: str = "us", field: str = "drug") -> list[dict]:
    """Return the top_k most similar chunks of one market. If `drugs` is given, only search chunks whose
    `field` (drug for US, group for PL/UA) is one of them. `timings` collects embed/search durations (ms)."""
    m = config.MARKETS[market]
    t0 = time.perf_counter()
    vector = embed_query(query, m["model"])
    t1 = time.perf_counter()
    flt = None
    if drugs:
        flt = models.Filter(must=[models.FieldCondition(key=field, match=models.MatchAny(any=drugs))])
    res = get_client().query_points(m["collection"], query=vector, limit=top_k, query_filter=flt, with_payload=True)
    if timings is not None:
        timings["embed_ms"] = round((t1 - t0) * 1000)
        timings["vector_search_ms"] = round((time.perf_counter() - t1) * 1000)
    return [{**p.payload, "score": round(p.score, 3)} for p in res.points]
