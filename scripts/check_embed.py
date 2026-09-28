"""Check that the Hugging Face Inference API gives the same embeddings as the local model used for indexing.

Run:  python -m scripts.check_embed        (needs HF_TOKEN in .env)
"""
import numpy as np

from app import config, store

texts = ["Czy mogę brać Ibuprom z alkoholem?", "Чи можна Глюкофаж при вагітності?", "Doliprane et grossesse ?"]
for model in (config.MULTI_EMBED_MODEL, config.EMBED_MODEL):
    local = [np.array(next(iter(store.get_embedder(model).query_embed(t)))) for t in texts]
    remote = [np.array(v) for v in store.remote_embed(texts, model)]
    sims = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(local, remote)]
    print(model, "cosine(local, API) =", [round(s, 3) for s in sims], "OK" if min(sims) > 0.95 else "MISMATCH")
