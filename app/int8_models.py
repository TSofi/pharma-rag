"""int8 versions of the two embedding models (4x smaller than float32, ~2x smaller than fastembed's float16).

They let both models run on Render's free 512 MB instance without an embedding API. Quality vs the models the
leaflets were indexed with (scripts/check_int8.py): cosine 0.988-0.997 on real questions, same top-1 leaflet
chunk in most cases, so the existing Qdrant collections stay valid (no re-indexing).

Built once into the Docker image:  python -m app.int8_models
"""
import glob
import os
import shutil

from . import config

# fastembed ships these models as float16, which ONNX Runtime can't re-quantize, so the int8 weights come from
# Xenova's exports of the same models (quantized from the original float32 weights).
SOURCES = {
    "BAAI/bge-small-en-v1.5": "Xenova/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
}


def path(model: str) -> str | None:
    """Folder with the int8 model, if it has been built."""
    d = os.path.join(config.INT8_MODEL_DIR, model.split("/")[1])
    return d if os.path.exists(os.path.join(d, "model_optimized.onnx")) else None


def build(cache_dir: str | None = None) -> None:
    from fastembed import TextEmbedding
    from huggingface_hub import hf_hub_download
    cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE_PATH") or "models_cache"
    for model, repo in SOURCES.items():
        TextEmbedding(model, cache_dir=cache_dir)  # tokenizer + config in the layout fastembed expects
        name = model.split("/")[1]
        src = glob.glob(os.path.join(cache_dir, f"models--*{name}*", "snapshots", "*"))[0]
        dst = os.path.join(config.INT8_MODEL_DIR, name)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=False)
        shutil.copyfile(hf_hub_download(repo, "onnx/model_quantized.onnx", cache_dir=cache_dir),
                        os.path.join(dst, "model_optimized.onnx"))
        print(f"{model}: int8 model in {dst}", flush=True)


if __name__ == "__main__":
    build()
