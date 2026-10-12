"""Can both embedding models run inside Render's free 512 MB if they are quantized to int8?

    python -m scripts.check_int8

1. Downloads both fastembed models into models_cache/, puts int8 versions into models_int8/.
2. Compares int8 vs original vectors (cosine) and Qdrant top-10 results on real questions.
3. Measures the memory of the API process (all country indexes + both models) in both variants.
"""
import glob
import os
import shutil
import subprocess
import sys

MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
CACHE, OUT = "models_cache", "models_int8"
QUESTIONS = {
    "pl": ["Czy mogę brać Ibuprom z alkoholem?", "dlaczego stosuje sie laremid"],
    "ua": ["Чи можна нурофен при вагітності?", "що робити коли насморк"],
    "cz": ["Mohu užívat Paralen s alkoholem?"],
    "es": ["¿Puedo tomar ibuprofeno en el embarazo?"],
    "fr": ["Puis-je prendre du Doliprane avec de l'alcool ?"],
    "us": ["Can I take Advil with alcohol?", "What are the side effects of Eliquis?"],
}


def mem_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes as w

        class PMC(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "a", "b", "c", "d",
                                               "PagefileUsage", "PeakPagefileUsage")]
        c = PMC(); c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c),
                                                 c.cb)
        return max(c.PeakWorkingSetSize, c.PeakPagefileUsage) / 1e6
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def model_dir(model: str) -> str:
    return glob.glob(os.path.join(CACHE, f"models--*{model.split('/')[1]}*", "snapshots", "*"))[0]


# fastembed's files are float16, which ONNX Runtime can't quantize directly, so the int8 weights come from
# Xenova's exports of the same models (int8, quantized from the original float32 weights).
INT8_SOURCE = {m: (f"Xenova/{m.split('/')[1]}", "onnx/model_quantized.onnx") for m in MODELS}


def build() -> None:
    from fastembed import TextEmbedding
    from huggingface_hub import hf_hub_download
    for m in MODELS:
        TextEmbedding(m, cache_dir=CACHE)
        src, dst = model_dir(m), os.path.join(OUT, m.split("/")[1])
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=False)  # tokenizer + config from fastembed
        repo, fname = INT8_SOURCE[m]
        q = hf_hub_download(repo, fname, cache_dir=CACHE)
        target = os.path.join(dst, "model_optimized.onnx")
        before = os.path.getsize(target) / 1e6
        shutil.copyfile(q, target)
        print(f"  {m}: {before:.0f} MB -> {os.path.getsize(target) / 1e6:.0f} MB")


def embedder(m: str, int8: bool):
    from fastembed import TextEmbedding
    if int8:
        return TextEmbedding(m, specific_model_path=os.path.join(OUT, m.split("/")[1]), cache_dir=CACHE)
    return TextEmbedding(m, cache_dir=CACHE)


def compare() -> None:
    import numpy as np
    from app import config, store
    client = store.get_client()
    for m in MODELS:
        a, b = embedder(m, False), embedder(m, True)
        for mk, qs in QUESTIONS.items():
            if config.MARKETS[mk]["model"] != m:
                continue
            for q in qs:
                va = np.array(next(iter(a.query_embed(q)))); vb = np.array(next(iter(b.query_embed(q))))
                cos = float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))
                coll = config.MARKETS[mk]["collection"]
                ta = [p.id for p in client.query_points(coll, query=va.tolist(), limit=10).points]
                tb = [p.id for p in client.query_points(coll, query=vb.tolist(), limit=10).points]
                print(f"  {mk}  cosine {cos:.4f}  same top-10: {len(set(ta) & set(tb))}/10  "
                      f"same #1: {ta[:1] == tb[:1]}  | {q}")


def measure(int8: bool) -> None:
    from app import config, markets
    for mk in config.MARKETS:
        markets.index(mk)
    base = mem_mb()
    for m in MODELS:
        e = embedder(m, int8)
        list(e.query_embed("warm up question about ibuprofen"))
    print(f"  {'int8' if int8 else 'original'}: indexes only {base:.0f} MB, + both models {mem_mb():.0f} MB (peak)")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # child process: measure one variant in a clean process
        measure(sys.argv[1] == "int8")
        sys.exit()
    print("1) quantizing"); build()
    print("2) quality (int8 vs original)"); compare()
    print("3) memory of the API process")
    for v in ("original", "int8"):
        subprocess.run([sys.executable, "-m", "scripts.check_int8", v], check=False)
