"""See what happens inside one question: detected products, retrieved passages and the model used.

Run:  python -m scripts.debug_ask pl "Czy mogę brać Ibuprom z alkoholem?"
"""
import sys
import time

from app import config, llm, markets, store

market, question = sys.argv[1], " ".join(sys.argv[2:])
found = markets.detect(market, question) if market != "us" or config.US_FULL else []
print("detected:", [(f["asked"], len(f["groups"])) for f in found])
groups = sorted({g for f in found for g in f["groups"]})
t = time.time()
chunks = store.search(question, top_k=8, drugs=groups or None, market=market,
                      field="group" if (market != "us" or config.US_FULL) else "drug")
print(f"search {time.time() - t:.1f}s, min_score={config.MARKETS[market]['min_score']}")
for c in chunks:
    print(f"  {c['score']:.2f}  {c['drug'][:25]:<25} {c['section'][:40]:<40} {c['text'][:90]!r}")
t = time.time()
print("LLM test:", llm.generate("Reply with one word.", "Say OK")[:40], f"({time.time() - t:.1f}s via {llm.last_model_used})")
