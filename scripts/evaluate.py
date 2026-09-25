"""Evaluate the RAG pipeline on eval/questions.jsonl and write eval/report.md.

Automatic metrics:
  - retrieval hit@k : did retrieval return a chunk from the expected drug AND section?
  - refusal accuracy: did the system refuse exactly when it should?
  - citation rate   : does every answered question contain at least one [n] citation?
The report also has an empty "Manual verdict" column: read each answer against its sources
and mark it correct / partially correct / hallucination.

Run:  python scripts/evaluate.py                 (retrieval + LLM answers)
      python scripts/evaluate.py --retrieval-only  (no LLM calls, costs nothing)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, rag, store  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "eval"


def retrieval_hit(chunks: list[dict], item: dict) -> bool | None:
    if item["should_refuse"]:
        return None
    return any(c["drug"] in item["expect_drugs"] and c["section"] in item["expect_sections"] for c in chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=4.0, help="pause between LLM calls (free-tier rate limits)")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(EVAL / "questions.jsonl", encoding="utf-8") if l.strip()]
    rows = []
    for it in items:
        drugs = rag.detect_drugs(it["question"])
        chunks = store.search(it["question"], config.TOP_K, drugs or None)
        hit = retrieval_hit(chunks, it)
        row = {**it, "detected": drugs, "hit": hit, "top": chunks[:3]}
        if not args.retrieval_only:
            res = rag.ask(it["question"])
            row.update(answer=res["answer"], refused=not res["found"],
                       cited=sum(s["cited"] for s in res["sources"]))
            time.sleep(args.sleep)
        rows.append(row)
        flag = {True: "HIT ", False: "MISS", None: "  - "}[hit]
        print(f"[{flag}] #{it['id']:>2} {it['question']}")

    # ---- metrics ----
    graded = [r for r in rows if r["hit"] is not None]
    hit_rate = sum(r["hit"] for r in graded) / len(graded)
    lines = ["# Evaluation report", "",
             f"- Questions: **{len(rows)}**  ·  top_k = {config.TOP_K}  ·  min_score = {config.MIN_SCORE}  ·  LLM = {config.LLM_PROVIDER}",
             f"- Retrieval hit@{config.TOP_K}: **{sum(r['hit'] for r in graded)}/{len(graded)} ({hit_rate:.0%})**"]
    if not args.retrieval_only:
        refusal_ok = sum(r["refused"] == r["should_refuse"] for r in rows)
        answered = [r for r in rows if not r["refused"]]
        with_cite = sum(r["cited"] > 0 for r in answered)
        lines += [f"- Refusal accuracy: **{refusal_ok}/{len(rows)}**",
                  f"- Answers with at least one citation: **{with_cite}/{len(answered)}**"]
    lines += ["", "| # | Type | Question | Retrieval | Top-3 retrieved (drug · section · score) |"
              + ("" if args.retrieval_only else " Answer | Refused (expected) | Manual verdict |"),
              "|---|---|---|---|---|" + ("" if args.retrieval_only else "---|---|---|")]
    for r in rows:
        top = "<br>".join(f"{c['drug']} · {c['section']} · {c['score']}" for c in r["top"])
        cells = [str(r["id"]), r["type"], r["question"], {True: "✅", False: "❌", None: "—"}[r["hit"]], top]
        if not args.retrieval_only:
            ans = r["answer"].replace("\n", " ").replace("|", "\\|")
            cells += [ans[:400] + ("…" if len(ans) > 400 else ""),
                      f"{'yes' if r['refused'] else 'no'} ({'yes' if r['should_refuse'] else 'no'})", ""]
        lines.append("| " + " | ".join(cells) + " |")

    store.get_client().close()
    out = EVAL / ("report_retrieval.md" if args.retrieval_only else "report.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nRetrieval hit@{config.TOP_K}: {hit_rate:.0%}  ->  {out.relative_to(EVAL.parent)}")


if __name__ == "__main__":
    main()
