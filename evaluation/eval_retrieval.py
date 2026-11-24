"""Retrieval evaluation + ablation: Recall@K and MRR per retrieval mode.

Test-set line format (evaluation/testset.jsonl):
  {"question": "...", "relevant": ["substring of a relevant URL or a key phrase", ...], "university": null}
A retrieved chunk counts as relevant if its URL contains, or its text contains
(case-insensitive), any of the `relevant` strings.

Usage:
  python -m evaluation.eval_retrieval                    # all modes, no query rewrite
  python -m evaluation.eval_retrieval --rewrite          # also run hybrid_rerank + LLM query rewrite
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rag.retriever import get_retriever

HERE = Path(__file__).parent
MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def is_relevant(hit, relevant: list[str]) -> bool:
    url, text = hit.chunk["url"].lower(), hit.chunk["text"].lower()
    return any(r.lower() in url or r.lower() in text for r in relevant)


def evaluate(cases: list[dict], mode: str, k: int, rewrite: bool = False, use_filter: bool = True) -> dict:
    retr = get_retriever()
    recall = mrr = 0.0
    misses = []
    for case in cases:
        extra = []
        if rewrite:
            from rag.query_rewrite import rewrite_queries

            extra = rewrite_queries(case["question"])
        uni = case.get("university") if use_filter else None
        hits = retr.search(case["question"], extra_queries=extra, top_k=k, mode=mode, university=uni)
        rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h, case["relevant"])), None)
        if rank:
            recall += 1
            mrr += 1 / rank
        else:
            misses.append(case["question"])
    n = len(cases)
    return {"mode": mode + ("+rewrite" if rewrite else ""), f"recall@{k}": recall / n, "mrr": mrr / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--no-filter", action="store_true", help="ignore university metadata filter")
    ap.add_argument("--modes", nargs="*", default=MODES)
    args = ap.parse_args()

    cases = [json.loads(l) for l in (HERE / "testset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    use_filter = not args.no_filter
    results = [evaluate(cases, m, args.k, use_filter=use_filter) for m in args.modes]
    if args.rewrite:
        results.append(evaluate(cases, "hybrid_rerank", args.k, rewrite=True, use_filter=use_filter))

    print(f"\n{len(cases)} questions, k={args.k}\n")
    print(f"| mode | recall@{args.k} | MRR |\n|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r[f'recall@{args.k}']:.3f} | {r['mrr']:.3f} |")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
