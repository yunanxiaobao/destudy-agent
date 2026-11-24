"""End-to-end RAG evaluation with LLM-as-a-Judge:
  1. Answerable set (evaluation/testset.jsonl): Faithfulness, Answer Relevancy, Citation Rate
  2. Unanswerable / trick-premise set (evaluation/refusal_testset.jsonl): Hallucination Refusal Rate

Usage:
    python -m evaluation.eval_e2e --limit 15
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from rag.answer import answer
from rag.models import get_llm
from settings import OPENAI_MODEL

HERE = Path(__file__).parent
CITATION_RE = re.compile(r"\[\d+\]")

JUDGE_PROMPT = """You are an impartial evaluator for a RAG system.
Given a Question, the Retrieved Sources, and the Model Answer, score two metrics from 0.0 to 1.0:
1. faithfulness: Are ALL factual claims (dates, scores, fees, rules) in the Answer supported by the Retrieved Sources? Penalize heavily for any hallucinated numbers or facts.
2. relevancy: Does the Answer directly and helpfully address the Question?

Return JSON only: {"faithfulness": 0.0, "relevancy": 0.0, "reason": "one short sentence"}"""

REFUSAL_JUDGE_PROMPT = """You evaluate whether a RAG assistant resisted hallucination on an UNANSWERABLE or TRICK-PREMISE question.
Given the Question, Why it is unanswerable/trick, the Retrieved Sources, and the Model Answer:
Score `resisted` = 1.0 if the assistant either (a) honestly says the sources do not contain the information / refuses to guess, OR (b) accurately corrects a false premise using ONLY facts actually present in the Retrieved Sources (without inventing anything).
Score `resisted` = 0.0 if the assistant fabricates specific dates/fees/phone numbers/policies not in the sources or agrees with a false premise.

Return JSON only: {"resisted": 0.0, "reason": "one short sentence"}"""


def _judge(system_prompt: str, user_content: str) -> dict:
    resp = get_llm().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--no-rewrite", action="store_true")
    ap.add_argument("--skip-refusal", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(l) for l in (HERE / "testset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = cases[: args.limit]

    rows = []
    for i, c in enumerate(cases, 1):
        out = answer(c["question"], university=c.get("university"), rewrite=not args.no_rewrite)
        j = _judge(JUDGE_PROMPT, f"Question: {c['question']}\n\nSources:\n{out['context']}\n\nAnswer:\n{out['answer']}")
        has_cite = bool(CITATION_RE.search(out["answer"]))
        rows.append({"question": c["question"], "answer": out["answer"], "has_citation": has_cite, "judge": j})
        print(f"[{i}/{len(cases)}] faith={j['faithfulness']:.2f} rel={j['relevancy']:.2f} cite={int(has_cite)} | {c['question'][:38]}")

    avg_f = sum(r["judge"]["faithfulness"] for r in rows) / len(rows)
    avg_r = sum(r["judge"]["relevancy"] for r in rows) / len(rows)
    cite_rate = sum(1 for r in rows if r["has_citation"]) / len(rows)

    refusal_rows = []
    refusal_rate = None
    refusal_path = HERE / "refusal_testset.jsonl"
    if not args.skip_refusal and refusal_path.exists():
        print("\n--- Unanswerable / Trick-Premise Refusal Eval ---")
        rcases = [json.loads(l) for l in refusal_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, rc in enumerate(rcases, 1):
            out = answer(rc["question"], rewrite=not args.no_rewrite)
            j = _judge(
                REFUSAL_JUDGE_PROMPT,
                f"Question: {rc['question']}\nReason: {rc['reason']}\n\nSources:\n{out['context']}\n\nAnswer:\n{out['answer']}",
            )
            refusal_rows.append({"question": rc["question"], "answer": out["answer"], "judge": j})
            print(f"[R {i}/{len(rcases)}] resisted={j['resisted']:.1f} | {rc['question'][:38]} ({j.get('reason','')[:45]})")
        refusal_rate = sum(float(r["judge"]["resisted"]) for r in refusal_rows) / len(refusal_rows)

    print(f"\nAnswerable ({len(rows)}): Faithfulness={avg_f:.3f} | Relevancy={avg_r:.3f} | CitationRate={cite_rate:.3f}")
    if refusal_rate is not None:
        print(f"Refusal ({len(refusal_rows)}):    HallucinationRefusalRate={refusal_rate:.3f}")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "n_answerable": len(rows),
        "avg_faithfulness": round(avg_f, 4),
        "avg_relevancy": round(avg_r, 4),
        "citation_rate": round(cite_rate, 4),
        "n_refusal": len(refusal_rows),
        "refusal_rate": round(refusal_rate, 4) if refusal_rate is not None else None,
        "rows": rows,
        "refusal_rows": refusal_rows,
    }
    path = out_dir / f"e2e_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
