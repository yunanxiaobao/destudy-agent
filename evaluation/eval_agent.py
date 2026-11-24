"""Agent trajectory & task evaluation:
  - Tool Selection Accuracy: did the agent invoke all expected MCP tools?
  - Task Pass Rate: did the final answer contain required verified facts / refusal?
  - Efficiency: avg LLM calls, avg tools, P50/P95 latency, avg total tokens

Usage:
    python -m evaluation.eval_agent             # run all cases in agent_testset.jsonl
    python -m evaluation.eval_agent --limit 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from agent.langgraph_agent import LangGraphDeStudyAgent
from agent.memory import Profile
from agent.react_agent import DeStudyAgent
from agent.tracer import summarize_traces

HERE = Path(__file__).parent


def _norm(text: str) -> str:
    return text.replace(",", "").replace("，", "").lower()


def _check_answer(answer: str, must_contain: list[str], match_mode: str = "all") -> bool:
    if not must_contain:
        return True
    hay = _norm(answer)
    hits = [_norm(tok) in hay for tok in must_contain]
    return any(hits) if match_mode == "any" else all(hits)


async def run_eval(cases: list[dict], engine: str = "react") -> dict:
    rows = []
    traces = []
    cls = LangGraphDeStudyAgent if engine == "langgraph" else DeStudyAgent
    # Reuse one MCP subprocess across all cases (fresh history & empty profile per case)
    async with cls(profile=Profile(), record_trace=False) as agent:
        for i, c in enumerate(cases, 1):
            agent.history.clear()
            agent.profile = Profile()
            ans = await agent.chat(c["question"])
            tr = agent.last_trace.to_dict()
            traces.append(tr)

            called = set(tr["tool_names"])
            expected = set(c.get("expected_tools", []))
            tool_ok = expected.issubset(called)
            if not tool_ok and expected == {"search_knowledge"} and "search_programs" in called:
                tool_ok = True
            ans_ok = _check_answer(ans, c.get("must_contain", []), c.get("match_mode", "all"))
            passed = tool_ok and ans_ok
            meta = tr.get("meta") or {}

            rows.append({
                "question": c["question"],
                "expected_tools": sorted(expected),
                "called_tools": tr["tool_names"],
                "route": meta.get("route"),
                "crag_rewrites": meta.get("crag_rewrites", 0),
                "tool_ok": tool_ok,
                "answer_ok": ans_ok,
                "passed": passed,
                "latency_ms": tr["latency_ms"],
                "total_tokens": tr["total_tokens"],
                "num_llm_calls": tr["num_llm_calls"],
                "answer_preview": ans[:300],
            })
            route_tag = f" [{meta['route']}]" if meta.get("route") else ""
            print(
                f"[{i:2d}/{len(cases)}]{route_tag} tool={'OK' if tool_ok else 'NO'} ans={'OK' if ans_ok else 'NO'} "
                f"| {tr['latency_ms']/1000:4.1f}s | {tr['total_tokens']:5d} tok | tools={tr['tool_names']}"
            )

    summary = summarize_traces(traces)
    n = len(rows)
    tool_acc = sum(1 for r in rows if r["tool_ok"]) / n
    ans_acc = sum(1 for r in rows if r["answer_ok"]) / n
    pass_rate = sum(1 for r in rows if r["passed"]) / n

    return {
        "engine": engine,
        "n_cases": n,
        "tool_accuracy": round(tool_acc, 4),
        "answer_accuracy": round(ans_acc, 4),
        "pass_rate": round(pass_rate, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "p50_latency_ms": summary["p50_latency_ms"],
        "p95_latency_ms": summary["p95_latency_ms"],
        "avg_tokens": summary["avg_tokens"],
        "avg_tools": summary["avg_tools"],
        "tool_error_rate": summary["tool_error_rate"],
        "tool_counts": summary["tool_counts"],
        "crag_rewrites_total": sum(r.get("crag_rewrites", 0) for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["react", "langgraph"], default="react")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = [json.loads(l) for l in (HERE / "agent_testset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        cases = cases[: args.limit]

    report = asyncio.run(run_eval(cases, engine=args.engine))
    print(
        f"\nAgent Eval [{report['engine']}] ({report['n_cases']} tasks):\n"
        f"  Tool Selection Accuracy : {report['tool_accuracy']*100:.1f}%\n"
        f"  Answer Factuality       : {report['answer_accuracy']*100:.1f}%\n"
        f"  Overall Pass Rate       : {report['pass_rate']*100:.1f}%\n"
        f"  Avg / P50 / P95 Latency : {report['avg_latency_ms']/1000:.2f}s / {report['p50_latency_ms']/1000:.2f}s / {report['p95_latency_ms']/1000:.2f}s\n"
        f"  Avg Tokens / Turn       : {report['avg_tokens']:.0f}\n"
        f"  Avg Tools / Turn        : {report['avg_tools']:.2f} (error rate {report['tool_error_rate']*100:.1f}%)"
    )

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"agent_{args.engine}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
