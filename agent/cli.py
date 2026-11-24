"""Terminal chat with live streaming.

Usage:
    python -m agent.cli                    # LangGraph (CRAG + parallel tools + streaming)
    python -m agent.cli --engine react     # Hand-written ReAct baseline
"""
from __future__ import annotations

import argparse
import asyncio
import json

from agent.langgraph_agent import LangGraphDeStudyAgent
from agent.react_agent import DeStudyAgent, Step


def show_react_step(step: Step):
    if step.kind == "tool_call":
        print(f"  🔧 {step.name}({json.dumps(step.content, ensure_ascii=False)})")
    elif step.kind == "tool_result":
        ms = f" [{step.duration_ms:.0f}ms]" if step.duration_ms is not None else ""
        print(f"  ↳{ms} {str(step.content)[:180]}...")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["langgraph", "react"], default="langgraph")
    args = ap.parse_args()

    if args.engine == "langgraph":
        async with LangGraphDeStudyAgent() as agent:
            print("DeStudy Agent [LangGraph + CRAG + Streaming] 已启动（输入 exit 退出）\n")
            while True:
                q = input("你：").strip()
                if q.lower() in ("exit", "quit"):
                    break
                if not q:
                    continue
                started_text = False
                async for ev in agent.chat_stream(q):
                    if ev.kind == "status":
                        print(f"  {ev.content}")
                    elif ev.kind == "tool_call":
                        print(f"  🔧 {ev.name}({json.dumps(ev.content, ensure_ascii=False)})")
                    elif ev.kind == "tool_result":
                        ms = f" [{ev.duration_ms:.0f}ms]" if ev.duration_ms is not None else ""
                        print(f"  ↳{ms} {str(ev.content)[:160]}...")
                    elif ev.kind == "token":
                        if not started_text:
                            print("\nDeStudy：", end="", flush=True)
                            started_text = True
                        print(ev.content, end="", flush=True)
                tr = agent.last_trace
                print(f"\n\n  [⏱ {tr.latency_ms/1000:.2f}s | {tr.total_tokens} tok | route={tr.meta.get('route')}]\n")
    else:
        async with DeStudyAgent(on_step=show_react_step) as agent:
            print("DeStudy Agent [Hand-written ReAct] 已启动（输入 exit 退出）\n")
            while True:
                q = input("你：").strip()
                if q.lower() in ("exit", "quit"):
                    break
                if q:
                    print(f"\nDeStudy：{await agent.chat(q)}\n")


if __name__ == "__main__":
    asyncio.run(main())
