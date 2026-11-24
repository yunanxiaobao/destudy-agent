"""Streamlit UI with 3 tabs + live streaming + LangGraph / ReAct engine switch:
  1. 申请顾问对话 (Live node status + streaming output + tool calls + latency/tokens)
  2. 德国项目库筛选 (SQLite interactive explorer over 1,611 DAAD + official programs)
  3. 可观测性与评测面板 (Live traces P50/P95/tokens/tools + latest benchmark reports)

Usage: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.langgraph_agent import LangGraphDeStudyAgent, StreamEvent  # noqa: E402
from agent.react_agent import DeStudyAgent, Step  # noqa: E402
from agent.tracer import load_traces, summarize_traces  # noqa: E402
from data_pipeline.programs_db import db_stats, query_programs  # noqa: E402
from settings import ROOT  # noqa: E402


class AgentRuntime:
    """Keeps one agent (and its MCP server subprocess) alive on a background event loop."""

    def __init__(self, engine: str = "langgraph"):
        self.engine = engine
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.steps: list[Step] = []
        if engine == "langgraph":
            self.agent = LangGraphDeStudyAgent()
        else:
            self.agent = DeStudyAgent(on_step=self.steps.append)
        asyncio.run_coroutine_threadsafe(self.agent.__aenter__(), self.loop).result(timeout=120)

    def stream_chat(self, q: str):
        """Yield StreamEvent objects synchronously from the background asyncio loop."""
        if self.engine == "langgraph":
            sync_q: queue.Queue[StreamEvent | Exception | None] = queue.Queue()

            async def _pump():
                try:
                    async for ev in self.agent.chat_stream(q):
                        sync_q.put(ev)
                except Exception as exc:
                    sync_q.put(exc)
                finally:
                    sync_q.put(None)

            asyncio.run_coroutine_threadsafe(_pump(), self.loop)
            while True:
                item = sync_q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        else:
            self.steps.clear()
            ans = asyncio.run_coroutine_threadsafe(self.agent.chat(q), self.loop).result(timeout=300)
            for s in self.steps:
                if s.kind in ("tool_call", "tool_result"):
                    yield StreamEvent(s.kind, s.name, s.content, s.duration_ms)
            yield StreamEvent("token", content=ans)
            yield StreamEvent("done", content=ans)

    def last_trace_dict(self) -> dict | None:
        return self.agent.last_trace.to_dict() if self.agent.last_trace else None


@st.cache_resource(show_spinner="启动 Agent 与 MCP Server…")
def runtime(engine: str) -> AgentRuntime:
    return AgentRuntime(engine)


def render_steps(steps: list[dict], trace: dict | None = None):
    calls = [s for s in steps if s["kind"] == "tool_call"]
    statuses = [s for s in steps if s["kind"] == "status"]
    if calls or statuses:
        label = f"🔧 执行轨迹（{len(calls)} 次工具调用）" if calls else "🧭 执行轨迹"
        with st.expander(label):
            for s in steps:
                if s["kind"] == "status":
                    st.caption(s["content"])
                elif s["kind"] == "tool_call":
                    st.markdown(f"**{s['name']}**")
                    st.code(json.dumps(s["content"], ensure_ascii=False, indent=1), language="json")
                elif s["kind"] == "tool_result":
                    ms = f" ({s['duration_ms']:.0f} ms)" if s.get("duration_ms") is not None else ""
                    txt = str(s["content"])
                    st.caption(f"↳{ms} " + txt[:500] + ("…" if len(txt) > 500 else ""))
    if trace:
        meta = trace.get("meta") or {}
        extra = ""
        if meta.get("route"):
            extra = f" · 🧭 {meta['route']}"
            if meta.get("crag_rewrites"):
                extra += f" (CRAG 重试×{meta['crag_rewrites']})"
        st.caption(
            f"⚙️ {trace.get('engine', 'react')} · ⏱ {trace['latency_ms']/1000:.2f}s · "
            f"🔢 {trace['total_tokens']} tokens (in {trace['prompt_tokens']} / out {trace['completion_tokens']}) · "
            f"🤖 {trace['num_llm_calls']} LLM calls{extra}"
        )


def _latest_result(prefix: str) -> dict | list | None:
    res_dir = ROOT / "evaluation" / "results"
    if not res_dir.exists():
        return None
    files = sorted(res_dir.glob(f"{prefix}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


st.set_page_config(page_title="DeStudy Agent", page_icon="🇩🇪", layout="wide")
st.title("🇩🇪 DeStudy 留德硕士申请 AI Agent")

stats = db_stats()
with st.sidebar:
    engine_label = st.radio(
        "编排引擎",
        ["LangGraph (CRAG + 并行工具 + 流式)", "手写 ReAct (基础对照)"],
        index=0,
    )
    engine = "langgraph" if engine_label.startswith("LangGraph") else "react"
    rt = runtime(engine)

    st.divider()
    st.subheader("知识库概览")
    st.metric("德国英语硕士项目数", stats["total_programs"])
    c1, c2 = st.columns(2)
    c1.metric("覆盖高校", stats["distinct_universities"])
    c2.metric("CS / AI / DS", stats["cs_ai_ds_programs"])
    st.caption(
        f"免学费项目：{stats['tuition_free_programs']} · "
        f"官方深度采集：{stats['by_source'].get('curated_official', 0) + stats['by_source'].get('llm_extracted', 0)}"
    )
    st.divider()
    st.subheader("我的档案（Agent 记忆）")
    st.json(rt.agent.profile.data or {})
    st.markdown(
        "**试试问：**\n"
        "- 帮我找免学费、冬季入学的德国 Artificial Intelligence 或 Machine Learning 英语硕士\n"
        "- 我均分 85，换算成德国分是多少？中国本科申 KIT 必须交 GRE 吗？\n"
        "- 我想申 2027 冬季的 TU Dresden 和 Freiburg，帮我排时间表\n"
        "- 德国驻成都总领事馆 APS 加急通道收费多少？（测 CRAG 拒答）"
    )

tab_chat, tab_db, tab_obs = st.tabs(["💬 申请顾问对话", "🎓 德国项目库筛选", "📊 可观测性与评测"])

# ---------------------------------------------------------------- Tab 1: Chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            render_steps(m.get("steps", []), m.get("trace"))
            st.markdown(m["content"])

    if q := st.chat_input("问我任何关于德国硕士申请的问题…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            recorded_steps: list[dict] = []
            tokens_acc: list[str] = []
            status_box = st.status("正在规划与检索…", expanded=True)
            answer_slot = st.empty()

            for ev in rt.stream_chat(q):
                if ev.kind == "status":
                    recorded_steps.append({"kind": "status", "name": ev.name, "content": ev.content})
                    status_box.write(ev.content)
                elif ev.kind == "tool_call":
                    recorded_steps.append({"kind": "tool_call", "name": ev.name, "content": ev.content})
                    status_box.write(f"🔧 调用工具 `{ev.name}`")
                elif ev.kind == "tool_result":
                    recorded_steps.append({"kind": "tool_result", "name": ev.name, "content": ev.content, "duration_ms": ev.duration_ms})
                elif ev.kind == "token":
                    tokens_acc.append(ev.content)
                    answer_slot.markdown("".join(tokens_acc) + "▌")

            final_answer = "".join(tokens_acc)
            trace = rt.last_trace_dict()
            status_box.update(
                label=f"完成（{len([s for s in recorded_steps if s['kind']=='tool_call'])} 次工具调用）",
                state="complete",
                expanded=False,
            )
            answer_slot.markdown(final_answer)
            if trace:
                render_steps([], trace)

        st.session_state.messages.append({
            "role": "assistant", "content": final_answer, "steps": recorded_steps, "trace": trace,
        })

# ---------------------------------------------------------------- Tab 2: Program DB
with tab_db:
    st.subheader("德国硕士项目库（SQLite：DAAD 目录 + 官方网页 LLM 结构化抽取）")
    f1, f2, f3, f4 = st.columns(4)
    kw = f1.text_input("关键词（支持 OR / 逗号多词）", value="Computer Science")
    uni = f2.text_input("学校名称", value="")
    city = f3.text_input("城市", value="")
    intake = f4.selectbox("入学学期", ["全部", "WS", "SS"])

    f5, f6, f7 = st.columns(3)
    free_only = f5.checkbox("仅看免学费（non-EU = 0 EUR）", value=False)
    gre_opt = f6.selectbox("GRE 要求", ["全部", "no_gre", "mandatory", "optional", "recommended"])
    src_opt = f7.selectbox("数据来源", ["全部", "curated_official", "llm_extracted", "daad_catalog"])

    rows = query_programs(
        keyword=kw or None,
        university=uni or None,
        city=city or None,
        intake=None if intake == "全部" else intake,
        max_tuition_eur=0 if free_only else None,
        gre_policy=None if gre_opt == "全部" else gre_opt,
        source_type=None if src_opt == "全部" else src_opt,
        limit=100,
    )
    st.caption(f"匹配 {len(rows)} 条项目（最多显示 100 条，官方深度校验项目排在最前）")
    if rows:
        df = pd.DataFrame([
            {
                "ID": r["id"],
                "学校": r["university"],
                "项目名称": r["name"],
                "城市": r["city"],
                "入学": ",".join(r["intakes"]),
                "非欧盟学费(€/学期)": r["tuition_non_eu_eur"],
                "雅思最低": r["ielts_min"],
                "GRE": r["gre_policy"],
                "来源": r["source_type"],
                "官网": r["url"],
            }
            for r in rows
        ])
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={"官网": st.column_config.LinkColumn()})

# ---------------------------------------------------------------- Tab 3: Observability & Eval
with tab_obs:
    st.subheader("Agent 运行监控（Traces）")
    traces = load_traces(200)
    summary = summarize_traces(traces)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("总对话轮次", summary["turns"])
    m2.metric("P50 延迟", f"{summary['p50_latency_ms']/1000:.2f}s")
    m3.metric("P95 延迟", f"{summary['p95_latency_ms']/1000:.2f}s")
    m4.metric("平均 Token / 轮", f"{summary['avg_tokens']:.0f}")
    m5.metric("工具错误率", f"{summary['tool_error_rate']*100:.1f}%")

    if summary["tool_counts"]:
        st.markdown("#### 工具调用分布")
        st.bar_chart(pd.Series(summary["tool_counts"], name="calls"))

    if traces:
        st.markdown("#### 最近对话 Trace")
        for t in reversed(traces[-15:]):
            eng = t.get("engine", "react")
            header = f"{t['timestamp']} | [{eng}] {t['latency_ms']/1000:.2f}s | {t['total_tokens']} tok | tools={t.get('tool_names', [])} | {t['question'][:42]}"
            with st.expander(header):
                st.json(t)

    st.divider()
    st.subheader("离线评测报告（evaluation/results/）")
    c_ret, c_e2e, c_ag = st.columns(3)

    with c_ret:
        st.markdown("**1. 检索消融 (Retrieval)**")
        ret_data = _latest_result("retrieval")
        if ret_data:
            st.dataframe(pd.DataFrame([
                {"mode": r["mode"], "recall@5": round(r.get("recall@5", 0), 3), "MRR": round(r.get("mrr", 0), 3)}
                for r in ret_data
            ]), hide_index=True, use_container_width=True)
        else:
            st.info("尚未运行 `python -m evaluation.eval_retrieval`")

    with c_e2e:
        st.markdown("**2. 端到端生成 & 拒答 (E2E)**")
        e2e_data = _latest_result("e2e")
        if e2e_data:
            st.metric("Faithfulness", f"{e2e_data.get('avg_faithfulness', 0):.3f}")
            st.metric("Answer Relevancy", f"{e2e_data.get('avg_relevancy', 0):.3f}")
            st.metric("引用覆盖率", f"{e2e_data.get('citation_rate', 0)*100:.1f}%")
            if e2e_data.get("refusal_rate") is not None:
                st.metric("防幻觉拒答率", f"{e2e_data['refusal_rate']*100:.1f}%")
        else:
            st.info("尚未运行 `python -m evaluation.eval_e2e`")

    with c_ag:
        st.markdown("**3. Agent 评测（最新）**")
        ag_data = _latest_result("agent")
        if ag_data:
            st.caption(f"引擎：`{ag_data.get('engine', 'react')}`")
            st.metric("工具选择准确率", f"{ag_data.get('tool_accuracy', 0)*100:.1f}%")
            st.metric("任务通过率", f"{ag_data.get('pass_rate', 0)*100:.1f}%")
            st.metric("平均延迟", f"{ag_data.get('avg_latency_ms', 0)/1000:.2f}s")
            st.metric("平均 Token", f"{ag_data.get('avg_tokens', 0):.0f}")
        else:
            st.info("尚未运行 `python -m evaluation.eval_agent`")
