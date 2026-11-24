"""LangGraph StateGraph agent with:
  1. Intent Routing (`crag` single-policy lookup vs `agent` multi-step tools)
  2. Corrective RAG (CRAG) loop: retrieve -> grade reranker confidence ->
     auto EN/DE query rewrite & retry -> abstain guardrail on persistent low confidence
  3. Parallel MCP tool execution (`asyncio.gather`)
  4. Native OpenAI token streaming (`chat_stream`) & unified TurnTrace logging

Compatible drop-in interface alongside hand-written `agent.react_agent.DeStudyAgent`.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from agent.memory import Profile
from agent.react_agent import MAX_STEPS, MAX_TOOL_CHARS, PROFILE_TOOL, Step
from agent.tracer import ToolSpan, TurnTrace, append_trace
from settings import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, ROOT, load_prompt

# bge-reranker-v2-m3 outputs sigmoid-like probabilities in [0, 1] (or raw logits);
# empirically relevant hits score >= 0.45 while off-topic hits score < 0.25.
CRAG_SCORE_THRESHOLD = 0.30
MAX_CRAG_REWRITES = 1


@dataclass
class StreamEvent:
    kind: str  # "status" | "tool_call" | "tool_result" | "token" | "done"
    name: str = ""
    content: Any = None
    duration_ms: float | None = None


class GraphState(TypedDict, total=False):
    question: str
    route: str  # "crag" | "agent"
    university: str | None
    pending_queries: list[str]
    passages: list[dict]
    top_score: float
    rewrites: int
    low_confidence: bool
    messages: list[dict]
    pending_tool_calls: list[dict]
    steps: int
    answer: str


def grade_crag_decision(top_score: float, rewrites: int, threshold: float = CRAG_SCORE_THRESHOLD, max_rewrites: int = MAX_CRAG_REWRITES) -> str:
    """Pure routing logic for CRAG confidence grading (unit-tested directly)."""
    if top_score >= threshold:
        return "synthesize"
    if rewrites < max_rewrites:
        return "rewrite"
    return "abstain"


@dataclass
class LangGraphDeStudyAgent:
    model: str = OPENAI_MODEL
    profile: Profile = field(default_factory=Profile.load)
    history: list[dict] = field(default_factory=list)
    on_step: Callable[[Step], None] | None = None
    record_trace: bool = True
    crag_threshold: float = CRAG_SCORE_THRESHOLD
    last_trace: TurnTrace | None = field(default=None, init=False)

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT))
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        mcp_tools = (await self.session.list_tools()).tools
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
            for t in mcp_tools
        ] + [PROFILE_TOOL]
        self.llm = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self._event_q: asyncio.Queue[StreamEvent] | None = None
        self._usage = {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0}
        self._spans: list[ToolSpan] = []
        self.graph = self._build_graph()
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def _emit_event(self, ev: StreamEvent) -> None:
        if self._event_q is not None:
            self._event_q.put_nowait(ev)
        if self.on_step and ev.kind in ("tool_call", "tool_result", "done"):
            kind = "answer" if ev.kind == "done" else ev.kind
            self.on_step(Step(kind=kind, name=ev.name, content=ev.content, duration_ms=ev.duration_ms))

    def _add_usage(self, usage: Any) -> None:
        self._usage["llm_calls"] += 1
        if usage:
            p = getattr(usage, "prompt_tokens", 0) or 0
            c = getattr(usage, "completion_tokens", 0) or 0
            t = getattr(usage, "total_tokens", 0) or (p + c)
            self._usage["prompt"] += p
            self._usage["completion"] += c
            self._usage["total"] += t

    def _system(self) -> str:
        return load_prompt("agent_system").replace("{profile}", self.profile.render())

    async def _call_tool(self, name: str, args: dict) -> tuple[str, bool, float]:
        self._emit_event(StreamEvent("tool_call", name=name, content=args))
        t0 = time.perf_counter()
        if name == "update_user_profile":
            self.profile.update(args.get("updates", {}))
            text, is_err = json.dumps({"ok": True, "profile": self.profile.data}, ensure_ascii=False), False
        else:
            try:
                res = await self.session.call_tool(name, args)
                if res.structured_content is not None:
                    text = json.dumps(res.structured_content, ensure_ascii=False, default=str)
                else:
                    text = "\n".join(getattr(c, "text", "") for c in res.content)
                is_err = bool(res.is_error)
                if is_err:
                    text = json.dumps({"error": text}, ensure_ascii=False)
            except Exception as e:
                text, is_err = json.dumps({"error": str(e)}, ensure_ascii=False), True
        dur = round((time.perf_counter() - t0) * 1000, 1)
        text = text[:MAX_TOOL_CHARS]
        self._spans.append(ToolSpan(name, args, dur, is_err, text[:300]))
        self._emit_event(StreamEvent("tool_result", name=name, content=text, duration_ms=dur))
        return text, is_err, dur

    async def _stream_completion(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2) -> tuple[str, list[dict]]:
        """Stream an OpenAI completion, forwarding text deltas live and assembling any tool calls."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self.llm.chat.completions.create(**kwargs)
        content_parts: list[str] = []
        tc_acc: dict[int, dict[str, Any]] = {}
        usage = None

        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                self._emit_event(StreamEvent("token", content=delta.content))
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tc_acc.setdefault(
                        tc.index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments

        self._add_usage(usage)
        ordered_tcs = [tc_acc[i] for i in sorted(tc_acc)]
        return "".join(content_parts), ordered_tcs

    # ------------------------------------------------------------------ Graph nodes
    async def _node_route(self, state: GraphState) -> GraphState:
        self._emit_event(StreamEvent("status", name="route_intent", content="🧭 识别问题意图与执行路径…"))
        prev_turns = self.history[:-1][-2:]
        user_payload = state["question"]
        if prev_turns:
            ctx = "\n".join(f"{m['role']}: {m['content'][:160]}" for m in prev_turns)
            user_payload = f"Previous conversation:\n{ctx}\n\nCurrent question:\n{state['question']}"
        resp = await self.llm.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": load_prompt("router")},
                {"role": "user", "content": user_payload},
            ],
        )
        self._add_usage(resp.usage)
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            data = {}
        route = data.get("route") if data.get("route") in ("crag", "agent") else "agent"
        uni = data.get("university") or None
        sq = (data.get("search_query") or state["question"]).strip()
        self._emit_event(
            StreamEvent(
                "status",
                name="route_intent",
                content=f"🧭 路由结果：`{route}`" + (f"（院校过滤：{uni}）" if uni else ""),
            )
        )
        if route == "crag":
            return {
                "route": "crag",
                "university": uni,
                "pending_queries": [sq],
                "passages": [],
                "top_score": 0.0,
                "rewrites": 0,
                "low_confidence": False,
            }
        messages = [{"role": "system", "content": self._system()}] + self.history[-20:]
        return {"route": "agent", "messages": messages, "pending_tool_calls": [], "steps": 0}

    async def _node_crag_retrieve(self, state: GraphState) -> GraphState:
        queries = state.get("pending_queries") or [state["question"]]
        uni = state.get("university")
        self._emit_event(
            StreamEvent("status", name="crag_retrieve", content=f"🔍 CRAG 检索官方知识库（{len(queries)} 条查询）…")
        )
        # Execute search_knowledge calls in parallel via MCP
        calls = [
            self._call_tool("search_knowledge", {"query": q, "university": uni, "top_k": 5})
            for q in queries
        ]
        results = await asyncio.gather(*calls)

        by_cid: dict[str, dict] = {p["chunk_id"]: p for p in state.get("passages", [])}
        for raw, is_err, _ in results:
            if is_err:
                continue
            try:
                parsed = json.loads(raw)
                items = parsed.get("result", parsed) if isinstance(parsed, dict) else parsed
                if isinstance(items, list):
                    for h in items:
                        cid = h.get("chunk_id")
                        if cid and (cid not in by_cid or h.get("score", 0) > by_cid[cid].get("score", 0)):
                            by_cid[cid] = h
            except json.JSONDecodeError:
                continue

        merged = sorted(by_cid.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:6]
        top_score = float(merged[0]["score"]) if merged else 0.0
        self._emit_event(
            StreamEvent(
                "status",
                name="crag_grade",
                content=f"⚖️ 证据置信度评估：top_score = {top_score:.3f}（阈值 {self.crag_threshold:.2f}）",
            )
        )
        return {"passages": merged, "top_score": top_score}

    async def _node_crag_rewrite(self, state: GraphState) -> GraphState:
        self._emit_event(
            StreamEvent("status", name="crag_rewrite", content="🔄 检索置信度偏低，自动触发中→德/英官方术语改写重试…")
        )
        resp = await self.llm.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": load_prompt("query_rewrite").replace("{n}", "2")},
                {"role": "user", "content": state["question"]},
            ],
        )
        self._add_usage(resp.usage)
        try:
            queries = [q for q in json.loads(resp.choices[0].message.content or "{}").get("queries", []) if isinstance(q, str) and q.strip()][:2]
        except json.JSONDecodeError:
            queries = []
        # Also drop university filter on retry in case router guessed a narrow filter
        return {
            "pending_queries": queries or [state["question"]],
            "university": None,
            "rewrites": state.get("rewrites", 0) + 1,
        }

    async def _node_crag_abstain_flag(self, state: GraphState) -> GraphState:
        self._emit_event(
            StreamEvent("status", name="crag_abstain", content="🛡️ 改写重试后置信度仍低，启用严格防幻觉拒答保护")
        )
        return {"low_confidence": True}

    async def _node_crag_synthesize(self, state: GraphState) -> GraphState:
        self._emit_event(StreamEvent("status", name="crag_synthesize", content="✍️ 基于官方证据流式生成回答…"))
        passages = state.get("passages", [])
        blocks = []
        for i, p in enumerate(passages, 1):
            blocks.append(
                f"[{i}] {p.get('university') or 'General'} | {p.get('title', '')} | {p.get('section', '')}\n"
                f"URL: {p.get('url', '')} (crawled {str(p.get('crawled_at', ''))[:10]})\n{p.get('text', '')}"
            )
        context = "\n\n---\n\n".join(blocks) if blocks else "(no relevant passages retrieved)"
        extra_guard = ""
        if state.get("low_confidence"):
            extra_guard = (
                "\n\n[LOW CONFIDENCE ALERT]: Retrieval confidence remained below threshold even after "
                "query rewriting. The official sources very likely do NOT contain the answer to this "
                "question. Explicitly state '当前知识库中未查到相关信息' unless a passage explicitly "
                "and unambiguously answers the exact question. Never extrapolate or guess."
            )
        messages = [
            {"role": "system", "content": load_prompt("rag_answer") + extra_guard},
            {"role": "user", "content": f"# Sources\n\n{context}\n\n# Question\n{state['question']}"},
        ]
        answer, _ = await self._stream_completion(messages, tools=None, temperature=0.1)
        return {"answer": answer}

    async def _node_agent_step(self, state: GraphState) -> GraphState:
        step_num = state.get("steps", 0)
        if step_num >= MAX_STEPS:
            fallback = "抱歉，这个问题需要的步骤太多，我没能在限定步数内完成。可以把问题拆小一点再问我。"
            self._emit_event(StreamEvent("token", content=fallback))
            return {"answer": fallback, "pending_tool_calls": []}

        self._emit_event(StreamEvent("status", name="agent_step", content=f"🧠 Agent 规划第 {step_num + 1} 步…"))
        content, tool_calls = await self._stream_completion(state["messages"], tools=self.tools, temperature=0.2)
        if not tool_calls:
            return {"answer": content, "pending_tool_calls": [], "steps": step_num + 1}

        assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            assistant_msg["content"] = content
        return {
            "messages": state["messages"] + [assistant_msg],
            "pending_tool_calls": tool_calls,
            "steps": step_num + 1,
        }

    async def _node_parallel_tools(self, state: GraphState) -> GraphState:
        tcs = state.get("pending_tool_calls", [])
        if len(tcs) > 1:
            self._emit_event(
                StreamEvent("status", name="parallel_tools", content=f"⚡ 并行执行 {len(tcs)} 个 MCP 工具…")
            )

        async def _run_one(tc: dict) -> dict:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result, _, _ = await self._call_tool(name, args)
            return {"role": "tool", "tool_call_id": tc["id"], "content": result}

        tool_msgs = await asyncio.gather(*(_run_one(tc) for tc in tcs))
        return {"messages": state["messages"] + list(tool_msgs), "pending_tool_calls": []}

    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("route_intent", self._node_route)
        g.add_node("crag_retrieve", self._node_crag_retrieve)
        g.add_node("crag_rewrite", self._node_crag_rewrite)
        g.add_node("crag_abstain_flag", self._node_crag_abstain_flag)
        g.add_node("crag_synthesize", self._node_crag_synthesize)
        g.add_node("agent_step", self._node_agent_step)
        g.add_node("parallel_tools", self._node_parallel_tools)

        g.add_edge(START, "route_intent")
        g.add_conditional_edges(
            "route_intent",
            lambda s: s["route"],
            {"crag": "crag_retrieve", "agent": "agent_step"},
        )
        g.add_conditional_edges(
            "crag_retrieve",
            lambda s: grade_crag_decision(s.get("top_score", 0.0), s.get("rewrites", 0), self.crag_threshold),
            {
                "synthesize": "crag_synthesize",
                "rewrite": "crag_rewrite",
                "abstain": "crag_abstain_flag",
            },
        )
        g.add_edge("crag_rewrite", "crag_retrieve")
        g.add_edge("crag_abstain_flag", "crag_synthesize")
        g.add_edge("crag_synthesize", END)

        g.add_conditional_edges(
            "agent_step",
            lambda s: "tools" if s.get("pending_tool_calls") else "end",
            {"tools": "parallel_tools", "end": END},
        )
        g.add_edge("parallel_tools", "agent_step")
        return g.compile()

    # ------------------------------------------------------------------ Public API
    async def chat(self, user_msg: str) -> str:
        t0 = time.perf_counter()
        self.history.append({"role": "user", "content": user_msg})
        self._usage = {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0}
        self._spans = []

        final_state = await self.graph.ainvoke({"question": user_msg})
        answer = final_state.get("answer") or ""
        self.history.append({"role": "assistant", "content": answer})

        self.last_trace = TurnTrace(
            question=user_msg,
            answer=answer,
            model=self.model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            num_llm_calls=self._usage["llm_calls"],
            prompt_tokens=self._usage["prompt"],
            completion_tokens=self._usage["completion"],
            total_tokens=self._usage["total"],
            tool_spans=list(self._spans),
            engine="langgraph",
            meta={
                "route": final_state.get("route", "agent"),
                "crag_rewrites": final_state.get("rewrites", 0),
                "crag_top_score": round(float(final_state.get("top_score", 0.0)), 4),
                "low_confidence": bool(final_state.get("low_confidence", False)),
            },
        )
        if self.record_trace:
            append_trace(self.last_trace)
        self._emit_event(StreamEvent("done", content=answer))
        return answer

    async def chat_stream(self, user_msg: str) -> AsyncIterator[StreamEvent]:
        """Async generator yielding live status, tool calls, token chunks, and final done event."""
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._event_q = q

        async def _runner():
            try:
                await self.chat(user_msg)
            finally:
                await q.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield ev
        finally:
            self._event_q = None
            await task
