"""Unit tests for LangGraph StateGraph: CRAG self-correction, abstain guardrail,
parallel tool execution, streaming events, and Mermaid graph export (no network)."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

from agent.langgraph_agent import LangGraphDeStudyAgent, grade_crag_decision
from agent.memory import Profile


def test_grade_crag_decision_thresholds():
    assert grade_crag_decision(0.85, rewrites=0, threshold=0.30) == "synthesize"
    assert grade_crag_decision(0.15, rewrites=0, threshold=0.30) == "rewrite"
    assert grade_crag_decision(0.15, rewrites=1, threshold=0.30) == "abstain"


class _FakeStream:
    def __init__(self, tokens: list[str] | None = None, tool_calls: list[dict] | None = None):
        self._tokens = tokens or []
        self._tcs = tool_calls or []

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for tok in self._tokens:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=tok, tool_calls=None))],
                usage=None,
            )
        for i, tc in enumerate(self._tcs):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=i,
                                    id=tc["id"],
                                    function=SimpleNamespace(name=tc["name"], arguments=json.dumps(tc["args"])),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )


def _make_stub_agent(create_side_effect, tool_side_effect) -> LangGraphDeStudyAgent:
    agent = LangGraphDeStudyAgent(profile=Profile(), record_trace=False)
    agent.tools = []
    agent._event_q = None
    agent._usage = {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0}
    agent._spans = []
    agent.llm = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_side_effect))
    )
    agent.session = SimpleNamespace(call_tool=tool_side_effect)
    agent.graph = agent._build_graph()
    return agent


def test_langgraph_crag_rewrite_recovers_and_streams():
    """Low initial reranker score -> triggers EN/DE query rewrite -> second retrieve succeeds -> streams answer."""
    call_idx = {"llm": 0, "search": 0}

    async def fake_create(**kwargs):
        call_idx["llm"] += 1
        if not kwargs.get("stream"):
            if call_idx["llm"] == 1:  # route_intent
                content = json.dumps({"route": "crag", "university": "KIT", "search_query": "KIT fees"})
            else:  # crag_rewrite
                content = json.dumps({"queries": ["KIT tuition fees non-EU", "KIT Studiengebühren"]})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=15, total_tokens=65),
            )
        return _FakeStream(tokens=["KIT 非欧盟学费为 ", "1,500 欧元 [1]。"])

    async def fake_tool(name, args):
        call_idx["search"] += 1
        score = 0.12 if call_idx["search"] == 1 else 0.91
        hits = [{"chunk_id": f"c{call_idx['search']}", "title": "KIT Fees", "url": "https://kit.edu", "score": score, "text": "1500 EUR"}]
        return SimpleNamespace(structured_content={"result": hits}, content=[], is_error=False)

    agent = _make_stub_agent(fake_create, fake_tool)

    async def _run():
        events = [ev async for ev in agent.chat_stream("KIT 非欧盟学费多少？")]
        return events

    events = asyncio.run(_run())
    token_text = "".join(ev.content for ev in events if ev.kind == "token")
    assert "1,500 欧元" in token_text
    assert agent.last_trace.engine == "langgraph"
    assert agent.last_trace.meta["route"] == "crag"
    assert agent.last_trace.meta["crag_rewrites"] == 1
    assert agent.last_trace.meta["low_confidence"] is False
    assert "crag_rewrite" in agent.mermaid()


def test_langgraph_crag_abstain_when_persistent_low_confidence():
    """Low score both before AND after rewrite -> sets low_confidence=True guardrail."""
    seen_system_prompts = []

    async def fake_create(**kwargs):
        if not kwargs.get("stream"):
            sys_msg = kwargs["messages"][0]["content"]
            if "route" in sys_msg:
                content = json.dumps({"route": "crag", "university": None, "search_query": "Chengdu APS express fee"})
            else:
                content = json.dumps({"queries": ["APS Chengdu express", "APS Eilverfahren"]})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=40, completion_tokens=10, total_tokens=50),
            )
        seen_system_prompts.append(kwargs["messages"][0]["content"])
        return _FakeStream(tokens=["当前知识库中未查到相关信息。"])

    async def fake_tool(name, args):
        hits = [{"chunk_id": "low1", "title": "Other", "url": "https://aps.org.cn", "score": 0.08, "text": "unrelated"}]
        return SimpleNamespace(structured_content={"result": hits}, content=[], is_error=False)

    agent = _make_stub_agent(fake_create, fake_tool)
    ans = asyncio.run(agent.chat("成都 APS 加急费用多少？"))
    assert "未查到" in ans
    assert agent.last_trace.meta["crag_rewrites"] == 1
    assert agent.last_trace.meta["low_confidence"] is True
    assert any("LOW CONFIDENCE ALERT" in p for p in seen_system_prompts)


def test_langgraph_parallel_tools_execute_concurrently():
    """Two tool calls emitted in one turn run concurrently via asyncio.gather."""
    step = {"n": 0}

    async def fake_create(**kwargs):
        step["n"] += 1
        if not kwargs.get("stream"):
            content = json.dumps({"route": "agent", "university": None, "search_query": ""})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=40),
            )
        if step["n"] == 2:
            return _FakeStream(tool_calls=[
                {"id": "tc1", "name": "convert_grade", "args": {"score": 85}},
                {"id": "tc2", "name": "calc_blocked_account", "args": {"months": 12}},
            ])
        return _FakeStream(tokens=["德国分 2.1，自保金 11,904 欧元。"])

    async def fake_tool(name, args):
        await asyncio.sleep(0.15)
        return SimpleNamespace(structured_content={"tool": name, "ok": True}, content=[], is_error=False)

    agent = _make_stub_agent(fake_create, fake_tool)
    t0 = time.perf_counter()
    ans = asyncio.run(agent.chat("算下 85 分德国分和 12 个月自保金"))
    elapsed = time.perf_counter() - t0
    assert "2.1" in ans
    assert set(agent.last_trace.tool_names) == {"convert_grade", "calc_blocked_account"}
    # Sequential would be >= 0.30s; concurrent asyncio.gather finishes around ~0.15s
    assert elapsed < 0.28
