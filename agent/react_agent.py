"""Hand-written tool-calling agent (no framework), with built-in turn tracing.

The agent is an MCP *client*: it spawns the destudy MCP server over stdio,
discovers its tools, converts them to OpenAI function schemas, and runs a
Reason -> Act -> Observe loop until the model produces a final answer.
It also owns one local tool (update_user_profile) for long-term memory.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

from agent.memory import Profile
from agent.tracer import ToolSpan, TurnTrace, append_trace
from settings import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, ROOT, load_prompt

MAX_STEPS = 8
MAX_TOOL_CHARS = 12000

PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_user_profile",
        "description": "Save facts the user tells you about themselves (major, GPA/grade scale, "
                       "language scores, APS status, target intake, target programs, budget...). "
                       "Call this whenever the user reveals new personal info.",
        "parameters": {
            "type": "object",
            "properties": {"updates": {"type": "object", "description": "key/value pairs to store", "additionalProperties": True}},
            "required": ["updates"],
        },
    },
}


@dataclass
class Step:
    kind: str  # "tool_call" | "tool_result" | "answer"
    name: str = ""
    content: Any = None
    duration_ms: float | None = None


@dataclass
class DeStudyAgent:
    model: str = OPENAI_MODEL
    profile: Profile = field(default_factory=Profile.load)
    history: list[dict] = field(default_factory=list)
    on_step: Callable[[Step], None] | None = None
    record_trace: bool = True
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
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    def _emit(self, step: Step):
        if self.on_step:
            self.on_step(step)

    def _system(self) -> str:
        return load_prompt("agent_system").replace("{profile}", self.profile.render())

    async def _call_tool(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "update_user_profile":
            self.profile.update(args.get("updates", {}))
            return json.dumps({"ok": True, "profile": self.profile.data}, ensure_ascii=False), False
        try:
            res = await self.session.call_tool(name, args)
        except Exception as e:  # surface tool errors to the model so it can recover
            return json.dumps({"error": str(e)}), True
        if res.structured_content is not None:
            text = json.dumps(res.structured_content, ensure_ascii=False, default=str)
        else:
            text = "\n".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return json.dumps({"error": text}), True
        return text[:MAX_TOOL_CHARS], False

    async def chat(self, user_msg: str) -> str:
        t0 = time.perf_counter()
        self.history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": self._system()}] + self.history[-20:]

        prompt_tok = comp_tok = total_tok = llm_calls = 0
        spans: list[ToolSpan] = []
        answer = "抱歉，这个问题需要的步骤太多，我没能在限定步数内完成。可以把问题拆小一点再问我。"

        for _ in range(MAX_STEPS):
            resp = await self.llm.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools, temperature=0.2,
            )
            llm_calls += 1
            if resp.usage:
                prompt_tok += getattr(resp.usage, "prompt_tokens", 0) or 0
                comp_tok += getattr(resp.usage, "completion_tokens", 0) or 0
                total_tok += getattr(resp.usage, "total_tokens", 0) or 0

            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = msg.content or ""
                self._emit(Step("answer", content=answer))
                break

            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                self._emit(Step("tool_call", tc.function.name, args))
                ts = time.perf_counter()
                result, is_err = await self._call_tool(tc.function.name, args)
                dur = round((time.perf_counter() - ts) * 1000, 1)
                spans.append(ToolSpan(tc.function.name, args, dur, is_err, result[:300]))
                self._emit(Step("tool_result", tc.function.name, result, duration_ms=dur))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        self.history.append({"role": "assistant", "content": answer})
        self.last_trace = TurnTrace(
            question=user_msg,
            answer=answer,
            model=self.model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            num_llm_calls=llm_calls,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=total_tok or (prompt_tok + comp_tok),
            tool_spans=spans,
        )
        if self.record_trace:
            append_trace(self.last_trace)
        return answer
