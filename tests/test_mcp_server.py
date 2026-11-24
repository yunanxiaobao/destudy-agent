"""Integration test: spawn the MCP server over stdio and exercise tools, resources, prompts."""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from settings import ROOT


async def _exercise_mcp():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            tool_names = {t.name for t in tools}
            assert {
                "search_knowledge",
                "convert_grade",
                "calc_blocked_account",
                "search_programs",
                "build_timeline",
            }.issubset(tool_names)

            resources = (await session.list_resources()).resources
            assert any(str(r.uri) == "destudy://programs" for r in resources)

            prompts = (await session.list_prompts()).prompts
            prompt_names = {p.name for p in prompts}
            assert {"application_checklist", "motivation_letter_review"}.issubset(prompt_names)

            # Call convert_grade
            res = await session.call_tool("convert_grade", {"score": 85, "max_score": 100, "min_pass": 60})
            assert not res.is_error
            payload = res.structured_content or json.loads(res.content[0].text)
            assert payload["german_grade"] == 2.1

            # Call calc_blocked_account
            res2 = await session.call_tool("calc_blocked_account", {"months": 12})
            assert not res2.is_error
            payload2 = res2.structured_content or json.loads(res2.content[0].text)
            assert payload2["total_eur"] == payload2["monthly_eur"] * 12

            # Call build_timeline
            res3 = await session.call_tool("build_timeline", {"intake": "WS2027"})
            assert not res3.is_error
            payload3 = res3.structured_content or json.loads(res3.content[0].text)
            assert payload3["semester_start"] == "2027-10-01"

            # Call search_programs
            res4 = await session.call_tool("search_programs", {"university": "KIT", "keyword": "Computer Science"})
            assert not res4.is_error
            payload4 = res4.structured_content or json.loads(res4.content[0].text)
            items = payload4.get("result", payload4) if isinstance(payload4, dict) else payload4
            assert items and items[0]["id"] == "kit-cs"

            # Read resource & get prompt
            res_read = await session.read_resource("destudy://programs")
            assert "tuda-cs" in res_read.contents[0].text
            prompt_res = await session.get_prompt("application_checklist", {"program_id": "freiburg-cs"})
            assert prompt_res.messages and "Freiburg" in prompt_res.messages[0].content.text

            # Verify input_schema exists on discovered tools
            for t in tools:
                assert isinstance(t.input_schema, dict)
                assert t.input_schema.get("type") == "object"


def test_mcp_stdio_roundtrip():
    asyncio.run(_exercise_mcp())
