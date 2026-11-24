"""DeStudy MCP Server — exposes German study-application tools over MCP.

Run (stdio):     python -m mcp_server.server
Run (HTTP):      python -m mcp_server.server --http      (streamable-http on :8765)
Inspect:         npx @modelcontextprotocol/inspector .venv/bin/python -m mcp_server.server
"""
from __future__ import annotations

import json
import sys

from mcp.server.mcpserver import MCPServer

from mcp_server import tools

mcp = MCPServer("destudy", instructions=(
    "Tools for planning applications to German universities (master's programs): "
    "knowledge-base search over official pages, SQLite database of 1,600+ English-taught "
    "German Master's programs, grade conversion, blocked-account calculation, and timeline."
))


# ------------------------------------------------------------------ tools
@mcp.tool()
def search_knowledge(query: str, university: str | None = None, top_k: int = 5) -> list[dict]:
    """Search official pages (university admission pages, DAAD, uni-assist, APS) with
    hybrid retrieval + reranking. Use for any question about requirements, deadlines,
    documents, language certificates, visa or APS. Queries in English or German work
    best; you may call it several times with different phrasings.

    Args:
        query: search query
        university: optional filter, e.g. "TU Dresden" (general DAAD/uni-assist pages are always included)
        top_k: number of passages to return (1-10)
    """
    from rag.retriever import get_retriever

    hits = get_retriever().search(query, top_k=max(1, min(top_k, 10)), university=university)
    return [h.to_dict(max_chars=1200) for h in hits]


@mcp.tool()
def convert_grade(score: float, max_score: float = 100, min_pass: float = 60) -> dict:
    """Convert a foreign grade (e.g. Chinese 0-100 average, or GPA 4.0 with min_pass 1.0)
    to the German 1.0-5.0 scale using the Modified Bavarian Formula."""
    return tools.convert_grade(score, max_score, min_pass)


@mcp.tool()
def calc_blocked_account(months: int = 12) -> dict:
    """Required amount for the German blocked account (Sperrkonto) for student visa."""
    return tools.calc_blocked_account(months)


@mcp.tool()
def search_programs(
    keyword: str | None = None,
    university: str | None = None,
    city: str | None = None,
    language: str | None = None,
    intake: str | None = None,
    max_tuition_eur: int | None = None,
    max_ielts: float | None = None,
    gre_policy: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search the SQLite database of 1,600+ German Master's programs (DAAD catalog +
    deep official records). Verified official university records are ranked first.

    Args:
        keyword: subject or title keyword, e.g. "computer science", "artificial intelligence", "data science".
                 For multiple alternatives, separate with " OR " (e.g. "Artificial Intelligence OR Machine Learning").
        university: university substring, e.g. "KIT", "TUM", "Freiburg"
        city: German city, e.g. "Munich", "Berlin", "Dresden"
        language: teaching language, "English" or "German"
        intake: "WS" (winter) or "SS" (summer)
        max_tuition_eur: max non-EU tuition per semester in EUR (use 0 for tuition-free only)
        max_ielts: max minimum IELTS score required (e.g. 6.5 or 7.0)
        gre_policy: "mandatory", "optional", "recommended", "none", or "no_gre"
        limit: max results to return (default 10, max 30)
    """
    return tools.search_programs(
        keyword=keyword,
        university=university,
        city=city,
        language=language,
        intake=intake,
        max_tuition_eur=max_tuition_eur,
        max_ielts=max_ielts,
        gre_policy=gre_policy,
        limit=max(1, min(limit, 30)),
    )


@mcp.tool()
def get_program_stats() -> dict:
    """Return summary statistics of the German Master's program SQLite database
    (total programs, universities, tuition-free count, CS/AI/DS count)."""
    return tools.get_program_stats()


@mcp.tool()
def build_timeline(
    intake: str,
    program_ids: list[str] | None = None,
    has_aps: bool = False,
    has_language_score: bool = False,
) -> dict:
    """Build a backwards-planned application timeline (APS, language test, documents,
    deadlines, blocked account, visa, housing).

    Args:
        intake: target semester, e.g. "WS2027" or "SS2027"
        program_ids: ids from search_programs, used to take real deadlines into account
        has_aps: whether the student already holds the APS certificate
        has_language_score: whether the student already has the required language score
    """
    return tools.build_timeline(intake, program_ids, has_aps, has_language_score)


# -------------------------------------------------------------- resources
@mcp.resource("destudy://programs")
def programs_resource() -> str:
    """Top curated and extracted programs in the database as JSON."""
    return json.dumps(tools.load_programs(limit=50), ensure_ascii=False, indent=1, default=str)


@mcp.resource("destudy://programs/{program_id}")
def program_resource(program_id: str) -> str:
    """One program by id."""
    return json.dumps(tools.get_program(program_id), ensure_ascii=False, indent=1, default=str)


# ---------------------------------------------------------------- prompts
@mcp.prompt()
def application_checklist(program_id: str) -> str:
    """Generate a personalised document checklist for one program."""
    p = tools.get_program(program_id)
    return (
        "请根据以下项目信息，并调用 search_knowledge 查询官方页面进行核实，为我生成一份申请材料清单"
        "（材料名称 / 要求 / 是否需要公证或翻译 / 来源链接）：\n\n"
        + json.dumps(p, ensure_ascii=False, indent=1, default=str)
    )


@mcp.prompt()
def motivation_letter_review(program_id: str, letter: str) -> str:
    """Review a motivation letter against a program's profile."""
    p = tools.get_program(program_id)
    return (
        "You are an admissions reviewer for the following program. Review the motivation letter: "
        "structure, fit with program focus, concrete evidence, and language. Give a score 1-10 and "
        f"specific edits.\n\nProgram:\n{json.dumps(p, ensure_ascii=False, default=str)}\n\nLetter:\n{letter}"
    )


if __name__ == "__main__":
    if "--http" in sys.argv:
        import os

        host = os.getenv("MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_PORT", "8765"))
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run(transport="stdio")
