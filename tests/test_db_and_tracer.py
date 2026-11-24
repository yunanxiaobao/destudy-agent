"""Unit tests for SQLite programs DB, Pydantic parsing, crawler denoising, and tracing."""
from __future__ import annotations

from agent.tracer import ToolSpan, TurnTrace, append_trace, load_traces, summarize_traces
from data_pipeline.crawl import link_ok, normalize
from data_pipeline.programs_db import build_db, db_stats, query_programs


def test_crawler_normalize_and_denoise():
    assert normalize("https://www.uni-assist.de/en/how-to-apply/?pk_campaign=432") == "https://www.uni-assist.de/en/how-to-apply"
    src = {
        "allow_prefixes": ["https://www.aps.org.cn/"],
        "link_keywords": ["aps", "verfahren", "bel"],
        "exclude_keywords": ["/bel", "/osterreich"],
    }
    assert not link_ok("https://www.aps.org.cn/zh/bel", src)
    assert not link_ok("https://www.aps.org.cn/datenschutz", src)
    assert link_ok("https://www.aps.org.cn/zh/verfahren-und-services-deutschland", src)


def test_sqlite_programs_db_filters_and_precedence(tmp_path):
    db_file = tmp_path / "programs.db"
    counts = build_db(db_file)
    assert counts["curated_official"] >= 4

    stats = db_stats(db_file)
    assert stats["total_programs"] >= 4

    # Free tuition filter returns TU Dresden / TU Darmstadt (and DAAD tuition-free), excludes Freiburg (1500)
    free_cs = query_programs(keyword="Computer Science", max_tuition_eur=0, db_path=db_file, limit=50)
    ids = {p["id"] for p in free_cs}
    assert "tud-cs" in ids
    assert "freiburg-cs" not in ids

    # Mandatory GRE filter catches KIT
    gre_mand = query_programs(gre_policy="mandatory", db_path=db_file)
    assert any(p["id"] == "kit-cs" for p in gre_mand)

    # Curated official records are ranked before DAAD catalog records
    all_rows = query_programs(keyword="Computer Science", db_path=db_file, limit=10)
    assert all_rows[0]["source_type"] == "curated_official"

    # Multi-term OR keyword query works
    ai_ml = query_programs(keyword="Artificial Intelligence OR Machine Learning", max_tuition_eur=0, db_path=db_file, limit=20)
    assert len(ai_ml) >= 1


def test_tracer_roundtrip_and_summary(tmp_path):
    log = tmp_path / "traces.jsonl"
    t1 = TurnTrace(
        question="q1", answer="a1", model="test", latency_ms=1000.0,
        num_llm_calls=2, prompt_tokens=500, completion_tokens=100, total_tokens=600,
        tool_spans=[ToolSpan("convert_grade", {"score": 85}, 12.0, False, "{}")],
    )
    t2 = TurnTrace(
        question="q2", answer="a2", model="test", latency_ms=3000.0,
        num_llm_calls=3, prompt_tokens=1000, completion_tokens=200, total_tokens=1200,
        tool_spans=[
            ToolSpan("search_programs", {}, 20.0, False, "[]"),
            ToolSpan("search_knowledge", {}, 500.0, True, "err"),
        ],
    )
    append_trace(t1, log)
    append_trace(t2, log)
    loaded = load_traces(path=log)
    assert len(loaded) == 2
    s = summarize_traces(loaded)
    assert s["turns"] == 2
    assert s["avg_tokens"] == 900.0
    assert s["tool_counts"]["convert_grade"] == 1
    assert abs(s["tool_error_rate"] - (1 / 3)) < 1e-3
