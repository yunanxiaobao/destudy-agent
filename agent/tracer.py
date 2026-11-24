"""Lightweight local observability & tracing for DeStudyAgent.

Logs every agent turn (latency, token usage, per-tool duration and errors) to
data/traces.jsonl so Streamlit's monitoring tab and offline analysis can inspect
cost, latency percentiles (P50/P95), and tool usage patterns.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from settings import TRACES_PATH


@dataclass
class ToolSpan:
    name: str
    args: dict[str, Any]
    duration_ms: float
    is_error: bool = False
    result_preview: str = ""


@dataclass
class TurnTrace:
    question: str
    answer: str
    model: str
    latency_ms: float
    num_llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_spans: list[ToolSpan] = field(default_factory=list)
    engine: str = "react"
    meta: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self.tool_spans]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_names"] = self.tool_names
        return d


def append_trace(trace: TurnTrace, path: Path = TRACES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def load_traces(limit: int = 200, path: Path = TRACES_PATH) -> list[dict]:
    if not path.exists():
        return []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [json.loads(l) for l in lines[-limit:]]


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return s[idx]


def summarize_traces(traces: list[dict]) -> dict:
    if not traces:
        return {
            "turns": 0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
            "avg_tokens": 0.0, "total_tokens": 0, "avg_tools": 0.0,
            "tool_counts": {}, "tool_error_rate": 0.0,
        }
    latencies = [float(t.get("latency_ms", 0)) for t in traces]
    tokens = [int(t.get("total_tokens", 0)) for t in traces]
    spans = [s for t in traces for s in t.get("tool_spans", [])]
    errors = sum(1 for s in spans if s.get("is_error"))
    counts = Counter(s.get("name", "unknown") for s in spans)
    return {
        "turns": len(traces),
        "p50_latency_ms": round(_percentile(latencies, 0.50), 1),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 1),
        "avg_tokens": round(sum(tokens) / len(tokens), 1),
        "total_tokens": sum(tokens),
        "avg_tools": round(len(spans) / len(traces), 2),
        "tool_counts": dict(counts.most_common()),
        "tool_error_rate": round(errors / len(spans), 4) if spans else 0.0,
    }
