"""Multi-query rewriting: turn a (possibly Chinese) user question into
English + German search queries so both BM25 and dense retrieval can hit
official German/English pages."""
from __future__ import annotations

import json

from rag.models import get_llm
from settings import OPENAI_MODEL, load_prompt


def rewrite_queries(question: str, n: int = 3) -> list[str]:
    resp = get_llm().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": load_prompt("query_rewrite").replace("{n}", str(n))},
            {"role": "user", "content": question},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:n]
    except (json.JSONDecodeError, AttributeError):
        return []
