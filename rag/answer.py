"""End-to-end RAG answer with numbered citations.

Usage: python -m rag.answer "TU Dresden 计算机硕士申请截止日期是什么时候？"
"""
from __future__ import annotations

import sys

from rag.models import get_llm
from rag.query_rewrite import rewrite_queries
from rag.retriever import Hit, get_retriever
from settings import OPENAI_MODEL, load_prompt


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(
            f"[{i}] {c.get('university') or 'General'} | {c['title']} | {c.get('section', '')}\n"
            f"URL: {c['url']} (crawled {c.get('crawled_at', '')[:10]})\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(question: str, university: str | None = None, top_k: int = 6, rewrite: bool = True) -> dict:
    extra = rewrite_queries(question) if rewrite else []
    hits = get_retriever().search(question, extra_queries=extra, top_k=top_k, university=university)
    context = format_context(hits)
    resp = get_llm().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": load_prompt("rag_answer")},
            {"role": "user", "content": f"# Sources\n\n{context}\n\n# Question\n{question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "queries": [question] + extra,
        "context": context,
        "sources": [h.to_dict() for h in hits],
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "TU Dresden 计算机硕士的申请截止日期是什么时候？"
    out = answer(q)
    print("Queries:", out["queries"], "\n")
    print(out["answer"], "\n")
    for i, s in enumerate(out["sources"], 1):
        print(f"[{i}] {s['score']:.3f} {s['title']} — {s['url']}")
