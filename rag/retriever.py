"""Hybrid retriever: dense (bge-m3 / Qdrant) + sparse (BM25), fused with RRF,
then re-ranked by a multilingual cross-encoder.

Modes (used by the eval ablation):
    dense | bm25 | hybrid | hybrid_rerank
"""
from __future__ import annotations

import atexit
import json
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from rag.models import get_embedder, get_reranker
from settings import CHUNKS_PATH, COLLECTION, QDRANT_PATH

TOKEN_RE = re.compile(r"[a-zA-Z0-9äöüßÄÖÜ]+|[\u4e00-\u9fff]")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    # CJK: add character bigrams so Chinese queries still get partial BM25 matches
    cjk = [t for t in toks if "\u4e00" <= t <= "\u9fff"]
    return toks + [a + b for a, b in zip(cjk, cjk[1:])]


@dataclass
class Hit:
    chunk: dict
    score: float
    sources: list[str] = field(default_factory=list)  # which retrievers found it

    def to_dict(self, max_chars: int | None = None) -> dict:
        c = self.chunk
        text = c["text"] if not max_chars else c["text"][:max_chars]
        return {
            "chunk_id": c["chunk_id"], "title": c["title"], "section": c.get("section", ""),
            "url": c["url"], "university": c.get("university", ""), "lang": c.get("lang", ""),
            "crawled_at": c.get("crawled_at", ""), "score": round(float(self.score), 4), "text": text,
        }


class HybridRetriever:
    def __init__(self):
        self.chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in self.chunks])
        self._qdrant = None

    @property
    def qdrant(self):
        if self._qdrant is None:
            from qdrant_client import QdrantClient

            self._qdrant = QdrantClient(path=str(QDRANT_PATH))
            atexit.register(self.close)
        return self._qdrant

    def close(self):
        if self._qdrant is not None:
            try:
                self._qdrant.close()
            except Exception:
                pass
            self._qdrant = None

    # ---------- single retrievers ----------
    def dense(self, query: str, limit: int, university: str | None = None) -> list[str]:
        from qdrant_client import models

        vec = get_embedder().encode([query], normalize_embeddings=True)[0].tolist()
        qfilter = None
        if university:
            qfilter = models.Filter(should=[
                models.FieldCondition(key="category", match=models.MatchValue(value="general")),
                models.FieldCondition(key="university", match=models.MatchText(text=university)),
            ])
        res = self.qdrant.query_points(
            COLLECTION, query=vec, query_filter=qfilter, limit=limit, with_payload=["chunk_id"]
        )
        return [p.payload["chunk_id"] for p in res.points]

    def sparse(self, query: str, limit: int, university: str | None = None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        out = []
        for i in scores.argsort()[::-1]:
            if scores[i] <= 0:
                break
            cid = self.chunks[i]["chunk_id"]
            if self._filter(cid, university):
                out.append(cid)
                if len(out) >= limit:
                    break
        return out

    # ---------- fusion ----------
    @staticmethod
    def rrf(ranked_lists: dict[str, list[str]]) -> dict[str, tuple[float, list[str]]]:
        fused: dict[str, tuple[float, list[str]]] = {}
        for name, ids in ranked_lists.items():
            for rank, cid in enumerate(ids):
                s, src = fused.get(cid, (0.0, []))
                fused[cid] = (s + 1.0 / (RRF_K + rank + 1), src + [name])
        return fused

    def _filter(self, cid: str, university: str | None) -> bool:
        if not university:
            return True
        c = self.by_id[cid]
        return c["category"] == "general" or university.lower() in c.get("university", "").lower()

    def search(
        self,
        query: str,
        extra_queries: list[str] | None = None,
        top_k: int = 5,
        mode: str = "hybrid_rerank",
        university: str | None = None,
        candidates: int = 30,
    ) -> list[Hit]:
        queries = [query] + list(extra_queries or [])
        lists: dict[str, list[str]] = {}
        for i, q in enumerate(queries):
            if mode in ("dense", "hybrid", "hybrid_rerank"):
                lists[f"dense{i}"] = self.dense(q, candidates, university=university)
            if mode in ("bm25", "hybrid", "hybrid_rerank"):
                lists[f"bm25{i}"] = self.sparse(q, candidates, university=university)

        fused = sorted(self.rrf(lists).items(), key=lambda kv: kv[1][0], reverse=True)[:candidates]
        hits = [Hit(self.by_id[cid], score, src) for cid, (score, src) in fused]

        if mode == "hybrid_rerank" and hits:
            pairs = [(query, h.chunk["embed_text"]) for h in hits]
            scores = get_reranker().predict(pairs, batch_size=16)
            for h, s in zip(hits, scores):
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
