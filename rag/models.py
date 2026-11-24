"""Lazy singletons for the local embedding / reranking models and the LLM client."""
from __future__ import annotations

from functools import lru_cache

from settings import EMBED_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, RERANK_MODEL, pick_device


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    dev = pick_device()
    try:
        return SentenceTransformer(EMBED_MODEL, device=dev, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBED_MODEL, device=dev)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    dev = pick_device()
    try:
        return CrossEncoder(RERANK_MODEL, device=dev, max_length=1024, local_files_only=True)
    except Exception:
        return CrossEncoder(RERANK_MODEL, device=dev, max_length=1024)


@lru_cache(maxsize=1)
def get_llm():
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (copy .env.example to .env)")
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
