"""Global settings, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
QDRANT_PATH = DATA_DIR / "qdrant"
PROGRAMS_PATH = DATA_DIR / "programs.yaml"
PROGRAMS_DB_PATH = DATA_DIR / "programs.db"
DAAD_PATH = DATA_DIR / "daad_programs.json"
EXTRACTED_PROGRAMS_PATH = DATA_DIR / "extracted_programs.json"
FACTS_PATH = DATA_DIR / "facts.yaml"
PROFILE_PATH = DATA_DIR / "profile.json"
TRACES_PATH = DATA_DIR / "traces.jsonl"
PROMPTS_DIR = ROOT / "prompts"

COLLECTION = "destudy_chunks"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.getenv("DEVICE", "")  # "", "mps", "cuda", "cpu" ("" = auto)

USER_AGENT = "DeStudyAgentBot/0.1 (personal study project; respects robots.txt)"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def pick_device() -> str:
    if DEVICE:
        return DEVICE
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
