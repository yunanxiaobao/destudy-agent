"""Structure-aware chunking: split markdown by headings, then pack paragraphs.

Each chunk is prefixed with a context header (title + heading path) so the
embedding knows *which page / section* the text belongs to.

Usage: python -m data_pipeline.chunk
"""
from __future__ import annotations

import json
import re

from settings import CHUNKS_PATH, RAW_DIR

MAX_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHARS = 80
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, body)]."""
    sections: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(([h for _, h in path], body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            path = [(l, h) for l, h in path if l < level] + [(level, heading)]
        else:
            buf.append(line)
    flush()
    return sections


def pack(body: str) -> list[str]:
    """Greedily pack paragraphs into chunks <= MAX_CHARS, with tail overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > MAX_CHARS:  # very long paragraph: hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:MAX_CHARS])
            p = p[MAX_CHARS - OVERLAP_CHARS:]
        if len(cur) + len(p) + 2 <= MAX_CHARS:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            chunks.append(cur)
            cur = (cur[-OVERLAP_CHARS:] + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    out = []
    for path, body in split_sections(doc["text"]):
        for i, piece in enumerate(pack(body)):
            if len(piece) < MIN_CHARS:
                continue
            header_parts = [doc.get("university") or "", doc["title"], " > ".join(path)]
            header = " | ".join(p for p in header_parts if p)
            out.append({
                "chunk_id": f"{doc['id']}-{len(out)}",
                "doc_id": doc["id"],
                "text": piece,
                "embed_text": f"{header}\n\n{piece}",
                "section": " > ".join(path),
                **{k: doc[k] for k in ("url", "title", "lang", "source_id", "category", "university", "program", "crawled_at")},
            })
    return out


def main() -> None:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RAW_DIR.glob("*/*.json"))]
    seen_texts: set[str] = set()
    n = 0
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            for c in chunk_doc(doc):
                key = c["text"][:300]
                if key in seen_texts:  # drop boilerplate repeated across pages
                    continue
                seen_texts.add(key)
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
    print(f"{len(docs)} docs -> {n} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
