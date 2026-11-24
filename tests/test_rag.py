"""Unit tests for chunking, tokenization, and RRF fusion (no heavy models required)."""
from __future__ import annotations

from data_pipeline.chunk import chunk_doc
from rag.retriever import HybridRetriever, tokenize


def test_chunk_doc_preserves_heading_context():
    doc = {
        "id": "doc1",
        "url": "https://example.de/admission",
        "title": "Admission",
        "lang": "en",
        "source_id": "test",
        "category": "program",
        "university": "TU Test",
        "program": "M.Sc. CS",
        "crawled_at": "2026-11-23T00:00:00+00:00",
        "text": "# Admission\n\n## Language\n\nEnglish C1 is required for all international applicants applying to this master's degree program in computer science.\n",
    }
    chunks = chunk_doc(doc)
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Admission > Language"
    assert "TU Test | Admission | Admission > Language" in chunks[0]["embed_text"]


def test_tokenize_handles_german_and_cjk_bigrams():
    toks = tokenize("申请 Sperrkonto 费用 Öffnungszeiten")
    assert "sperrkonto" in toks
    assert "öffnungszeiten" in toks
    assert "申请" in toks
    assert "费用" in toks


def test_rrf_boosts_items_found_by_multiple_retrievers():
    fused = HybridRetriever.rrf({
        "dense0": ["a", "b", "c"],
        "bm250": ["b", "d", "a"],
    })
    ranked = [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)]
    # Both 'a' and 'b' appear in both lists (ranks 0+2 and 1+0 -> 'b' slightly ahead of 'a')
    assert ranked[0] == "b"
    assert ranked[1] == "a"
    assert set(fused["b"][1]) == {"dense0", "bm250"}
