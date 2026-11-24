"""Embed chunks with bge-m3 and write them into a local Qdrant collection.

Usage: python -m data_pipeline.index
"""
from __future__ import annotations

import json
import uuid

from qdrant_client import QdrantClient, models

from rag.models import get_embedder
from settings import CHUNKS_PATH, COLLECTION, QDRANT_PATH


def main(batch_size: int = 16) -> None:
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    embedder = get_embedder()
    dim = (
        embedder.get_embedding_dimension()
        if hasattr(embedder, "get_embedding_dimension")
        else embedder.get_sentence_embedding_dimension()
    )

    client = QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embedder.encode([c["embed_text"] for c in batch], normalize_embeddings=True, batch_size=batch_size)
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
                    vector=v.tolist(),
                    payload={k: v2 for k, v2 in c.items() if k != "embed_text"},
                )
                for c, v in zip(batch, vecs)
            ],
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")
    client.close()
    print(f"done: {len(chunks)} chunks, dim={dim}, collection={COLLECTION}")


if __name__ == "__main__":
    main()
