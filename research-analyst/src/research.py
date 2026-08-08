"""Research helpers: planned multi-query retrieval and evidence consolidation."""

from __future__ import annotations

from src.retrieval import Chunk


def retrieve_for_queries(retriever, queries: list[str], *, k_per_query: int = 3, max_chunks: int = 8) -> list[Chunk]:
    """Retrieve for multiple planned queries, dedupe by chunk id, keep strongest score."""
    by_id: dict[str, Chunk] = {}
    for query in queries:
        for chunk in retriever.retrieve(query, k=k_per_query):
            existing = by_id.get(chunk.doc_id)
            if existing is None or chunk.score > existing.score:
                by_id[chunk.doc_id] = chunk
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)[:max_chunks]
