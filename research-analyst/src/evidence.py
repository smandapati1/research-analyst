"""Deterministic citation and evidence checks used alongside LLM reviewers."""

from __future__ import annotations

import re


def analyze_citations(answer: str, chunks: list) -> dict:
    valid_sources = {c.source for c in chunks}
    citations = re.findall(r"\[([^\]]+\.md)\]", answer)
    invalid = sorted({c for c in citations if c not in valid_sources})

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    factual = [s for s in sentences if len(re.findall(r"\b\w+\b", s)) >= 3]
    supported = 0
    for sentence in factual:
        cited = re.findall(r"\[([^\]]+\.md)\]", sentence)
        if cited and any(c in valid_sources for c in cited):
            supported += 1

    coverage = supported / len(factual) if factual else 0.0
    return {
        "citation_count": len(citations),
        "unique_citations": sorted(set(citations)),
        "invalid_citations": invalid,
        "sentence_citation_coverage": round(coverage, 3),
        "factual_sentence_count": len(factual),
    }
