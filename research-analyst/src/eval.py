"""Evaluation and tracing for research runs."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

from src.evidence import analyze_citations

TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def score_faithfulness(final_answer: str, chunks: list) -> float:
    return analyze_citations(final_answer, chunks)["sentence_citation_coverage"]


def score_run(state: dict) -> dict:
    citation = analyze_citations(state["final_answer"], state["chunks"])
    unique_sources = len({c.source for c in state["chunks"]})
    return {
        "faithfulness": citation["sentence_citation_coverage"],
        "citation_count": citation["citation_count"],
        "invalid_citation_count": len(citation["invalid_citations"]),
        "source_diversity": unique_sources,
        "retrieved_chunk_count": len(state["chunks"]),
        "revision_count": state["revision_count"],
    }


def log_trace(state: dict, faithfulness_score: float | None = None, elapsed_seconds: float = 0.0) -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    metrics = score_run(state)
    if faithfulness_score is not None:
        metrics["faithfulness"] = faithfulness_score
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state["query"],
        "subquestions": state.get("subquestions", []),
        "plan_summary": state.get("plan_summary", ""),
        "retrieved_chunks": [
            {
                "doc_id": c.doc_id,
                "source": c.source,
                "score": c.score,
                "score_breakdown": c.score_breakdown,
            }
            for c in state["chunks"]
        ],
        "revision_count": state["revision_count"],
        "critique_history": state.get("critique_history", []),
        "fact_check_history": state.get("fact_check_history", []),
        "judge_history": state.get("judge_history", []),
        "citation_analysis": state.get("citation_analysis", {}),
        "status": state["status"],
        "metrics": metrics,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "final_answer": state["final_answer"],
    }
    filename = os.path.join(TRACE_DIR, f"trace_{int(time.time() * 1000)}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return filename
