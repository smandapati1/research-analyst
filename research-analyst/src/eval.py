"""
Evaluation and tracing layer.

This is a lightweight, self-built substitute for Ragas + LangSmith (which
require separate hosted accounts and API keys). It implements the same core
idea: score each run for faithfulness and log a full trace so failure modes
are visible after the fact instead of only through spot-checking outputs.

Faithfulness score: fraction of the final answer's sentences that contain a
source citation matching one of the retrieved chunks' source files. This is
a crude proxy (a real Ragas-style faithfulness check uses an LLM judge to
verify each claim against the source), but it is cheap, fast, and catches
the most common failure mode seen in testing: the synthesis agent dropping
citations on later sentences after citing correctly early in the answer.

To swap in real Ragas: replace `score_faithfulness` with a call to
`ragas.metrics.faithfulness` and feed it (question, answer, contexts).
To swap in real LangSmith: replace `log_trace` with `@traceable`-decorated
functions per LangSmith's SDK.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def score_faithfulness(final_answer: str, chunks: list) -> float:
    valid_sources = {c.source for c in chunks}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", final_answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = 0
    for s in sentences:
        found_sources = re.findall(r"\[([^\]]+\.md)\]", s)
        if any(src in valid_sources for src in found_sources):
            cited += 1
    return round(cited / len(sentences), 3)


def log_trace(state: dict, faithfulness_score: float, elapsed_seconds: float) -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state["query"],
        "retrieved_sources": [c.source for c in state["chunks"]],
        "revision_count": state["revision_count"],
        "critique_history": state["critique_history"],
        "status": state["status"],
        "faithfulness_score": faithfulness_score,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "final_answer": state["final_answer"],
    }
    filename = os.path.join(TRACE_DIR, f"trace_{int(time.time() * 1000)}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return filename
