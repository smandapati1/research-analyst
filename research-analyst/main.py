"""CLI entry point for the Multi-Agent Research Analyst."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.eval import log_trace, score_run
from src.graph import run_query
from src.retrieval import HybridRetriever

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "your question here"')
        raise SystemExit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY before running.")
        raise SystemExit(1)

    query = sys.argv[1]
    retriever = HybridRetriever(CORPUS_DIR)
    start = time.time()
    state = run_query(retriever, query)
    elapsed = time.time() - start
    metrics = score_run(state)
    trace_path = log_trace(state, elapsed_seconds=elapsed)

    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)
    print(f"STATUS: {state['status']} | REVISIONS: {state['revision_count']} | TIME: {elapsed:.1f}s")
    print(f"FAITHFULNESS: {metrics['faithfulness']} | SOURCE DIVERSITY: {metrics['source_diversity']}")
    print("\nPLAN:")
    for i, q in enumerate(state["subquestions"], 1):
        print(f"  {i}. {q}")
    print("\nRETRIEVED EVIDENCE:")
    for c in state["chunks"]:
        print(f"  - {c.source} ({c.doc_id}) score={c.score:.4f}")
    print("\n--- ANSWER ---\n")
    print(state["final_answer"])
    print(f"\n(full trace logged to {trace_path})")


if __name__ == "__main__":
    main()
