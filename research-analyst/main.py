"""
Multi-Agent Research Analyst — CLI entry point.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python main.py "What regulatory catalysts are pushing embodied carbon
    data into commercial real estate underwriting?"

Runs the query through retrieve -> synthesize -> critique (with bounded
revision loop) -> finalize, then scores and logs the trace.
"""

import sys
import time
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.retrieval import TfidfRetriever
from src.graph import run_query
from src.eval import score_faithfulness, log_trace

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "your question here"')
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY before running.")
        sys.exit(1)

    query = sys.argv[1]
    retriever = TfidfRetriever(CORPUS_DIR)

    start = time.time()
    state = run_query(retriever, query)
    elapsed = time.time() - start

    faithfulness = score_faithfulness(state["final_answer"], state["chunks"])
    trace_path = log_trace(state, faithfulness, elapsed)

    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    print(f"\nSTATUS: {state['status']}  |  REVISIONS: {state['revision_count']}  |  "
          f"FAITHFULNESS: {faithfulness}  |  TIME: {elapsed:.1f}s")
    print(f"SOURCES USED: {[c.source for c in state['chunks']]}")
    print("\n--- ANSWER ---\n")
    print(state["final_answer"])
    print(f"\n(full trace logged to {trace_path})")


if __name__ == "__main__":
    main()
