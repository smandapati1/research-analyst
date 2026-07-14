"""
Tests for the Multi-Agent Research Analyst.

Run with: python -m pytest tests/ -v
(or just: python tests/test_graph.py)

These tests mock the LLM calls (synthesize/critique) so they run for free
and don't require an ANTHROPIC_API_KEY, while still exercising the real
LangGraph orchestration logic and the real TF-IDF retriever.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval import TfidfRetriever
from src import graph as graph_mod
from src.eval import score_faithfulness
from src.retrieval import Chunk

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


def test_retrieval_returns_relevant_chunks():
    retriever = TfidfRetriever(CORPUS_DIR)
    results = retriever.retrieve("regulatory catalysts embodied carbon underwriting", k=4)
    assert len(results) > 0
    sources = {c.source for c in results}
    # AB 2446 and the underwriting doc are the most directly relevant sources
    assert "doc2_ab2446.md" in sources or "doc5_underwriting.md" in sources


def test_critique_rejection_triggers_revision_and_approval():
    retriever = TfidfRetriever(CORPUS_DIR)
    call_count = {"synthesize": 0, "critique": 0}

    def fake_synthesize(query, chunks):
        call_count["synthesize"] += 1
        if call_count["synthesize"] == 1:
            return "Embodied carbon is becoming important for lenders."  # uncited, should fail
        return ("Embodied carbon underwriting is being pushed by GRESB asset-level "
                "scoring [doc1_gresb.md] and California AB 2446 [doc2_ab2446.md].")

    def fake_critique(query, chunks, draft):
        call_count["critique"] += 1
        if "[doc" in draft:
            return {"approved": True, "issues": [], "feedback": ""}
        return {"approved": False, "issues": ["no citations found"],
                "feedback": "Add inline citations like [source.md] for every claim."}

    with patch.object(graph_mod, "synthesize", side_effect=fake_synthesize), \
         patch.object(graph_mod, "critique", side_effect=fake_critique):
        result = graph_mod.run_query(retriever, "What regulatory catalysts affect embodied carbon underwriting?")

    assert result["status"] == "approved"
    assert result["revision_count"] == 1
    assert call_count["synthesize"] == 2
    assert call_count["critique"] == 2
    assert "[doc" in result["final_answer"]


def test_max_revisions_forces_unresolved_finalize():
    retriever = TfidfRetriever(CORPUS_DIR)

    def always_bad_synthesize(query, chunks):
        return "Uncited claim with no sources."

    def always_reject_critique(query, chunks, draft):
        return {"approved": False, "issues": ["no citations"], "feedback": "add citations"}

    with patch.object(graph_mod, "synthesize", side_effect=always_bad_synthesize), \
         patch.object(graph_mod, "critique", side_effect=always_reject_critique):
        result = graph_mod.run_query(retriever, "test query")

    # Should stop after MAX_REVISIONS (2) rather than looping forever
    assert result["status"] == "unresolved"
    assert result["revision_count"] == graph_mod.MAX_REVISIONS


def test_faithfulness_scorer_penalizes_uncited_claims():
    chunks = [Chunk(doc_id="a", text="...", source="doc1_gresb.md")]
    fully_cited = "GRESB is shifting to asset-level scoring [doc1_gresb.md]."
    partially_cited = "GRESB is shifting to asset-level scoring [doc1_gresb.md]. This will double values."

    assert score_faithfulness(fully_cited, chunks) == 1.0
    assert score_faithfulness(partially_cited, chunks) < 1.0


if __name__ == "__main__":
    test_retrieval_returns_relevant_chunks()
    print("PASS: test_retrieval_returns_relevant_chunks")
    test_critique_rejection_triggers_revision_and_approval()
    print("PASS: test_critique_rejection_triggers_revision_and_approval")
    test_max_revisions_forces_unresolved_finalize()
    print("PASS: test_max_revisions_forces_unresolved_finalize")
    test_faithfulness_scorer_penalizes_uncited_claims()
    print("PASS: test_faithfulness_scorer_penalizes_uncited_claims")
    print("\nAll tests passed.")
