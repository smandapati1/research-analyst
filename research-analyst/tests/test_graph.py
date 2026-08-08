"""Integration tests for retrieval, evidence checks, and LangGraph routing."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import graph as graph_mod
from src.evidence import analyze_citations
from src.eval import score_faithfulness
from src.retrieval import Chunk, HybridRetriever

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


def test_hybrid_retrieval_returns_relevant_chunks_and_scores():
    retriever = HybridRetriever(CORPUS_DIR)
    results = retriever.retrieve("regulatory catalysts embodied carbon underwriting", k=4)
    assert results
    sources = {c.source for c in results}
    assert "doc2_ab2446.md" in sources or "doc5_underwriting.md" in sources
    assert all(c.score > 0 for c in results)
    assert all({"bm25", "word_tfidf", "char_tfidf"} <= set(c.score_breakdown) for c in results)


def test_citation_analysis_catches_invalid_source():
    chunks = [Chunk(doc_id="a", text="x", source="doc1_gresb.md")]
    result = analyze_citations(
        "GRESB has asset-level scoring [doc1_gresb.md]. A fake source says more [fake.md].",
        chunks,
    )
    assert result["invalid_citations"] == ["fake.md"]
    assert result["citation_count"] == 2


def test_full_pipeline_revision_then_acceptance():
    retriever = HybridRetriever(CORPUS_DIR)
    calls = {"write": 0, "judge": 0}

    def fake_plan(query):
        return {"subquestions": ["AB 2446 embodied carbon", "underwriting embodied carbon"], "reasoning_summary": "two angles"}

    def fake_synthesize(query, chunks, feedback=""):
        calls["write"] += 1
        if calls["write"] == 1:
            return "Embodied carbon is becoming important for lenders."
        return "California AB 2446 creates embodied-carbon reporting pressure [doc2_ab2446.md]."

    def fake_critique(query, chunks, draft):
        if "[doc2_ab2446.md]" in draft:
            return {"approved": True, "issues": [], "feedback": ""}
        return {"approved": False, "issues": ["uncited"], "feedback": "Add evidence-backed citations."}

    def fake_fact_check(query, chunks, draft, citation_analysis):
        passed = "[doc2_ab2446.md]" in draft
        return {"passed": passed, "unsupported_claims": [] if passed else ["uncited claim"], "citation_errors": [], "feedback": "Cite the claim." if not passed else ""}

    def fake_judge(query, critique_result, fact_check_result, citation_analysis):
        calls["judge"] += 1
        return {"decision": "accept" if fact_check_result["passed"] else "revise", "reason": "quality gate"}

    def fake_final_edit(query, answer):
        return answer

    with patch.object(graph_mod, "plan_query", side_effect=fake_plan), \
         patch.object(graph_mod, "synthesize", side_effect=fake_synthesize), \
         patch.object(graph_mod, "critique", side_effect=fake_critique), \
         patch.object(graph_mod, "fact_check", side_effect=fake_fact_check), \
         patch.object(graph_mod, "judge", side_effect=fake_judge), \
         patch.object(graph_mod, "final_edit", side_effect=fake_final_edit):
        result = graph_mod.run_query(retriever, "How is embodied carbon entering underwriting?")

    assert result["status"] == "approved"
    assert result["revision_count"] == 1
    assert calls["write"] == 2
    assert calls["judge"] == 2
    assert result["subquestions"]


def test_repeated_revision_exhausts_to_unresolved():
    retriever = HybridRetriever(CORPUS_DIR)

    with patch.object(graph_mod, "plan_query", return_value={"subquestions": ["test query"], "reasoning_summary": ""}), \
         patch.object(graph_mod, "synthesize", return_value="Uncited unsupported answer."), \
         patch.object(graph_mod, "critique", return_value={"approved": False, "issues": ["bad"], "feedback": "fix it"}), \
         patch.object(graph_mod, "fact_check", return_value={"passed": False, "unsupported_claims": ["bad"], "citation_errors": [], "feedback": "fix it"}), \
         patch.object(graph_mod, "judge", return_value={"decision": "revise", "reason": "still bad"}), \
         patch.object(graph_mod, "final_edit"):
        result = graph_mod.run_query(retriever, "test query")

    assert result["status"] == "unresolved"
    assert result["revision_count"] == graph_mod.MAX_REVISIONS


def test_judge_rejects_without_final_edit():
    retriever = HybridRetriever(CORPUS_DIR)

    with patch.object(graph_mod, "plan_query", return_value={"subquestions": ["test query"], "reasoning_summary": ""}), \
         patch.object(graph_mod, "synthesize", return_value="The corpus does not contain enough evidence."), \
         patch.object(graph_mod, "critique", return_value={"approved": False, "issues": ["insufficient evidence"], "feedback": ""}), \
         patch.object(graph_mod, "fact_check", return_value={"passed": True, "unsupported_claims": [], "citation_errors": [], "feedback": ""}), \
         patch.object(graph_mod, "judge", return_value={"decision": "reject", "reason": "insufficient evidence"}), \
         patch.object(graph_mod, "final_edit") as editor:
        result = graph_mod.run_query(retriever, "test query")

    assert result["status"] == "rejected"
    editor.assert_not_called()


def test_faithfulness_scorer_penalizes_uncited_claims():
    chunks = [Chunk(doc_id="a", text="...", source="doc1_gresb.md")]
    fully_cited = "GRESB is shifting to asset-level scoring [doc1_gresb.md]."
    partially_cited = "GRESB is shifting to asset-level scoring [doc1_gresb.md]. This will double values."
    assert score_faithfulness(fully_cited, chunks) == 1.0
    assert score_faithfulness(partially_cited, chunks) < 1.0
