"""LangGraph orchestration for the production research pipeline.

plan -> multi-query retrieve -> write -> critique -> fact-check -> judge
                                             ^                     |
                                             |---- revise ----------|
accept -> final edit -> finalize; reject/exhausted -> unresolved finalize
"""

from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.agents import plan_query, synthesize, critique, fact_check, judge, final_edit
from src.evidence import analyze_citations
from src.research import retrieve_for_queries
from src.retrieval import HybridRetriever

MAX_REVISIONS = 2


class AnalystState(TypedDict):
    query: str
    subquestions: list[str]
    plan_summary: str
    chunks: list
    draft: str
    revision_count: int
    critique_history: list
    fact_check_history: list
    judge_history: list
    citation_analysis: dict
    final_answer: str
    status: str  # approved | rejected | unresolved


def build_graph(retriever: HybridRetriever):
    def plan_node(state: AnalystState) -> AnalystState:
        plan = plan_query(state["query"])
        subs = [q.strip() for q in plan.get("subquestions", []) if isinstance(q, str) and q.strip()]
        if state["query"] not in subs:
            subs.insert(0, state["query"])
        return {**state, "subquestions": subs[:6], "plan_summary": plan.get("reasoning_summary", "")}

    def retrieve_node(state: AnalystState) -> AnalystState:
        chunks = retrieve_for_queries(retriever, state["subquestions"], k_per_query=3, max_chunks=8)
        return {**state, "chunks": chunks}

    def synthesize_node(state: AnalystState) -> AnalystState:
        feedback_parts: list[str] = []
        if state.get("critique_history"):
            feedback_parts.append(state["critique_history"][-1].get("feedback", ""))
        if state.get("fact_check_history"):
            feedback_parts.append(state["fact_check_history"][-1].get("feedback", ""))
        draft = synthesize(state["query"], state["chunks"], "\n".join(p for p in feedback_parts if p))
        return {**state, "draft": draft}

    def critique_node(state: AnalystState) -> AnalystState:
        result = critique(state["query"], state["chunks"], state["draft"])
        return {**state, "critique_history": state.get("critique_history", []) + [result]}

    def fact_check_node(state: AnalystState) -> AnalystState:
        citation_analysis = analyze_citations(state["draft"], state["chunks"])
        result = fact_check(state["query"], state["chunks"], state["draft"], citation_analysis)
        return {
            **state,
            "citation_analysis": citation_analysis,
            "fact_check_history": state.get("fact_check_history", []) + [result],
        }

    def judge_node(state: AnalystState) -> AnalystState:
        result = judge(
            state["query"],
            state["critique_history"][-1],
            state["fact_check_history"][-1],
            state["citation_analysis"],
        )
        return {**state, "judge_history": state.get("judge_history", []) + [result]}

    def route_after_judge(state: AnalystState) -> str:
        decision = state["judge_history"][-1].get("decision", "revise")
        deterministic_failure = bool(state["citation_analysis"].get("invalid_citations"))
        if decision == "accept" and not deterministic_failure:
            return "final_edit"
        if decision == "reject":
            return "finalize_rejected"
        if state["revision_count"] >= MAX_REVISIONS:
            return "finalize_unresolved"
        return "revise"

    def bump_revision(state: AnalystState) -> AnalystState:
        return {**state, "revision_count": state["revision_count"] + 1}

    def final_edit_node(state: AnalystState) -> AnalystState:
        edited = final_edit(state["query"], state["draft"])
        return {**state, "final_answer": edited, "status": "approved"}

    def finalize_rejected_node(state: AnalystState) -> AnalystState:
        return {**state, "final_answer": state["draft"], "status": "rejected"}

    def finalize_unresolved_node(state: AnalystState) -> AnalystState:
        return {**state, "final_answer": state["draft"], "status": "unresolved"}

    graph = StateGraph(AnalystState)
    for name, fn in [
        ("plan", plan_node),
        ("retrieve", retrieve_node),
        ("synthesize", synthesize_node),
        ("critique", critique_node),
        ("fact_check", fact_check_node),
        ("judge", judge_node),
        ("bump_revision", bump_revision),
        ("final_edit", final_edit_node),
        ("finalize_rejected", finalize_rejected_node),
        ("finalize_unresolved", finalize_unresolved_node),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", "fact_check")
    graph.add_edge("fact_check", "judge")
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "final_edit": "final_edit",
            "finalize_rejected": "finalize_rejected",
            "finalize_unresolved": "finalize_unresolved",
            "revise": "bump_revision",
        },
    )
    graph.add_edge("bump_revision", "synthesize")
    graph.add_edge("final_edit", END)
    graph.add_edge("finalize_rejected", END)
    graph.add_edge("finalize_unresolved", END)
    return graph.compile()


def run_query(retriever: HybridRetriever, query: str) -> AnalystState:
    app = build_graph(retriever)
    initial_state: AnalystState = {
        "query": query,
        "subquestions": [],
        "plan_summary": "",
        "chunks": [],
        "draft": "",
        "revision_count": 0,
        "critique_history": [],
        "fact_check_history": [],
        "judge_history": [],
        "citation_analysis": {},
        "final_answer": "",
        "status": "",
    }
    return app.invoke(initial_state)
