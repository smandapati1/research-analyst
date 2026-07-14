"""
LangGraph orchestration for the Multi-Agent Research Analyst.

Graph shape:

    retrieve -> synthesize -> critique --(approved)--> finalize
                    ^                |
                    |         (not approved, revisions < MAX)
                    +----------------+

The critique node can send the graph back to synthesize with specific
feedback, up to MAX_REVISIONS times, before forcing finalize with a flagged
"unresolved" status. This bounded loop is what makes the critique agent
meaningful rather than decorative: it can actually change the final output,
not just log an opinion after the fact.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.retrieval import TfidfRetriever, Chunk
from src.agents import synthesize, critique

MAX_REVISIONS = 2


class AnalystState(TypedDict):
    query: str
    chunks: list
    draft: str
    revision_count: int
    critique_history: list
    final_answer: str
    status: str  # "approved" | "unresolved"


def build_graph(retriever: TfidfRetriever):

    def retrieve_node(state: AnalystState) -> AnalystState:
        chunks = retriever.retrieve(state["query"], k=4)
        return {**state, "chunks": chunks}

    def synthesize_node(state: AnalystState) -> AnalystState:
        feedback_note = ""
        if state.get("critique_history"):
            last = state["critique_history"][-1]
            feedback_note = f"\n\nRevision instructions from reviewer: {last['feedback']}"
        draft = synthesize(state["query"] + feedback_note, state["chunks"])
        return {**state, "draft": draft}

    def critique_node(state: AnalystState) -> AnalystState:
        result = critique(state["query"], state["chunks"], state["draft"])
        history = state.get("critique_history", []) + [result]
        return {**state, "critique_history": history}

    def route_after_critique(state: AnalystState) -> str:
        last = state["critique_history"][-1]
        if last["approved"]:
            return "finalize"
        if state["revision_count"] >= MAX_REVISIONS:
            return "finalize_unresolved"
        return "revise"

    def bump_revision(state: AnalystState) -> AnalystState:
        return {**state, "revision_count": state["revision_count"] + 1}

    def finalize_node(state: AnalystState) -> AnalystState:
        return {**state, "final_answer": state["draft"], "status": "approved"}

    def finalize_unresolved_node(state: AnalystState) -> AnalystState:
        return {**state, "final_answer": state["draft"], "status": "unresolved"}

    graph = StateGraph(AnalystState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critique", critique_node)
    graph.add_node("bump_revision", bump_revision)
    graph.add_node("finalize", finalize_node)
    graph.add_node("finalize_unresolved", finalize_unresolved_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"finalize": "finalize", "finalize_unresolved": "finalize_unresolved", "revise": "bump_revision"},
    )
    graph.add_edge("bump_revision", "synthesize")
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_unresolved", END)

    return graph.compile()


def run_query(retriever: TfidfRetriever, query: str) -> AnalystState:
    app = build_graph(retriever)
    initial_state: AnalystState = {
        "query": query,
        "chunks": [],
        "draft": "",
        "revision_count": 0,
        "critique_history": [],
        "final_answer": "",
        "status": "",
    }
    return app.invoke(initial_state)
