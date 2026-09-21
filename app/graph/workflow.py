from langgraph.graph import END, START, StateGraph

from app.graph.edges import after_critic, after_evidence_analyzer
from app.graph.nodes import (
    critic,
    evidence_analyzer,
    finalize,
    planner,
    query_rewriter,
    researcher,
    retriever,
    synthesizer,
)
from app.graph.state import ResearchState


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("evidence_analyzer", evidence_analyzer)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("critic", critic)
    graph.add_node("researcher", researcher)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "evidence_analyzer")
    graph.add_conditional_edges(
        "evidence_analyzer",
        after_evidence_analyzer,
        {"query_rewriter": "query_rewriter", "synthesizer": "synthesizer"},
    )
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges(
        "critic",
        after_critic,
        {"researcher": "researcher", "finalize": "finalize"},
    )
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("finalize", END)
    return graph.compile()
