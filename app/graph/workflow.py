from __future__ import annotations

import json
import threading
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config import ROOT_DIR
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
from app.graph.state import MEMORY_LIMIT, ResearchState

_session_lock = threading.Lock()


def session_path(thread_id: str):
    """Conversation file for one thread. This is context, not a retrieval cache."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in thread_id) or "default"
    return ROOT_DIR / "data" / "sessions" / f"{safe}.json"


def load_thread_messages(thread_id: str) -> list[dict[str, Any]]:
    path = session_path(thread_id)
    with _session_lock:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages") or []
    return list(messages)[-MEMORY_LIMIT:]


def save_thread_messages(thread_id: str, messages: list[dict[str, Any]]) -> None:
    path = session_path(thread_id)
    kept = list(messages or [])[-MEMORY_LIMIT:]
    payload = json.dumps({"thread_id": thread_id, "messages": kept}, indent=2)
    with _session_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def build_graph(checkpointer=None):
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
    return graph.compile(checkpointer=checkpointer)
