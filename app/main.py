"""CLI for the LangGraph research agent.

  python -m app.main "Compare RAG and fine-tuning for domain-specific QA"
  python -m app.main "..." --max-iterations 3
  python -m app.main --dry-run-retrieve "RAG vs fine-tuning" --k 3
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.graph.state import ResearchState
from app.graph.workflow import build_graph, load_thread_messages, save_thread_messages
from app.observability import get_logger, preview, setup_logging, span, start_run
from app.tools import QdrantUnavailable, ensure_qdrant, retrieve_chunks

log = get_logger("app.main")


def _initial_state(question: str) -> ResearchState:
    return {
        "question": question,
        "search_queries": [],
        "retrieved_chunks": [],
        "evidence": [],
        "draft_answer": "",
        "final_answer": "",
        "needs_more_research": False,
        "iteration": 0,
        "critic_pass": False,
        "critic_notes": "",
        "seen_chunk_ids": [],
        "messages": [],
        "trace": [],
        "critic_retrieve_used": False,
        "researcher_document_id": "",
    }


def _apply_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    if "messages" in update:
        state["messages"] = ((state.get("messages") or []) + list(update["messages"]))[-6:]
        rest = {key: value for key, value in update.items() if key != "messages"}
        state.update(rest)
        return
    state.update(update)


def _apply_max_iterations(n: int | None) -> None:
    if n is None:
        return
    if n < 1:
        raise SystemExit("--max-iterations must be >= 1")
    os.environ["MAX_RESEARCH_ITERATIONS"] = str(n)
    get_settings.cache_clear()


def _print_update(node: str, update: dict[str, Any]) -> None:
    print(f"\n--- {node} ---")
    if node == "session":
        print(f"thread: {update.get('thread_id')}  prior_messages: {update.get('prior_messages')}")
    if "search_queries" in update:
        queries = update["search_queries"] or []
        print("queries:")
        for q in queries:
            print(f"  - {q}")
    if "iteration" in update:
        print(f"iteration: {update['iteration']}")
    if "needs_more_research" in update:
        print(f"needs_more_research: {update['needs_more_research']}")
    if "critic_pass" in update:
        print(f"critic_pass: {update['critic_pass']}")
        notes = update.get("critic_notes") or ""
        if notes:
            print(f"critic_notes: {notes}")
    if "n_chunks" in update:
        print(f"chunks: {update['n_chunks']}")
    path = update.get("path") or update.get("trace")
    if path:
        print("path:", " -> ".join(path))


def public_update(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Fields a client can show while a run is still going."""
    event: dict[str, Any] = {"node": node}
    if "search_queries" in update:
        event["search_queries"] = list(update["search_queries"] or [])
    if "retrieved_chunks" in update:
        event["n_chunks"] = len(update["retrieved_chunks"] or [])
    if "iteration" in update:
        event["iteration"] = update["iteration"]
    if "needs_more_research" in update:
        event["needs_more_research"] = bool(update["needs_more_research"])
    if "critic_pass" in update:
        event["critic_pass"] = bool(update["critic_pass"])
        event["critic_notes"] = update.get("critic_notes") or ""
    if "final_answer" in update:
        event["final_answer"] = update.get("final_answer") or ""
    if "trace" in update:
        event["path"] = list(update["trace"] or [])
    return event


def iter_research(question: str, thread_id: str = "default") -> Iterator[dict[str, Any]]:
    """Run one research question and yield a public event after each node."""
    prior = load_thread_messages(thread_id)
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    state: dict[str, Any] = dict(_initial_state(question))
    state["messages"] = prior
    yield {
        "node": "session",
        "thread_id": thread_id,
        "prior_messages": len(prior),
    }
    start_run(
        kind="research",
        question=preview(question),
        max_iterations=get_settings().max_research_iterations,
        thread_id=thread_id,
    )
    with span("graph.run", component="graph", question=preview(question)) as bag:
        try:
            for event in graph.stream(state, config, stream_mode="updates"):
                for node, update in event.items():
                    if not isinstance(update, dict):
                        continue
                    _apply_update(state, update)
                    log.info(
                        "graph.node.update",
                        extra={
                            "component": "graph",
                            "event": "graph.node.update",
                            "node": node,
                            "path": state.get("trace"),
                        },
                    )
                    yield public_update(node, update)
        except Exception:
            log.exception(
                "graph.run.fail",
                extra={
                    "component": "graph",
                    "event": "graph.run.fail",
                    "path": state.get("trace"),
                    "iteration": state.get("iteration"),
                },
            )
            raise
        bag["path"] = state.get("trace")
        bag["iteration"] = state.get("iteration")
        bag["critic_pass"] = state.get("critic_pass")
        bag["n_evidence"] = len(state.get("evidence") or [])
    snapshot = graph.get_state(config)
    if snapshot.values.get("messages"):
        state["messages"] = list(snapshot.values["messages"])[-6:]
    save_thread_messages(thread_id, state.get("messages") or [])
    yield {
        "node": "done",
        "path": list(state.get("trace") or []),
        "final_answer": state.get("final_answer") or state.get("draft_answer") or "",
        "messages": list(state.get("messages") or []),
    }


def dry_run_retrieve(query: str, k: int) -> None:
    settings = get_settings()
    start_run(kind="retrieval", query=preview(query), k=k)
    print(f"Qdrant: {settings.qdrant_url}  collection={settings.qdrant_collection!r}")
    print(f"query: {query!r}  k={k}\n")
    hits = retrieve_chunks(query, k=k)
    log.info(
        "cli.dry_run",
        extra={
            "component": "retrieval",
            "event": "cli.dry_run",
            "n_hits": len(hits),
            "query": preview(query),
        },
    )
    if not hits:
        print("No hits. Ingest PDFs first: python scripts/ingest.py")
        return
    for i, row in enumerate(hits, start=1):
        title = row.get("title") or row.get("source")
        text_preview = (row.get("text") or "").replace("\n", " ")[:220]
        print(f"[{i}] {title}, p. {row.get('page')}  score={row.get('score')}")
        print(f"    chunk_id={row.get('chunk_id')}")
        print(f"    {text_preview}\n")


def run_research(question: str, thread_id: str = "default") -> dict[str, Any]:
    prior = load_thread_messages(thread_id)
    print(f"Question: {question}")
    print(f"thread: {thread_id}  prior_messages: {len(prior)}")
    print(f"max retrieve/rewrite loops: {get_settings().max_research_iterations}")
    final: dict[str, Any] = {}
    for event in iter_research(question, thread_id=thread_id):
        if event.get("node") == "done":
            final = event
            continue
        _print_update(str(event["node"]), event)
    messages = final.get("messages") or []
    print("\n========== memory ==========")
    for message in messages:
        label = message.get("role") or "?"
        if message.get("name"):
            label = f"{label}:{message['name']}"
        print(f"{label}: {message.get('content')}")
    print("\n========== path ==========")
    print(" -> ".join(final.get("path") or []))
    print("\n========== answer ==========\n")
    print(final.get("final_answer") or "(no answer)")
    return final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Autonomous Research RAG Agent (LangGraph CLI)")
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap on retrieve/rewrite loops (default: MAX_RESEARCH_ITERATIONS from .env)",
    )
    parser.add_argument(
        "--dry-run-retrieve",
        metavar="QUERY",
        help="Vector-search Qdrant only (no LangGraph / no Gemini generation)",
    )
    parser.add_argument("-k", type=int, default=5, help="Top-k for --dry-run-retrieve")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default: LOG_LEVEL)")
    parser.add_argument(
        "--thread",
        default="default",
        help="Short-memory thread id. The next question on this thread sees the last 6 messages.",
    )
    args = parser.parse_args(argv)

    _apply_max_iterations(args.max_iterations)
    settings = get_settings()
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
        get_settings.cache_clear()
        settings = get_settings()
    if settings.log_json:
        os.environ["LOG_JSON"] = "true"
    log_path = setup_logging(level=args.log_level or settings.log_level, log_dir=settings.log_dir)
    log.info("cli.start", extra={"component": "graph", "event": "cli.start", "log_path": str(log_path)})

    try:
        ensure_qdrant()
    except QdrantUnavailable as exc:
        log.error("cli.qdrant_down", extra={"component": "retrieval", "event": "cli.qdrant_down"})
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.dry_run_retrieve:
        dry_run_retrieve(args.dry_run_retrieve, args.k)
        return
    if not args.question:
        parser.error("question is required unless --dry-run-retrieve is set")
    run_research(args.question, thread_id=args.thread)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except QdrantUnavailable as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except Exception:
        log.exception("cli.crash", extra={"component": "graph", "event": "cli.crash"})
        raise
