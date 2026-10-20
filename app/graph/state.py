from __future__ import annotations

from typing import Annotated, Any, TypedDict

MEMORY_LIMIT = 6


def keep_last_messages(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Append, then keep only the last 6 turns (user, tool, assistant)."""
    return ((existing or []) + (new or []))[-MEMORY_LIMIT:]


class ResearchState(TypedDict):
    messages: Annotated[list[dict], keep_last_messages]
    question: str
    search_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    needs_more_research: bool
    iteration: int
    critic_pass: bool
    critic_notes: str
    seen_chunk_ids: list[str]
    trace: list[str]
    critic_retrieve_used: bool
    researcher_document_id: str
