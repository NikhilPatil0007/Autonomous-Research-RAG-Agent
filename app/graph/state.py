from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict):
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
