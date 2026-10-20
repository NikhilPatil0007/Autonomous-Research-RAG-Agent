from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.graph.state import ResearchState
from app.llm import get_llm, invoke_structured
from app.observability import get_logger, preview, span
from app.tools import get_document, retrieve_chunks, search_again

log = get_logger("app.graph.nodes")


def _trace(state: ResearchState, name: str) -> list[str]:
    return list(state.get("trace") or []) + [name]


def _history_text(state: ResearchState) -> str:
    history = list(state.get("messages") or [])[-6:]
    if not history:
        return "(none)"
    lines = []
    for message in history:
        role = message.get("role") or "unknown"
        name = message.get("name")
        label = f"{role}:{name}" if name else role
        lines.append(f"{label}: {message.get('content') or ''}")
    return "\n".join(lines)


def _user_message(state: ResearchState) -> list[dict]:
    question = state.get("question") or ""
    history = state.get("messages") or []
    if history and history[-1].get("role") == "user" and history[-1].get("content") == question:
        return []
    return [{"role": "user", "content": question}]


def _tool_message(name: str, content: str) -> dict:
    return {"role": "tool", "name": name, "content": content[:500]}


def _hit_brief(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows[:8]:
        title = row.get("title") or row.get("source") or "source"
        parts.append(f"{title} p.{row.get('page')} ({row.get('chunk_id')})")
    return "; ".join(parts) or "none"


def _as_text(value: Any) -> str:
    """Gemini/LangChain may return content as str or a list of blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                text = getattr(item, "text", None)
                parts.append(str(text) if text is not None else str(item))
        return "\n".join(p for p in parts if p)
    return str(value)


def _chunk_id(row: dict[str, Any]) -> str:
    return str(row.get("chunk_id") or "")


def _merge_unique(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            cid = _chunk_id(row)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append(row)
    return merged


def _format_evidence(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for i, row in enumerate(chunks, start=1):
        title = row.get("title") or row.get("source") or "source"
        lines.append(
            f"[{i}] {title}, p. {row.get('page')} | {row.get('chunk_id')}\n"
            f"{row.get('text') or ''}"
        )
    return "\n\n".join(lines)


class PlannerOut(BaseModel):
    search_queries: list[str] = Field(min_length=2, max_length=4)


class AnalyzerOut(BaseModel):
    covered: list[str]
    gaps: list[str]
    needs_more_research: bool


class RewriterOut(BaseModel):
    search_queries: list[str] = Field(min_length=1, max_length=4)


class CriticOut(BaseModel):
    critic_pass: bool
    critic_notes: str
    followup_query: str = ""
    document_id: str = ""


def planner(state: ResearchState) -> dict:
    history_text = _history_text(state)
    with span("planner", component="generation", question=preview(state["question"])) as bag:
        out = invoke_structured(
            PlannerOut,
            "Decompose the user question into 2-4 focused search queries "
            "for a local research PDF corpus. Cover distinct facets "
            "(e.g. RAG advantages, fine-tuning limits, when each is appropriate). "
            "If the question is a follow-up, rewrite it into standalone search queries "
            "using the recent user, tool, and assistant messages. "
            "Do not search for pronouns alone.\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Question: {state['question']}",
        )
        bag["n_queries"] = len(out.search_queries)
        bag["queries"] = out.search_queries
        log.info(
            "planner.ok",
            extra={
                "component": "generation",
                "event": "planner.ok",
                "node": "planner",
                "n_queries": len(out.search_queries),
                "queries": out.search_queries,
            },
        )
        return {
            "search_queries": out.search_queries,
            "messages": _user_message(state),
            "trace": _trace(state, "planner"),
        }


def retriever(state: ResearchState) -> dict:
    seen = list(state.get("seen_chunk_ids") or [])
    existing = list(state.get("retrieved_chunks") or [])
    queries = state.get("search_queries") or [state["question"]]
    new_rows: list[dict[str, Any]] = []
    with span(
        "retriever",
        component="retrieval",
        n_queries=len(queries),
        n_seen=len(seen),
        n_existing=len(existing),
        iteration=int(state.get("iteration") or 0) + 1,
    ) as bag:
        for q in queries:
            if seen:
                new_rows.extend(search_again(q, seen))
            else:
                new_rows.extend(retrieve_chunks(q, k=5))
        merged = _merge_unique(existing, new_rows)
        n_new = len(merged) - len(existing)
        bag["n_new"] = n_new
        bag["n_chunks"] = len(merged)
        if n_new == 0:
            log.warning(
                "retriever.no_new_chunks",
                extra={
                    "component": "retrieval",
                    "event": "retriever.no_new_chunks",
                    "node": "retriever",
                    "n_queries": len(queries),
                    "n_existing": len(existing),
                    "n_seen": len(seen),
                },
            )
        log.info(
            "retriever.ok",
            extra={
                "component": "retrieval",
                "event": "retriever.ok",
                "node": "retriever",
                "iteration": int(state.get("iteration") or 0) + 1,
                "n_queries": len(queries),
                "n_new": n_new,
                "n_chunks": len(merged),
            },
        )
        return {
            "retrieved_chunks": merged,
            "seen_chunk_ids": [_chunk_id(r) for r in merged],
            "iteration": int(state.get("iteration") or 0) + 1,
            "messages": [
                _tool_message(
                    "retrieve_chunks",
                    f"queries={queries} | new_hits={_hit_brief(new_rows)}",
                )
            ],
            "trace": _trace(state, "retriever"),
        }


def evidence_analyzer(state: ResearchState) -> dict:
    snippets = []
    for i, row in enumerate(state.get("retrieved_chunks") or [], start=1):
        snippets.append(
            f"[{i}] {row.get('title')} p.{row.get('page')} ({row.get('chunk_id')})\n"
            f"{(row.get('text') or '')[:1200]}"
        )
    blob = "\n\n".join(snippets) or "(no chunks)"
    with span(
        "evidence_analyzer",
        component="generation",
        n_chunks=len(state.get("retrieved_chunks") or []),
    ) as bag:
        out = invoke_structured(
            AnalyzerOut,
            "Judge whether retrieved chunks cover the question. "
            "If one side of a comparison is thin, set needs_more_research=true "
            "and list gaps. Do not invent papers.\n\n"
            f"Question: {state['question']}\n\nChunks:\n{blob}",
        )
        bag["needs_more_research"] = out.needs_more_research
        bag["n_gaps"] = len(out.gaps)
        log.info(
            "analyzer.ok",
            extra={
                "component": "generation",
                "event": "analyzer.ok",
                "node": "evidence_analyzer",
                "needs_more_research": out.needs_more_research,
                "covered": out.covered,
                "gaps": out.gaps,
                "n_chunks": len(state.get("retrieved_chunks") or []),
            },
        )
        return {
            "needs_more_research": out.needs_more_research,
            "evidence": list(state.get("retrieved_chunks") or []),
            "critic_notes": f"covered={out.covered}; gaps={out.gaps}",
            "trace": _trace(state, "evidence_analyzer"),
        }


def query_rewriter(state: ResearchState) -> dict:
    with span("query_rewriter", component="generation") as bag:
        out = invoke_structured(
            RewriterOut,
            "Write 1-4 NEW search queries that target ONLY the evidence gaps. "
            "Do not repeat prior queries. Use the conversation to keep the same topic.\n\n"
            f"Recent conversation:\n{_history_text(state)}\n\n"
            f"Question: {state['question']}\n"
            f"Prior queries: {state.get('search_queries')}\n"
            f"Analyzer notes: {state.get('critic_notes')}",
        )
        bag["queries"] = out.search_queries
        log.info(
            "rewriter.ok",
            extra={
                "component": "generation",
                "event": "rewriter.ok",
                "node": "query_rewriter",
                "queries": out.search_queries,
            },
        )
        return {
            "search_queries": out.search_queries,
            "needs_more_research": False,
            "trace": _trace(state, "query_rewriter"),
        }


def synthesizer(state: ResearchState) -> dict:
    chunks = list(state.get("evidence") or state.get("retrieved_chunks") or [])
    with span("synthesizer", component="generation", n_evidence=len(chunks)) as bag:
        if not chunks:
            log.warning(
                "synthesizer.no_evidence",
                extra={
                    "component": "generation",
                    "event": "synthesizer.no_evidence",
                    "node": "synthesizer",
                },
            )
            draft = (
                "I could not find supporting passages in the local corpus. "
                "I will not invent papers or citations."
            )
            return {
                "draft_answer": draft,
                "evidence": [],
                "trace": _trace(state, "synthesizer"),
            }
        llm = get_llm(temperature=0.2)
        with span("llm.invoke", component="generation", schema="synthesizer_draft"):
            msg = llm.invoke(
                "Write a synthesis that answers the question using ONLY the evidence. "
                "Cite with [n] markers that match the numbered evidence items. "
                "Do not invent papers, titles, or page numbers. "
                "If evidence is one-sided, say so. "
                "Use the recent conversation to resolve follow-ups, not as a citation source.\n\n"
                f"Recent conversation:\n{_history_text(state)}\n\n"
                f"Question: {state['question']}\n\nEvidence:\n{_format_evidence(chunks)}"
            )
        draft = _as_text(getattr(msg, "content", None) or msg)
        bag["draft_chars"] = len(draft)
        log.info(
            "synthesizer.ok",
            extra={
                "component": "generation",
                "event": "synthesizer.ok",
                "node": "synthesizer",
                "n_evidence": len(chunks),
                "draft_chars": len(draft),
                "draft_preview": preview(draft),
            },
        )
        return {
            "draft_answer": draft,
            "evidence": chunks,
            "trace": _trace(state, "synthesizer"),
        }


def critic(state: ResearchState) -> dict:
    chunks = list(state.get("evidence") or [])
    catalog = _format_evidence(chunks) if chunks else "(no evidence)"
    with span("critic", component="generation", n_evidence=len(chunks)) as bag:
        out = invoke_structured(
            CriticOut,
            "Check the draft: are claims supported? citations [n] present? "
            "contradictions? used retrieved evidence only? "
            "If fail, set critic_pass=false and optionally followup_query "
            "and document_id from evidence payloads for one more retrieve.\n\n"
            f"Question: {state['question']}\n\nDraft:\n{state.get('draft_answer')}\n\n"
            f"Evidence:\n{catalog}",
        )
        followup = out.followup_query.strip()
        bag["critic_pass"] = out.critic_pass
        log.info(
            "critic.ok",
            extra={
                "component": "generation",
                "event": "critic.ok",
                "node": "critic",
                "critic_pass": out.critic_pass,
                "critic_notes": preview(out.critic_notes, 240),
                "followup_query": preview(followup),
                "document_id": out.document_id.strip() or None,
                "n_evidence": len(chunks),
            },
        )
        if not out.critic_pass:
            log.warning(
                "critic.fail",
                extra={
                    "component": "generation",
                    "event": "critic.fail",
                    "node": "critic",
                    "critic_notes": preview(out.critic_notes, 240),
                },
            )
        return {
            "critic_pass": out.critic_pass,
            "critic_notes": out.critic_notes,
            "search_queries": [followup] if followup else list(state.get("search_queries") or []),
            "researcher_document_id": out.document_id.strip(),
            "trace": _trace(state, "critic"),
        }


def researcher(state: ResearchState) -> dict:
    seen = list(state.get("seen_chunk_ids") or [])
    extra: list[dict[str, Any]] = []
    queries = [q for q in (state.get("search_queries") or []) if q]
    doc_id = (state.get("researcher_document_id") or "").strip()
    with span(
        "researcher",
        component="retrieval",
        followup_query=preview(queries[0] if queries else ""),
        document_id=doc_id or None,
        n_seen=len(seen),
    ) as bag:
        if queries:
            extra.extend(search_again(queries[0], seen))
        if doc_id:
            extra.extend(get_document(doc_id))
        merged = _merge_unique(list(state.get("retrieved_chunks") or []), extra)
        bag["n_extra"] = len(extra)
        bag["n_chunks"] = len(merged)
        log.info(
            "researcher.ok",
            extra={
                "component": "retrieval",
                "event": "researcher.ok",
                "node": "researcher",
                "n_extra": len(extra),
                "n_chunks": len(merged),
                "document_id": doc_id or None,
            },
        )
        return {
            "retrieved_chunks": merged,
            "evidence": merged,
            "seen_chunk_ids": [_chunk_id(r) for r in merged],
            "critic_retrieve_used": True,
            "researcher_document_id": "",
            "messages": [
                _tool_message(
                    "researcher",
                    f"query={queries[0] if queries else ''} | document_id={doc_id or '-'} | "
                    f"new_hits={_hit_brief(extra)}",
                )
            ],
            "trace": _trace(state, "researcher"),
        }


def finalize(state: ResearchState) -> dict:
    chunks = list(state.get("evidence") or [])
    draft = _as_text(state.get("draft_answer"))
    lines = [draft, "", "Evidence:"]
    for i, row in enumerate(chunks, start=1):
        title = row.get("title") or row.get("source") or "source"
        lines.append(f"[{i}] {title}, p. {row.get('page')}")
    answer = "\n".join(lines).strip()
    log.info(
        "finalize.ok",
        extra={
            "component": "generation",
            "event": "finalize.ok",
            "node": "finalize",
            "n_evidence": len(chunks),
            "critic_pass": state.get("critic_pass"),
            "iteration": state.get("iteration"),
            "path": list(state.get("trace") or []) + ["finalize"],
        },
    )
    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": draft[:500]}],
        "trace": _trace(state, "finalize"),
    }
