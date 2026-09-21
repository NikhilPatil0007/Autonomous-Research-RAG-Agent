from __future__ import annotations

from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.config import get_settings
from app.embeddings import embed_texts
from app.observability import get_logger, preview, span

log = get_logger("app.tools")

PAPER_SEARCH_K = 10
GET_DOCUMENT_CAP = 40
_PAYLOAD_KEYS = ("text", "source", "page", "title", "section", "chunk_id", "document_id")


class QdrantUnavailable(RuntimeError):
    """Local Qdrant is down or unreachable."""


def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, check_compatibility=False)


def ensure_qdrant() -> None:
    """Fail fast with a short message if Docker/Qdrant is not up."""
    settings = get_settings()
    try:
        _client().get_collections()
    except Exception as exc:
        log.error(
            "qdrant.unavailable",
            extra={
                "component": "retrieval",
                "event": "qdrant.unavailable",
                "qdrant_url": settings.qdrant_url,
                "error_type": type(exc).__name__,
                "error": preview(str(exc), 200),
            },
        )
        raise QdrantUnavailable(
            f"Qdrant is not running at {settings.qdrant_url} "
            "(connection refused). Start Docker Desktop, then:\n"
            "  docker compose up -d\n"
            "Wait until http://localhost:6333/dashboard loads, then rerun."
        ) from exc


def _embed_query(query: str) -> list[float]:
    vectors = embed_texts([query], task_type="RETRIEVAL_QUERY")
    return vectors[0]


def _hit_to_dict(hit) -> dict[str, Any]:
    payload = hit.payload or {}
    row = {key: payload.get(key) for key in _PAYLOAD_KEYS}
    row["score"] = float(hit.score) if hit.score is not None else None
    return row


def _point_to_dict(point) -> dict[str, Any]:
    payload = point.payload or {}
    return {key: payload.get(key) for key in _PAYLOAD_KEYS}


def retrieve_chunks(
    query: str,
    k: int = 5,
    *,
    exclude_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Vector search over research_documents."""
    settings = get_settings()
    query_filter = None
    ids = [i for i in (exclude_ids or []) if i]
    with span(
        "retrieve_chunks",
        component="retrieval",
        query=preview(query),
        k=k,
        n_exclude=len(ids),
    ) as bag:
        if ids:
            query_filter = Filter(
                must_not=[FieldCondition(key="chunk_id", match=MatchAny(any=list(ids)))]
            )
        try:
            hits = _client().query_points(
                collection_name=settings.qdrant_collection,
                query=_embed_query(query),
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )
        except (ResponseHandlingException, OSError, ConnectionError) as exc:
            raise QdrantUnavailable(
                f"Qdrant is not running at {settings.qdrant_url}. "
                "Start Docker Desktop, then: docker compose up -d"
            ) from exc
        rows = [_hit_to_dict(h) for h in hits.points]
        scores = [r["score"] for r in rows if r.get("score") is not None]
        bag["n_hits"] = len(rows)
        bag["score_max"] = max(scores) if scores else None
        bag["score_min"] = min(scores) if scores else None
        bag["chunk_ids"] = [r.get("chunk_id") for r in rows]
        if not rows:
            log.warning(
                "retrieve.empty",
                extra={
                    "component": "retrieval",
                    "event": "retrieve.empty",
                    "query": preview(query),
                    "k": k,
                    "n_exclude": len(ids),
                },
            )
        else:
            log.info(
                "retrieve.ok",
                extra={
                    "component": "retrieval",
                    "event": "retrieve.ok",
                    "query": preview(query),
                    "k": k,
                    "n_hits": len(rows),
                    "score_max": bag["score_max"],
                    "score_min": bag["score_min"],
                },
            )
        return rows


def search_research_papers(query: str) -> list[dict[str, Any]]:
    """Same search as retrieve_chunks, slightly higher k for paper-style coverage."""
    return retrieve_chunks(query, k=PAPER_SEARCH_K)


def search_again(query: str, exclude_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Retrieve while excluding already-seen chunk_ids."""
    return retrieve_chunks(query, k=PAPER_SEARCH_K, exclude_ids=exclude_ids)


def get_document(document_id: str, *, cap: int = GET_DOCUMENT_CAP) -> list[dict[str, Any]]:
    """All chunks for one paper, ordered by page, capped."""
    settings = get_settings()
    with span("get_document", component="retrieval", document_id=document_id, cap=cap) as bag:
        points, _offset = _client().scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=cap,
            with_payload=True,
            with_vectors=False,
        )
        rows = [_point_to_dict(p) for p in points]
        rows.sort(key=lambda r: (r.get("page") is None, r.get("page") or 0, r.get("chunk_id") or ""))
        rows = rows[:cap]
        bag["n_chunks"] = len(rows)
        if not rows:
            log.warning(
                "get_document.empty",
                extra={
                    "component": "retrieval",
                    "event": "get_document.empty",
                    "document_id": document_id,
                },
            )
        return rows


def get_document_metadata(document_id: str) -> dict[str, Any] | None:
    """Title, source, and page range for one paper."""
    chunks = get_document(document_id, cap=GET_DOCUMENT_CAP)
    if not chunks:
        return None
    pages = [c["page"] for c in chunks if isinstance(c.get("page"), int)]
    first = chunks[0]
    return {
        "document_id": document_id,
        "title": first.get("title"),
        "source": first.get("source"),
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "n_chunks_loaded": len(chunks),
    }
