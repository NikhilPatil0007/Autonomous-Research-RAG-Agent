"""Qdrant retrieval: dense, BM25, or hybrid RRF. Graph nodes call these; they do not pick tools."""

from __future__ import annotations

import re
from typing import Any, Literal, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embeddings import embed_texts
from app.observability import get_logger, preview, span

log = get_logger("app.tools")

RetrievalStrategy = Literal["dense", "bm25", "hybrid"]

PAPER_SEARCH_K = 10
GET_DOCUMENT_CAP = 40
RRF_K = 60
CANDIDATE_MULT = 4
_TOKEN = re.compile(r"[a-z0-9]+")
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


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def scroll_chunks(exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """All chunk payloads, minus ids already seen."""
    settings = get_settings()
    skipped = exclude or set()
    rows: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = _client().scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            row = _point_to_dict(point)
            cid = str(row.get("chunk_id") or "")
            if cid and cid not in skipped and (row.get("text") or "").strip():
                rows.append(row)
        if offset is None:
            break
    return rows


def _bm25_topk(query: str, corpus: list[dict[str, Any]], n: int) -> list[str]:
    if not corpus or n <= 0:
        return []
    tokenized = [_tokens(row.get("text") or "") for row in corpus]
    if not any(tokenized):
        return []
    scores = BM25Okapi(tokenized).get_scores(_tokens(query))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    ids: list[str] = []
    for i in order:
        if float(scores[i]) <= 0:
            break
        ids.append(str(corpus[i]["chunk_id"]))
        if len(ids) >= n:
            break
    return ids


def _rrf_fuse(*ranked_id_lists: list[str]) -> list[tuple[str, float]]:
    """score(d) = sum 1 / (RRF_K + rank). Rank is 1-based."""
    scores: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _dense_rows(query: str, k: int, exclude: set[str]) -> list[dict[str, Any]]:
    settings = get_settings()
    query_filter = None
    if exclude:
        query_filter = Filter(
            must_not=[FieldCondition(key="chunk_id", match=MatchAny(any=list(exclude)))]
        )
    hits = _client().query_points(
        collection_name=settings.qdrant_collection,
        query=_embed_query(query),
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )
    rows = []
    for hit in hits.points:
        row = _hit_to_dict(hit)
        cid = str(row.get("chunk_id") or "")
        if cid and cid not in exclude:
            row["sources"] = ["dense"]
            rows.append(row)
    return rows


def rerank_bm25(query: str, candidates: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Reorder a dense candidate pool with BM25 (retrieve-then-rerank)."""
    ids = _bm25_topk(query, candidates, k)
    by_id = {str(row["chunk_id"]): row for row in candidates}
    rows = []
    for cid in ids:
        row = dict(by_id[cid])
        row["sources"] = ["dense", "bm25_rerank"]
        rows.append(row)
    return rows


def retrieve_chunks(
    query: str,
    k: int = 5,
    *,
    exclude_ids: Sequence[str] | None = None,
    strategy: RetrievalStrategy = "hybrid",
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """Dense, BM25, or hybrid RRF. Optional BM25 rerank of the dense pool."""
    settings = get_settings()
    exclude = {i for i in (exclude_ids or []) if i}
    pool = max(k * CANDIDATE_MULT, k)
    with span(
        "retrieve_chunks",
        component="retrieval",
        query=preview(query),
        k=k,
        n_exclude=len(exclude),
        strategy=strategy,
        rerank=rerank,
    ) as bag:
        try:
            dense_rows: list[dict[str, Any]] = []
            corpus: list[dict[str, Any]] = []
            if strategy in {"dense", "hybrid"} or rerank:
                dense_rows = _dense_rows(query, pool if strategy != "dense" or rerank else k, exclude)
            if strategy in {"bm25", "hybrid"} or rerank:
                corpus = scroll_chunks(exclude)
        except (ResponseHandlingException, OSError, ConnectionError) as exc:
            raise QdrantUnavailable(
                f"Qdrant is not running at {settings.qdrant_url}. "
                "Start Docker Desktop, then: docker compose up -d"
            ) from exc

        by_id: dict[str, dict[str, Any]] = {str(row["chunk_id"]): row for row in corpus}
        dense_ids: list[str] = []
        for row in dense_rows:
            cid = str(row["chunk_id"])
            dense_ids.append(cid)
            by_id[cid] = row

        if rerank:
            rows = rerank_bm25(query, dense_rows, k)
        elif strategy == "dense":
            rows = dense_rows[:k]
        elif strategy == "bm25":
            rows = []
            for cid in _bm25_topk(query, corpus, k):
                row = dict(by_id[cid])
                row["sources"] = ["bm25"]
                rows.append(row)
        else:
            bm25_ids = _bm25_topk(query, corpus, pool)
            dense_set, bm25_set = set(dense_ids), set(bm25_ids)
            rows = []
            for cid, rrf_score in _rrf_fuse(dense_ids, bm25_ids)[:k]:
                row = dict(by_id[cid])
                sources = []
                if cid in dense_set:
                    sources.append("dense")
                if cid in bm25_set:
                    sources.append("bm25")
                row["score"] = rrf_score
                row["sources"] = sources
                rows.append(row)

        bag["n_hits"] = len(rows)
        bag["chunk_ids"] = [r.get("chunk_id") for r in rows]
        log.info(
            "retrieve.ok",
            extra={
                "component": "retrieval",
                "event": "retrieve.ok",
                "strategy": strategy,
                "rerank": rerank,
                "query": preview(query),
                "k": k,
                "n_hits": len(rows),
            },
        )
        if not rows:
            log.warning(
                "retrieve.empty",
                extra={
                    "component": "retrieval",
                    "event": "retrieve.empty",
                    "strategy": strategy,
                    "query": preview(query),
                    "k": k,
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
