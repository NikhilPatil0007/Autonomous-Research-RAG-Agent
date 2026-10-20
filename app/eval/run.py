"""Compare retrieval, BM25 rerank, and chunk size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken

from app.eval.metrics import ir_scores, mean_metrics
from app.observability import get_logger
from app.tools import _bm25_topk, retrieve_chunks, scroll_chunks

log = get_logger("app.eval")
_enc = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZES = (320, 640, 960)
OVERLAP = 80


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if "question" not in row or "reference" not in row:
            raise ValueError(f"Each gold line needs question and reference: {line[:80]}")
        row.setdefault("reference_contexts", [])
        rows.append(row)
    if not rows:
        raise ValueError(f"No questions in {path}")
    return rows


def _rechunk(corpus: list[dict[str, Any]], target_tokens: int) -> list[dict[str, Any]]:
    """Rebuild chunks per document at a new token size. BM25-only comparison."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in corpus:
        grouped.setdefault(str(row.get("document_id") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for doc_id, rows in grouped.items():
        rows.sort(key=lambda r: (r.get("page") is None, r.get("page") or 0, r.get("chunk_id") or ""))
        text = "\n\n".join((r.get("text") or "").strip() for r in rows if (r.get("text") or "").strip())
        tokens = _enc.encode(text)
        start = 0
        idx = 0
        title = rows[0].get("title") if rows else ""
        while start < len(tokens):
            end = min(start + target_tokens, len(tokens))
            piece = _enc.decode(tokens[start:end]).strip()
            if piece:
                out.append(
                    {
                        "text": piece,
                        "chunk_id": f"{doc_id}:t{target_tokens}:c{idx}",
                        "document_id": doc_id,
                        "title": title,
                        "page": None,
                    }
                )
                idx += 1
            if end >= len(tokens):
                break
            start = max(end - OVERLAP, start + 1)
    return out


def _bm25_retrieve(query: str, corpus: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    by_id = {str(row["chunk_id"]): row for row in corpus}
    return [by_id[cid] for cid in _bm25_topk(query, corpus, k)]


def retrieval_configs() -> list[dict[str, Any]]:
    return [
        {"name": "dense", "strategy": "dense", "rerank": False},
        {"name": "bm25", "strategy": "bm25", "rerank": False},
        {"name": "hybrid_rrf", "strategy": "hybrid", "rerank": False},
        {"name": "dense_bm25_rerank", "strategy": "dense", "rerank": True},
    ]


def evaluate_retrieval(gold: list[dict[str, Any]], *, k: int, use_ragas: bool) -> list[dict[str, Any]]:
    corpus = scroll_chunks()
    reports: list[dict[str, Any]] = []
    for config in retrieval_configs():
        per_q: list[dict[str, float]] = []
        ragas_rows: list[dict[str, Any]] = []
        for item in gold:
            hits = retrieve_chunks(
                item["question"],
                k=k,
                strategy=config["strategy"],
                rerank=config["rerank"],
            )
            per_q.append(
                ir_scores(hits, corpus, item["reference"], item.get("reference_contexts"), k)
            )
            if use_ragas:
                from app.eval.ragas_runner import answer_from_contexts

                contexts = [(row.get("text") or "")[:1500] for row in hits]
                ragas_rows.append(
                    {
                        "question": item["question"],
                        "reference": item["reference"],
                        "contexts": contexts,
                        "response": answer_from_contexts(item["question"], contexts),
                    }
                )
        summary = {"name": config["name"], "kind": "retrieval", **mean_metrics(per_q)}
        if use_ragas and ragas_rows:
            from app.eval.ragas_runner import score_samples

            summary.update(score_samples(ragas_rows))
        reports.append(summary)
        log.info("eval.strategy", extra={"component": "retrieval", "event": "eval.strategy", **summary})
    return reports


def evaluate_chunking(gold: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    """BM25 over re-chunked text. Does not re-embed, so dense chunk sizes are not compared here."""
    base = scroll_chunks()
    reports = []
    for size in CHUNK_SIZES:
        corpus = _rechunk(base, size)
        per_q = []
        for item in gold:
            hits = _bm25_retrieve(item["question"], corpus, k)
            per_q.append(
                ir_scores(hits, corpus, item["reference"], item.get("reference_contexts"), k)
            )
        summary = {"name": f"chunk_{size}_bm25", "kind": "chunking", **mean_metrics(per_q)}
        reports.append(summary)
        log.info("eval.chunking", extra={"component": "retrieval", "event": "eval.chunking", **summary})
    return reports


def run_eval(gold_path: Path, *, k: int = 5, use_ragas: bool = True) -> list[dict[str, Any]]:
    gold = load_gold(gold_path)
    reports = evaluate_retrieval(gold, k=k, use_ragas=use_ragas)
    reports.extend(evaluate_chunking(gold, k=k))
    return reports
