"""Classical retrieval metrics. A chunk is relevant when it covers the reference."""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")
_COVER = 0.22


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def is_relevant(chunk_text: str, reference: str, reference_contexts: list[str] | None = None) -> bool:
    needles = [c for c in (reference_contexts or []) if c.strip()] or [reference]
    chunk_tokens = _tokens(chunk_text)
    if not chunk_tokens:
        return False
    lowered = (chunk_text or "").lower()
    for needle in needles:
        needle_tokens = _tokens(needle)
        if needle_tokens and len(needle_tokens & chunk_tokens) / len(needle_tokens) >= _COVER:
            return True
        snippet = " ".join(needle.split())[:80].lower()
        if len(snippet) >= 24 and snippet in lowered:
            return True
    return False


def ir_scores(
    retrieved: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    reference: str,
    reference_contexts: list[str] | None,
    k: int,
) -> dict[str, float]:
    """Precision@k, recall@k, and hit@k (accuracy: at least one relevant chunk in top k)."""
    top = retrieved[:k]
    hits = [1 if is_relevant(row.get("text") or "", reference, reference_contexts) else 0 for row in top]
    n_hit = sum(hits)
    relevant_total = sum(
        1 for row in corpus if is_relevant(row.get("text") or "", reference, reference_contexts)
    )
    precision_at_k = n_hit / k if k else 0.0
    recall_at_k = n_hit / relevant_total if relevant_total else 0.0
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_at_k": 1.0 if n_hit else 0.0,
        "n_relevant_in_corpus": float(relevant_total),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
