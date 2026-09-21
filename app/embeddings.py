"""Gemini embeddings with configurable output dim, batching, and 429 backoff."""

from __future__ import annotations

import random
import time
from typing import Sequence

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from app.config import get_settings
from app.observability import get_logger, span

log = get_logger("app.embeddings")

_MAX_RETRIES = 10
_BATCH_SIZE = 4
_BATCH_PAUSE_S = 2.0
_INITIAL_BACKOFF_S = 5.0
_MAX_BACKOFF_S = 90.0


def _client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.gemini_api_key)


def _is_rate_limited(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, "429"):
        return True
    if isinstance(exc, ClientError) and getattr(exc, "status", "") == "RESOURCE_EXHAUSTED":
        return True
    text = str(exc).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "resource exhausted" in text
        or "503" in text
        or "unavailable" in text
        or "high demand" in text
    )


def _with_backoff(fn):
    delay = _INITIAL_BACKOFF_S
    last: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry only rate limits
            last = exc
            if not _is_rate_limited(exc) or attempt == _MAX_RETRIES - 1:
                raise
            sleep_for = delay + random.uniform(0, 1.0)
            log.warning(
                "gemini.429",
                extra={
                    "component": "generation",
                    "event": "gemini.429",
                    "attempt": attempt + 1,
                    "max_retries": _MAX_RETRIES,
                    "sleep_s": round(sleep_for, 1),
                },
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, _MAX_BACKOFF_S)
    raise last  # pragma: no cover


def embed_texts(
    texts: Sequence[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed texts in small batches. Empty strings become zero vectors of configured dim."""
    settings = get_settings()
    client = _client()
    dim = settings.embedding_dim
    out: list[list[float]] = []
    with span(
        "embed_texts",
        component="ingestion" if task_type == "RETRIEVAL_DOCUMENT" else "retrieval",
        task_type=task_type,
        n_texts=len(texts),
        model=settings.embedding_model,
        dim=dim,
    ) as bag:
        n_zero = 0
        for i in range(0, len(texts), _BATCH_SIZE):
            if i > 0:
                time.sleep(_BATCH_PAUSE_S)
            batch = list(texts[i : i + _BATCH_SIZE])
            nonempty = [(j, t) for j, t in enumerate(batch) if t.strip()]
            vectors: list[list[float] | None] = [None] * len(batch)

            if nonempty:
                contents = [t for _, t in nonempty]
                log.info(
                    "embed.batch",
                    extra={
                        "component": bag.get("component") or "retrieval",
                        "event": "embed.batch",
                        "batch_start": i + 1,
                        "batch_end": i + len(batch),
                        "n_total": len(texts),
                        "n_in_batch": len(contents),
                        "task_type": task_type,
                    },
                )

                def _call(contents=contents):
                    return client.models.embed_content(
                        model=settings.embedding_model,
                        contents=contents,
                        config=types.EmbedContentConfig(
                            output_dimensionality=dim,
                            task_type=task_type,
                        ),
                    )

                result = _with_backoff(_call)
                embeddings = result.embeddings or []
                if len(embeddings) != len(nonempty):
                    raise RuntimeError(
                        f"Embedding count mismatch: got {len(embeddings)}, expected {len(nonempty)}"
                    )
                for (j, _), emb in zip(nonempty, embeddings):
                    values = list(emb.values or [])
                    if len(values) != dim:
                        raise RuntimeError(
                            f"Embedding dim {len(values)} != configured EMBEDDING_DIM={dim}"
                        )
                    vectors[j] = values

            for vec in vectors:
                if vec is None:
                    n_zero += 1
                    out.append([0.0] * dim)
                else:
                    out.append(vec)
        bag["n_vectors"] = len(out)
        bag["n_zero_vectors"] = n_zero
        return out


def probe_embedding_dim() -> int:
    """One embed to confirm the live vector size."""
    vecs = embed_texts(["dimension probe"], task_type="RETRIEVAL_DOCUMENT")
    return len(vecs[0])
