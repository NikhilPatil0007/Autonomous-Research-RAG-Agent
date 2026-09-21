"""Walk data/pdfs/**/*.pdf → chunk → embed → upsert Qdrant (idempotent by chunk_id)."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.embeddings import embed_texts, probe_embedding_dim
from app.loaders import Chunk, document_id_for, iter_pdfs, load_pdf_chunks
from app.observability import get_logger, setup_logging, span, start_run

log = get_logger("app.ingest")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(vector_size: int) -> None:
    settings = get_settings()
    client = _client()
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        log.info(
            "qdrant.collection.created",
            extra={
                "component": "ingestion",
                "event": "qdrant.collection.created",
                "collection": name,
                "vector_size": vector_size,
            },
        )
        print(f"Created collection {name!r} size={vector_size}")
        return

    info = client.get_collection(name)
    existing = info.config.params.vectors
    size = existing.size if hasattr(existing, "size") else existing["size"]
    if int(size) != int(vector_size):
        raise SystemExit(
            f"Collection {name!r} has vector size {size}, but probe/config is {vector_size}. "
            f"Set EMBEDDING_DIM to {size} or recreate the collection."
        )
    print(f"Using existing collection {name!r} size={size}")
    log.info(
        "qdrant.collection.reuse",
        extra={
            "component": "ingestion",
            "event": "qdrant.collection.reuse",
            "collection": name,
            "vector_size": int(size),
        },
    )


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    settings = get_settings()
    points = [
        PointStruct(
            id=_point_id(chunk.chunk_id),
            vector=vec,
            payload=chunk.payload(),
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client = _client()
    batch = 64
    for i in range(0, len(points), batch):
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points[i : i + batch],
            wait=True,
        )


def write_manifest(pdf_path: Path, chunks: list[Chunk]) -> None:
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.processed_dir / f"{chunks[0].document_id}.json"
    dest.write_text(
        json.dumps(
            {
                "source": chunks[0].source,
                "title": chunks[0].title,
                "document_id": chunks[0].document_id,
                "pdf": str(pdf_path),
                "n_chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "chunk_ids": [c.chunk_id for c in chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def ingest_one(pdf_path: Path, *, skip_existing: bool = True) -> int:
    print(f"\n==> {pdf_path}")
    settings = get_settings()
    manifest = settings.processed_dir / f"{document_id_for(pdf_path)}.json"
    if skip_existing and manifest.exists():
        log.info(
            "ingest.skip",
            extra={
                "component": "ingestion",
                "event": "ingest.skip",
                "source": str(pdf_path),
                "manifest": manifest.name,
            },
        )
        print(f"    skip: already ingested ({manifest.name})")
        return 0
    chunks = load_pdf_chunks(pdf_path)
    if not chunks:
        log.warning(
            "ingest.empty_pdf",
            extra={"component": "ingestion", "event": "ingest.empty_pdf", "source": str(pdf_path)},
        )
        print("    skip: no text extracted")
        return 0
    print(f"    chunks={len(chunks)} title={chunks[0].title!r}")
    with span(
        "ingest_one",
        component="ingestion",
        source=str(pdf_path),
        n_chunks=len(chunks),
        document_id=chunks[0].document_id,
    ):
        vectors = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
        upsert_chunks(chunks, vectors)
        write_manifest(pdf_path, chunks)
    log.info(
        "ingest.ok",
        extra={
            "component": "ingestion",
            "event": "ingest.ok",
            "source": str(pdf_path),
            "document_id": chunks[0].document_id,
            "n_chunks": len(chunks),
        },
    )
    return len(chunks)


def main() -> None:
    settings = get_settings()
    log_path = setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    start_run(kind="ingestion", pdf_dir=str(settings.pdf_dir), log_path=str(log_path))
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    pdfs = iter_pdfs()
    if not pdfs:
        raise SystemExit(f"No PDFs under {settings.pdf_dir}")
    client = _client()
    name = settings.qdrant_collection
    if client.collection_exists(name):
        info = client.get_collection(name)
        existing = info.config.params.vectors
        dim = int(existing.size if hasattr(existing, "size") else existing["size"])
        print(f"Reusing collection {name!r} size={dim} (skip embedding probe)")
    else:
        try:
            dim = probe_embedding_dim()
        except Exception as exc:
            raise SystemExit(f"Embedding probe failed (is GEMINI_API_KEY set?): {exc}") from exc
        if dim != settings.embedding_dim:
            print(
                f"Warning: probe dim={dim}, EMBEDDING_DIM={settings.embedding_dim}; using probe dim"
            )
        print(f"Embedding dim={dim} model={settings.embedding_model}")
    try:
        ensure_collection(dim)
    except UnexpectedResponse as exc:
        raise SystemExit(f"Qdrant error at {settings.qdrant_url}: {exc}") from exc
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Qdrant at {settings.qdrant_url}. "
            f"Start it with: docker compose up -d\n{exc}"
        ) from exc
    total = 0
    for pdf in pdfs:
        total += ingest_one(pdf)
    print(f"\nDone. upserted {total} chunks into {settings.qdrant_collection!r}")
    print(f"Dashboard: {settings.qdrant_url.rstrip('/')}/dashboard")
    log.info(
        "ingest.done",
        extra={
            "component": "ingestion",
            "event": "ingest.done",
            "n_pdfs": len(pdfs),
            "n_chunks_upserted": total,
            "collection": settings.qdrant_collection,
        },
    )


if __name__ == "__main__":
    main()
