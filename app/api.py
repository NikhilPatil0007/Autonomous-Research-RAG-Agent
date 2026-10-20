"""Streaming research API.

  python -m uvicorn app.api:app --host 127.0.0.1 --port 8000

POST /research returns Server-Sent Events, one JSON object per graph node.
The LangGraph run stays synchronous and runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.main import iter_research
from app.observability import get_logger, setup_logging
from app.tools import QdrantUnavailable, ensure_qdrant

log = get_logger("app.api")


class _Gate:
    """Rejects new runs once MAX_CONCURRENT_RESEARCH are already in flight."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._active = 0
        self._lock = asyncio.Lock()

    async def try_enter(self) -> bool:
        async with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    async def leave(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


_gate: _Gate | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _gate
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    _gate = _Gate(settings.max_concurrent_research)
    log.info(
        "api.start",
        extra={
            "component": "graph",
            "event": "api.start",
            "max_concurrent_research": _gate._limit,
        },
    )
    yield
    _gate = None


app = FastAPI(title="Autonomous Research RAG Agent", lifespan=_lifespan)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str = "default"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health", response_model=None)
def health() -> JSONResponse | dict[str, Any]:
    try:
        ensure_qdrant()
    except QdrantUnavailable as exc:
        return JSONResponse({"ok": False, "qdrant": str(exc)}, status_code=503)
    settings = get_settings()
    return {"ok": True, "qdrant": settings.qdrant_url, "collection": settings.qdrant_collection}


@app.post("/research")
async def research(body: ResearchRequest) -> StreamingResponse:
    gate = _gate
    if gate is None:
        raise HTTPException(status_code=503, detail="Server is still starting")
    if not await gate.try_enter():
        raise HTTPException(
            status_code=503,
            detail="Research concurrency cap reached. Retry when a run finishes.",
        )

    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def produce() -> None:
            try:
                for event in iter_research(body.question, thread_id=body.thread_id):
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", f"{type(exc).__name__}: {exc}"))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

        worker = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    break
                if kind == "error":
                    yield _sse({"node": "error", "error": payload})
                    break
                yield _sse(payload)
            await worker
        finally:
            await gate.leave()

    return StreamingResponse(events(), media_type="text/event-stream")
