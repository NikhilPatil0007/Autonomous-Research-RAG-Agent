"""Structured logs + in-process spans.

JSON lines are written to logs/agent.jsonl (Loki/Grafana later).
Each run has a trace_id; each span has span_id + parent_span_id + duration_ms.
Do not log API keys or full PDF text.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
_SPAN_ID: ContextVar[str] = ContextVar("span_id", default="")
_PARENT_SPAN_ID: ContextVar[str] = ContextVar("parent_span_id", default="")

_CONFIGURED = False
_SKIP_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
}


def preview(text: str | None, n: int = 160) -> str:
    if not text:
        return ""
    folded = " ".join(str(text).split())
    return folded if len(folded) <= n else folded[: n - 1] + "…"


def current_trace_id() -> str:
    return _TRACE_ID.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None) or _TRACE_ID.get() or None,
            "span_id": getattr(record, "span_id", None) or _SPAN_ID.get() or None,
            "parent_span_id": getattr(record, "parent_span_id", None)
            or _PARENT_SPAN_ID.get()
            or None,
        }
        for key, value in record.__dict__.items():
            if key in payload or key in _SKIP_RECORD_ATTRS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tid = (getattr(record, "trace_id", None) or _TRACE_ID.get() or "-")[:8]
        component = getattr(record, "component", "") or record.name
        extra_bits = []
        for key in (
            "node",
            "route",
            "n_hits",
            "n_chunks",
            "n_new",
            "iteration",
            "duration_ms",
            "schema",
            "status",
            "needs_more_research",
            "critic_pass",
        ):
            if hasattr(record, key):
                extra_bits.append(f"{key}={getattr(record, key)}")
        suffix = (" " + " ".join(extra_bits)) if extra_bits else ""
        return f"{record.levelname:<5} {tid} [{component}] {record.getMessage()}{suffix}"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "trace_id", None):
            record.trace_id = _TRACE_ID.get()
        if not getattr(record, "span_id", None):
            record.span_id = _SPAN_ID.get()
        if not getattr(record, "parent_span_id", None):
            record.parent_span_id = _PARENT_SPAN_ID.get()
        return True


def setup_logging(*, level: str | None = None, log_dir: Path | None = None) -> Path:
    global _CONFIGURED
    root = Path(__file__).resolve().parent.parent
    dest = Path(log_dir or os.getenv("LOG_DIR") or (root / "logs"))
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "agent.jsonl"

    if _CONFIGURED:
        return log_path

    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    ctx = ContextFilter()

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(ctx)

    console = logging.StreamHandler()
    use_json = os.getenv("LOG_JSON", "false").lower() in {"1", "true", "yes"}
    console.setFormatter(JsonFormatter() if use_json else ConsoleFormatter())
    console.addFilter(ctx)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("app.observability").info(
        "logging.ready",
        extra={"component": "observability", "event": "logging.ready", "log_path": str(log_path)},
    )
    return log_path


def start_run(*, kind: str, **attrs: Any) -> str:
    trace_id = uuid.uuid4().hex
    _TRACE_ID.set(trace_id)
    _SPAN_ID.set("")
    _PARENT_SPAN_ID.set("")
    logging.getLogger("app.observability").info(
        "run.start",
        extra={"component": kind, "event": "run.start", "run_kind": kind, **attrs},
    )
    return trace_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def span(name: str, *, component: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    parent = _SPAN_ID.get()
    sid = uuid.uuid4().hex[:16]
    token_span = _SPAN_ID.set(sid)
    token_parent = _PARENT_SPAN_ID.set(parent)
    logger = logging.getLogger(f"app.span.{component}")
    started = time.perf_counter()
    bag: dict[str, Any] = dict(attrs)
    logger.info(
        "span.start",
        extra={"component": component, "event": "span.start", "span_name": name, **bag},
    )
    try:
        yield bag
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "span.end",
            extra={
                "component": component,
                "event": "span.end",
                "span_name": name,
                "status": "ok",
                "duration_ms": duration_ms,
                **bag,
            },
        )
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "span.error",
            extra={
                "component": component,
                "event": "span.error",
                "span_name": name,
                "status": "error",
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
                "error": preview(str(exc), 240),
                **bag,
            },
        )
        raise
    finally:
        _SPAN_ID.reset(token_span)
        _PARENT_SPAN_ID.reset(token_parent)
