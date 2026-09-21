"""Gemini chat + structured JSON (planner / analyzer / critic)."""

from __future__ import annotations

import random
import time
from typing import TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.config import get_settings
from app.embeddings import _is_rate_limited
from app.observability import get_logger, preview, span

log = get_logger("app.llm")

T = TypeVar("T", bound=BaseModel)
_MAX_RETRIES = 8


def get_llm(*, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )


def invoke_structured(schema: type[T], prompt: str) -> T:
    # json_schema avoids Gemini AFC/tool-calling, which warns and is unreliable here.
    llm = get_llm().with_structured_output(schema, method="json_schema")
    delay = 5.0
    last: BaseException | None = None
    with span(
        "llm.structured",
        component="generation",
        schema=schema.__name__,
        prompt_chars=len(prompt),
        model=get_settings().gemini_model,
    ) as bag:
        for attempt in range(_MAX_RETRIES):
            try:
                result = llm.invoke(prompt)
                parsed = result if isinstance(result, schema) else schema.model_validate(result)
                bag["attempts"] = attempt + 1
                return parsed
            except Exception as exc:
                last = exc
                if not _is_rate_limited(exc) or attempt == _MAX_RETRIES - 1:
                    log.error(
                        "llm.fail",
                        extra={
                            "component": "generation",
                            "event": "llm.fail",
                            "schema": schema.__name__,
                            "error_type": type(exc).__name__,
                            "error": preview(str(exc), 240),
                            "attempt": attempt + 1,
                        },
                    )
                    raise
                sleep_for = delay + random.uniform(0, 1.0)
                log.warning(
                    "gemini.429",
                    extra={
                        "component": "generation",
                        "event": "gemini.429",
                        "schema": schema.__name__,
                        "attempt": attempt + 1,
                        "sleep_s": round(sleep_for, 1),
                    },
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, 90.0)
        raise last  # pragma: no cover
