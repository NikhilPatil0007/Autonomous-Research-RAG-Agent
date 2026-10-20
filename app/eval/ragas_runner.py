"""RAGAS scores for one strategy: context precision/recall, faithfulness, answer relevancy."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.llm import get_llm
from app.observability import get_logger

log = get_logger("app.eval.ragas")


def answer_from_contexts(question: str, contexts: list[str]) -> str:
    """Short grounded answer so faithfulness measures the retriever, not the full graph."""
    blob = "\n\n".join(contexts) or "(no context)"
    msg = get_llm(temperature=0.0).invoke(
        "Answer using ONLY the context. If the context is insufficient, say so. "
        "Do not invent papers.\n\n"
        f"Question: {question}\n\nContext:\n{blob}"
    )
    content = getattr(msg, "content", None) or str(msg)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(p for p in parts if p)
    return str(content)


def score_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Mean RAGAS metrics. Each sample needs question, reference, contexts, response."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    settings = get_settings()
    evaluator_llm = LangchainLLMWrapper(get_llm(temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
    )
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                reference=row["reference"],
                retrieved_contexts=row["contexts"] or ["(empty)"],
                response=row["response"],
            )
            for row in samples
        ]
    )
    metrics = [
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings),
    ]
    result = evaluate(dataset=dataset, metrics=metrics)
    frame = result.to_pandas()
    means: dict[str, float] = {}
    for column in frame.columns:
        if column in {"user_input", "retrieved_contexts", "response", "reference"}:
            continue
        series = frame[column].dropna()
        if len(series):
            means[column] = float(series.mean())
    log.info(
        "ragas.done",
        extra={"component": "retrieval", "event": "ragas.done", "metrics": means, "n": len(samples)},
    )
    return means
