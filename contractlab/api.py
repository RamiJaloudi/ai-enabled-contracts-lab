from __future__ import annotations

import asyncio
import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from contractlab.assistant import analyze_contract, contract_chat_response, heuristic_prediction
from contractlab.inference import ClausePredictor
from contractlab.legal_ai import LegalAnalysisOrchestrator, OllamaClient, load_legal_ai_settings
from contractlab.legal_ai.chat_service import answer_contract_question
from contractlab.legal_ai.document_pipeline import (
    aggregate_passage_predictions,
    classify_passages,
    evidence_payload,
    segment_document,
)
from contractlab.legal_ai.llm import LLMError
from contractlab.legal_ai.routing import build_classification_route
from contractlab.legal_ai.security import validate_input_text
from contractlab.paths import PROJECT_ROOT

app = FastAPI(
    title="AI-Enabled Contracts Lab API",
    version="0.2.0",
    description="Local classifier and generative legal-analysis API. Educational use only.",
)

_REQUEST_SEMAPHORE = asyncio.Semaphore(
    load_legal_ai_settings().maximum_concurrent_requests
)


@app.middleware("http")
async def protect_local_api(request: Request, call_next):
    settings = load_legal_ai_settings()
    expected_key = os.getenv("CONTRACTLAB_API_KEY")
    if expected_key and request.headers.get("x-contractlab-api-key") != expected_key:
        return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            too_large = int(content_length) > settings.maximum_request_characters * 4
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length header."}, status_code=400)
        if too_large:
            return JSONResponse(
                {"detail": "Request body exceeds the configured limit."}, status_code=413
            )
    async with _REQUEST_SEMAPHORE:
        return await call_next(request)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)
    checkpoint: str | None = None
    domain_checkpoint: str | None = None
    doctrine_checkpoint: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeRequest(ClassifyRequest):
    jurisdiction: str | None = None


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)
    question: str = Field(min_length=1)
    checkpoint: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    mode: Literal["auto", "review", "draft"] = "auto"


def _safe_checkpoint(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "artifacts" / "runs").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise HTTPException(400, "Checkpoint must be inside artifacts/runs.")
    if not resolved.exists():
        raise HTTPException(404, "Checkpoint not found.")
    return resolved


@lru_cache(maxsize=4)
def _predictor(checkpoint: str) -> ClausePredictor:
    return ClausePredictor(checkpoint)


def _heuristic_batch(texts: list[str]) -> list[dict[str, Any]]:
    return [heuristic_prediction(text) for text in texts]


def _domain_predictions(request: ClassifyRequest) -> list[dict[str, Any]]:
    if request.domain_checkpoint:
        checkpoint = _safe_checkpoint(request.domain_checkpoint)
        return _predictor(str(checkpoint)).predict_top_k(
            [request.text], top_k=request.top_k, prediction_type="domain"
        )[0]
    return [
        {
            "label": "Contract Law",
            "confidence": 0.55,
            "model_version": "heuristic-contract-domain-v1",
            "taxonomy_version": "contract-domain-v1",
            "prediction_type": "domain",
            "rank": 1,
            "calibrated": False,
            "calibration_version": "uncalibrated",
        }
    ]


def _doctrine_predictions(request: ClassifyRequest) -> list[dict[str, Any]]:
    checkpoint_value = request.doctrine_checkpoint or request.checkpoint
    if checkpoint_value:
        checkpoint = _safe_checkpoint(checkpoint_value)
        return _predictor(str(checkpoint)).predict_top_k(
            [request.text], top_k=request.top_k, prediction_type="doctrine"
        )[0]
    prediction = heuristic_prediction(request.text)
    return [
        {
            **prediction,
            "prediction_type": "doctrine",
            "rank": 1,
            "calibrated": False,
            "calibration_version": "uncalibrated",
        }
    ]


def _classification_bundle(request: ClassifyRequest) -> dict[str, Any]:
    settings = load_legal_ai_settings()
    validate_input_text(request.text, maximum_characters=settings.maximum_request_characters)
    domains = _domain_predictions(request)
    doctrines = _doctrine_predictions(request)
    route = build_classification_route(
        domain_predictions=domains,
        doctrine_predictions=doctrines,
        domain_threshold=settings.domain_threshold,
        doctrine_review_threshold=settings.doctrine_review_threshold,
        auto_route_threshold=settings.auto_route_threshold,
        minimum_margin=settings.minimum_margin,
    )
    return {
        "domain_predictions": domains,
        "doctrine_predictions": doctrines,
        "route": route,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    settings = load_legal_ai_settings()
    client = OllamaClient(base_url=settings.base_url, model=settings.model, timeout_seconds=5)
    return {
        "status": "ok",
        "ollama_available": client.is_available(),
        "generative_model": settings.model,
        "rag_enabled": settings.rag_enabled,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    settings = load_legal_ai_settings()
    client = OllamaClient(base_url=settings.base_url, model=settings.model, timeout_seconds=10)
    try:
        installed = client.list_models()
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"provider": settings.provider, "selected": settings.model, "installed": installed}


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict[str, Any]:
    try:
        bundle = _classification_bundle(request)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "domain_predictions": bundle["domain_predictions"],
        "doctrine_predictions": bundle["doctrine_predictions"],
        "route": bundle["route"].model_dump(),
        "predictions_are_hypotheses": True,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        bundle = _classification_bundle(request)
        result = LegalAnalysisOrchestrator.from_settings().analyze(
            text=request.text,
            predictions=bundle["doctrine_predictions"],
            domain_predictions=bundle["domain_predictions"],
            classification_route=bundle["route"],
            jurisdiction=request.jurisdiction,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    return result.model_dump()


@app.post("/analyze-document")
def analyze_document(request: AnalyzeRequest) -> dict[str, Any]:
    settings = load_legal_ai_settings()
    try:
        validate_input_text(
            request.text,
            maximum_characters=settings.maximum_request_characters,
        )
        passages, truncated = segment_document(
            request.text,
            maximum_passage_characters=settings.maximum_passage_characters,
            overlap_characters=settings.passage_overlap_characters,
            maximum_passages=settings.maximum_passages,
        )
        checkpoint_value = request.doctrine_checkpoint or request.checkpoint
        passage_predictor: Any
        if checkpoint_value:
            passage_predictor = _predictor(str(_safe_checkpoint(checkpoint_value)))
        else:
            passage_predictor = _heuristic_batch
        passage_rows = classify_passages(
            passages, passage_predictor, top_k=request.top_k
        )
        doctrines = aggregate_passage_predictions(passage_rows, top_k=request.top_k)
        domains = _domain_predictions(request)
        route = build_classification_route(
            domain_predictions=domains,
            doctrine_predictions=doctrines,
            domain_threshold=settings.domain_threshold,
            doctrine_review_threshold=settings.doctrine_review_threshold,
            auto_route_threshold=settings.auto_route_threshold,
            minimum_margin=settings.minimum_margin,
        )
        evidence = evidence_payload(passages, doctrines)
        result = LegalAnalysisOrchestrator.from_settings().analyze(
            text=request.text,
            predictions=[row.model_dump() for row in doctrines],
            domain_predictions=domains,
            classification_route=route,
            jurisdiction=request.jurisdiction,
            evidence_passages=evidence,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        **result.model_dump(),
        "document": {
            "passage_count": len(passages),
            "passages_truncated": truncated,
            "passages": [row.model_dump() for row in passages],
        },
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    predictor = _predictor(str(_safe_checkpoint(request.checkpoint))) if request.checkpoint else None
    analysis_result = analyze_contract(request.text, predictor)
    orchestrator = LegalAnalysisOrchestrator.from_settings()
    reply = answer_contract_question(
        orchestrator,
        contract_text=request.text,
        analysis_result=analysis_result,
        question=request.question,
        history=request.history,
        fallback=contract_chat_response,
        draft_mode=request.mode,
    )
    artifacts = [
        {
            "filename": artifact.filename,
            "media_type": artifact.media_type,
            "data_base64": base64.b64encode(artifact.data).decode("ascii"),
        }
        for artifact in reply.artifacts
    ]
    return {
        "content": reply.content,
        "model": reply.model,
        "provider": reply.provider,
        "used_model_fallback": reply.used_model_fallback,
        "used_local_fallback": reply.used_local_fallback,
        "error": reply.error,
        "source_label": reply.source_label,
        "contract_profile": analysis_result["contract_profile"],
        "artifacts": artifacts,
    }
