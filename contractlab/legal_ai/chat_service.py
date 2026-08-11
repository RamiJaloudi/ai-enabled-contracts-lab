from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from contractlab.legal_ai.chat_artifacts import (
    build_chat_response_docx,
    response_artifact_stem,
    response_text_bytes,
)
from contractlab.legal_ai.contract_artifacts import (
    build_contract_docx,
    contract_text_bytes,
    safe_contract_filename,
)
from contractlab.legal_ai.deterministic_drafting import deterministic_contract_draft
from contractlab.legal_ai.llm import LLMError
from contractlab.legal_ai.orchestrator import LegalAnalysisOrchestrator
from contractlab.legal_ai.schemas import ContractDraft

FULL_CONTRACT_DRAFT_PATTERN = re.compile(
    r"\b(?:create|draft|generate|prepare|redraft|write)\b.{0,80}"
    r"\b(?:agreement|contract|nda|non-disclosure|terms of service|policy)\b|"
    r"\b(?:complete|full)\s+(?:revised\s+)?(?:agreement|contract|draft)\b",
    re.IGNORECASE,
)

DOCUMENT_CITATION_GLYPH_PATTERN = re.compile(r"[\u3010\uff3b]\s*(D\d+)\s*[\u3011\uff3d]", re.IGNORECASE)


def normalize_document_citations(content: str) -> str:
    """Normalize known full-width model citation glyphs to stable ASCII document IDs."""
    return DOCUMENT_CITATION_GLYPH_PATTERN.sub(lambda match: f"[{match.group(1).upper()}]", content)


@dataclass(frozen=True, slots=True)
class ChatArtifact:
    filename: str
    media_type: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ResilientChatResult:
    content: str
    provider: str
    model: str
    used_model_fallback: bool = False
    used_local_fallback: bool = False
    error: str | None = None
    artifacts: tuple[ChatArtifact, ...] = ()

    @property
    def source_label(self) -> str:
        if self.used_local_fallback:
            return "Local deterministic fallback"
        suffix = " (fallback model)" if self.used_model_fallback else ""
        return f"{self.provider}: {self.model}{suffix}"


def normalize_chat_history(
    messages: list[dict[str, Any]], *, limit: int = 10
) -> list[dict[str, str]]:
    history = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history[-limit:]


def is_full_contract_draft_request(question: str) -> bool:
    return bool(FULL_CONTRACT_DRAFT_PATTERN.search(question))


def _draft_artifacts(draft: ContractDraft) -> tuple[ChatArtifact, ChatArtifact]:
    stem = safe_contract_filename(draft.title)
    return (
        ChatArtifact(
            filename=f"{stem}.docx",
            media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            data=build_contract_docx(draft),
        ),
        ChatArtifact(
            filename=f"{stem}.txt",
            media_type="text/plain",
            data=contract_text_bytes(draft),
        ),
    )


def _response_artifacts(
    content: str,
    *,
    title: str = "Contract chat response",
    sources: list[str] | None = None,
) -> tuple[ChatArtifact, ChatArtifact]:
    stem = response_artifact_stem(title)
    return (
        ChatArtifact(
            filename=f"{stem}.docx",
            media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            data=build_chat_response_docx(title, content, sources),
        ),
        ChatArtifact(
            filename=f"{stem}.txt",
            media_type="text/plain",
            data=response_text_bytes(title, content, sources),
        ),
    )


def answer_contract_question(
    orchestrator: LegalAnalysisOrchestrator,
    *,
    contract_text: str,
    analysis_result: dict[str, Any],
    question: str,
    history: list[dict[str, str]],
    fallback: Callable[[str, dict[str, Any]], str],
    draft_mode: Literal["auto", "review", "draft"] = "auto",
) -> ResilientChatResult:
    if draft_mode not in {"auto", "review", "draft"}:
        raise ValueError(f"Unknown contract chat mode: {draft_mode}")
    should_draft = draft_mode == "draft" or (
        draft_mode == "auto" and is_full_contract_draft_request(question)
    )
    try:
        if should_draft:
            generated = orchestrator.draft_contract(
                request=question,
                contract_text=contract_text,
                analysis_result=analysis_result,
                history=history,
            )
            return ResilientChatResult(
                content=generated.draft.as_text(),
                provider=generated.provider,
                model=generated.model,
                used_model_fallback=generated.used_fallback,
                artifacts=_draft_artifacts(generated.draft),
            )
        result = orchestrator.contract_chat(
            contract_text=contract_text,
            analysis_result=analysis_result,
            question=question,
            history=history,
        )
    except LLMError as exc:
        if should_draft:
            draft = deterministic_contract_draft(
                contract_text=contract_text,
                analysis_result=analysis_result,
                request=question,
            )
            return ResilientChatResult(
                content=draft.as_text(),
                provider="local",
                model="deterministic-contract-drafter",
                used_local_fallback=True,
                error=str(exc),
                artifacts=_draft_artifacts(draft),
            )
        return ResilientChatResult(
            content=fallback(question, analysis_result),
            provider="local",
            model="deterministic-contract-chat",
            used_local_fallback=True,
            error=str(exc),
        )
    return ResilientChatResult(
        content=result.content,
        provider=result.provider,
        model=result.model,
        used_model_fallback=result.used_fallback,
    )


def answer_contract_workspace_question(
    orchestrator: LegalAnalysisOrchestrator,
    *,
    documents: list[dict[str, str]],
    question: str,
    history: list[dict[str, str]],
    analysis_result: dict[str, Any] | None = None,
    fallback: Callable[[str, dict[str, Any]], str] | None = None,
    draft_mode: Literal["auto", "review", "draft"] = "auto",
) -> ResilientChatResult:
    if draft_mode not in {"auto", "review", "draft"}:
        raise ValueError(f"Unknown contract chat mode: {draft_mode}")
    contract_text = "\n\n".join(
        f"[{row.get('id', f'D{index}')} | {row.get('filename', 'document')}]\n{row.get('text', '')}"
        for index, row in enumerate(documents, start=1)
    )
    should_draft = draft_mode == "draft" or (
        draft_mode == "auto" and is_full_contract_draft_request(question)
    )
    source_names = [
        f"{row.get('id', f'D{index}')}: {row.get('filename', 'document')}"
        for index, row in enumerate(documents, start=1)
    ]
    try:
        if should_draft:
            generated = orchestrator.draft_contract(
                request=question,
                contract_text=contract_text,
                analysis_result=analysis_result or {},
                history=history,
            )
            return ResilientChatResult(
                content=generated.draft.as_text(),
                provider=generated.provider,
                model=generated.model,
                used_model_fallback=generated.used_fallback,
                artifacts=_draft_artifacts(generated.draft),
            )
        result = orchestrator.contract_workspace_chat(
            documents=documents,
            question=question,
            history=history,
            analysis_result=analysis_result,
        )
    except LLMError as exc:
        if should_draft:
            draft = deterministic_contract_draft(
                contract_text=contract_text,
                analysis_result=analysis_result or {},
                request=question,
            )
            return ResilientChatResult(
                content=draft.as_text(),
                provider="local",
                model="deterministic-contract-drafter",
                used_local_fallback=True,
                error=str(exc),
                artifacts=_draft_artifacts(draft),
            )
        if documents and fallback is not None:
            content = fallback(question, analysis_result or {})
        else:
            content = (
                "The configured generative model is unavailable, so I cannot safely answer this "
                "general contract question without inventing information. Start the local model "
                "and try again."
            )
        content = normalize_document_citations(content)
        return ResilientChatResult(
            content=content,
            provider="local",
            model="deterministic-contract-chat",
            used_local_fallback=True,
            error=str(exc),
            artifacts=_response_artifacts(content, sources=source_names),
        )
    content = normalize_document_citations(result.content)
    return ResilientChatResult(
        content=content,
        provider=result.provider,
        model=result.model,
        used_model_fallback=result.used_fallback,
        artifacts=_response_artifacts(content, sources=source_names),
    )


def answer_evaluation_question(
    orchestrator: LegalAnalysisOrchestrator,
    *,
    summary: dict[str, Any],
    question: str,
    history: list[dict[str, str]],
    fallback_content: str,
) -> ResilientChatResult:
    try:
        result = orchestrator.evaluation_chat(
            summary=summary,
            question=question,
            history=history,
        )
    except LLMError as exc:
        return ResilientChatResult(
            content=fallback_content,
            provider="local",
            model="deterministic-evaluation-chat",
            used_local_fallback=True,
            error=str(exc),
        )
    return ResilientChatResult(
        content=result.content,
        provider=result.provider,
        model=result.model,
        used_model_fallback=result.used_fallback,
    )
