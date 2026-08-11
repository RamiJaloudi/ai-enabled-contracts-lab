from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from contractlab.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "legal_ai.json"
PREFERENCE_PATH = PROJECT_ROOT / "artifacts" / "preferences" / "legal_ai.json"


@dataclass(frozen=True, slots=True)
class LegalAISettings:
    top_k: int
    domain_threshold: float
    doctrine_review_threshold: float
    auto_route_threshold: float
    minimum_margin: float
    classifier_max_length: int
    provider: str
    base_url: str
    model: str
    fallback_model: str
    temperature: float
    drafting_temperature: float
    max_output_tokens: int
    context_window: int
    max_context_characters: int
    rag_enabled: bool
    prompt_paths: dict[str, Path]
    schema_path: Path
    config_path: Path
    maximum_request_characters: int
    maximum_passage_characters: int
    passage_overlap_characters: int
    maximum_passages: int
    maximum_history_messages: int
    maximum_history_characters: int
    maximum_concurrent_requests: int
    contract_chat_storage_path: Path
    maximum_files_per_message: int
    maximum_file_bytes: int
    maximum_archive_entries: int
    maximum_archive_uncompressed_bytes: int
    maximum_documents_per_chat: int
    maximum_spreadsheet_cells: int
    maximum_ocr_pages: int
    maximum_extracted_characters_per_document: int


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=4)
def load_legal_ai_settings(path: Path | None = None) -> LegalAISettings:
    target = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    llm = raw["llm"]
    bert = raw["bert"]
    security = raw.get("security", {})
    contract_chat = raw.get("contract_chat", {})
    preference = load_llm_preference()
    return LegalAISettings(
        top_k=int(bert.get("top_k", 5)),
        domain_threshold=float(bert.get("domain_threshold", 0.5)),
        doctrine_review_threshold=float(bert.get("doctrine_review_threshold", 0.2)),
        auto_route_threshold=float(bert.get("auto_route_threshold", 0.6)),
        minimum_margin=float(bert.get("minimum_margin", 0.15)),
        classifier_max_length=int(bert.get("max_length", 256)),
        provider=os.getenv("CONTRACTLAB_LLM_PROVIDER", str(llm["provider"])),
        base_url=os.getenv("CONTRACTLAB_LLM_BASE_URL", str(llm["base_url"])).rstrip("/"),
        model=os.getenv("CONTRACTLAB_LLM_MODEL", str(preference.get("model") or llm["model"])),
        fallback_model=str(llm["fallback_model"]),
        temperature=float(llm["temperature"]),
        drafting_temperature=float(llm.get("drafting_temperature", llm["temperature"])),
        max_output_tokens=int(llm.get("max_output_tokens", 8192)),
        context_window=int(llm.get("context_window", 16384)),
        max_context_characters=int(llm["max_context_characters"]),
        rag_enabled=bool(llm.get("rag_enabled", False)),
        prompt_paths={name: _project_path(value) for name, value in raw["prompts"].items()},
        schema_path=_project_path(str(llm["schema"])),
        config_path=target,
        maximum_request_characters=int(security.get("maximum_request_characters", 250000)),
        maximum_passage_characters=int(security.get("maximum_passage_characters", 2400)),
        passage_overlap_characters=int(security.get("passage_overlap_characters", 240)),
        maximum_passages=int(security.get("maximum_passages", 200)),
        maximum_history_messages=int(security.get("maximum_history_messages", 10)),
        maximum_history_characters=int(security.get("maximum_history_characters", 16000)),
        maximum_concurrent_requests=int(security.get("maximum_concurrent_requests", 4)),
        contract_chat_storage_path=_project_path(
            str(contract_chat.get("storage_path", "artifacts/contract_chats"))
        ),
        maximum_files_per_message=int(contract_chat.get("maximum_files_per_message", 10)),
        maximum_file_bytes=int(contract_chat.get("maximum_file_bytes", 26214400)),
        maximum_archive_entries=int(contract_chat.get("maximum_archive_entries", 100)),
        maximum_archive_uncompressed_bytes=int(
            contract_chat.get("maximum_archive_uncompressed_bytes", 52428800)
        ),
        maximum_documents_per_chat=int(contract_chat.get("maximum_documents_per_chat", 50)),
        maximum_spreadsheet_cells=int(contract_chat.get("maximum_spreadsheet_cells", 100000)),
        maximum_ocr_pages=int(contract_chat.get("maximum_ocr_pages", 50)),
        maximum_extracted_characters_per_document=int(
            contract_chat.get("maximum_extracted_characters_per_document", 250000)
        ),
    )


def load_llm_preference() -> dict[str, Any]:
    if not PREFERENCE_PATH.exists():
        return {}
    try:
        return json.loads(PREFERENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_llm_preference(model: str) -> None:
    PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCE_PATH.write_text(json.dumps({"model": model}, indent=2), encoding="utf-8")
    load_legal_ai_settings.cache_clear()
