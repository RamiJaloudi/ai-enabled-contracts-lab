from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from contractlab.document_parser import ExtractedDocument


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_filename(value: str) -> str:
    name = Path(value.replace(chr(92), "/")).name
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return clean[:180] or "document"


def _conversation_id(value: str) -> str:
    return str(UUID(value))


class StoredArtifact(BaseModel):
    filename: str
    relative_path: str
    media_type: str
    size_bytes: int


class StoredMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str
    content: str
    created_at: datetime = Field(default_factory=_now)
    source_label: str | None = None
    artifacts: list[StoredArtifact] = Field(default_factory=list)


class StoredDocument(BaseModel):
    id: str
    filename: str
    source_filename: str
    upload_relative_path: str
    media_type: str
    size_bytes: int
    sha256: str
    text: str
    warnings: list[str] = Field(default_factory=list)
    ocr_used: bool = False
    page_count: int | None = None
    created_at: datetime = Field(default_factory=_now)


class ContractConversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New contract chat"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    messages: list[StoredMessage] = Field(default_factory=list)
    documents: list[StoredDocument] = Field(default_factory=list)


class ContractChatStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, conversation_id: str) -> Path:
        return self.root / _conversation_id(conversation_id)

    def _save(self, conversation: ContractConversation) -> None:
        conversation.updated_at = _now()
        directory = self._directory(conversation.id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "conversation.json"
        temporary = directory / "conversation.json.tmp"
        temporary.write_text(conversation.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(target)

    def create(self, title: str = "New contract chat") -> ContractConversation:
        conversation = ContractConversation(title=title.strip()[:120] or "New contract chat")
        self._save(conversation)
        return conversation

    def load(self, conversation_id: str) -> ContractConversation:
        target = self._directory(conversation_id) / "conversation.json"
        return ContractConversation.model_validate_json(target.read_text(encoding="utf-8"))

    def try_load(self, conversation_id: object) -> ContractConversation | None:
        """Return a valid saved conversation without trusting stale UI state."""
        if not isinstance(conversation_id, str):
            return None
        try:
            return self.load(conversation_id)
        except (OSError, ValueError):
            return None

    def list(self) -> list[ContractConversation]:
        conversations: list[ContractConversation] = []
        for target in self.root.glob("*/conversation.json"):
            try:
                conversations.append(
                    ContractConversation.model_validate_json(target.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(conversations, key=lambda item: item.updated_at, reverse=True)

    def add_message(
        self,
        conversation: ContractConversation,
        *,
        role: str,
        content: str,
        source_label: str | None = None,
        artifacts: list[StoredArtifact] | None = None,
    ) -> StoredMessage:
        message = StoredMessage(
            role=role,
            content=content,
            source_label=source_label,
            artifacts=artifacts or [],
        )
        conversation.messages.append(message)
        if conversation.title == "New contract chat" and role == "user":
            conversation.title = re.sub(r"\s+", " ", content).strip()[:80] or conversation.title
        self._save(conversation)
        return message

    def clear_messages(self, conversation: ContractConversation) -> None:
        conversation.messages.clear()
        conversation.title = "New contract chat"
        self._save(conversation)

    def add_upload(
        self,
        conversation: ContractConversation,
        *,
        upload_filename: str,
        upload_data: bytes,
        extracted: list[ExtractedDocument],
    ) -> list[StoredDocument]:
        directory = self._directory(conversation.id)
        uploads = directory / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(upload_filename)
        digest = hashlib.sha256(upload_data).hexdigest()
        stored_name = f"{digest[:12]}-{safe}"
        target = uploads / stored_name
        if not target.exists():
            target.write_bytes(upload_data)
        relative = target.relative_to(directory).as_posix()
        added: list[StoredDocument] = []
        existing_hashes = {document.sha256 for document in conversation.documents}
        for item in extracted:
            if item.sha256 in existing_hashes:
                continue
            document = StoredDocument(
                id=f"D{len(conversation.documents) + 1}",
                filename=item.filename,
                source_filename=safe,
                upload_relative_path=relative,
                media_type=item.media_type,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                text=item.text,
                warnings=list(item.warnings),
                ocr_used=item.ocr_used,
                page_count=item.page_count,
            )
            conversation.documents.append(document)
            added.append(document)
            existing_hashes.add(item.sha256)
        self._save(conversation)
        return added

    def save_artifact(
        self,
        conversation: ContractConversation,
        *,
        filename: str,
        media_type: str,
        data: bytes,
    ) -> StoredArtifact:
        directory = self._directory(conversation.id)
        outputs = directory / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        safe = _safe_filename(filename)
        target = outputs / f"{uuid4().hex[:10]}-{safe}"
        target.write_bytes(data)
        return StoredArtifact(
            filename=safe,
            relative_path=target.relative_to(directory).as_posix(),
            media_type=media_type,
            size_bytes=len(data),
        )

    def artifact_bytes(self, conversation_id: str, artifact: StoredArtifact) -> bytes:
        directory = self._directory(conversation_id).resolve()
        target = (directory / artifact.relative_path).resolve()
        if directory not in target.parents:
            raise ValueError("Artifact path escapes the conversation directory.")
        return target.read_bytes()
