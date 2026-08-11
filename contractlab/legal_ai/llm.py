from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised when a configured generative provider cannot return a usable response."""


class GenerativeLLM(Protocol):
    model: str
    provider: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str: ...


@dataclass(slots=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_seconds: float = 180.0
    temperature: float = 0.2
    max_output_tokens: int = 8192
    context_window: int = 16384
    opener: Callable[..., Any] = urlopen
    provider: str = "ollama"

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

    def list_models(self) -> list[str]:
        return [str(row["name"]) for row in self._request("/api/tags").get("models", [])]

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except LLMError:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.context_window,
            },
        }
        if schema is not None:
            payload["format"] = schema
        result = self._request("/api/chat", payload)
        content = str(result.get("message", {}).get("content", "")).strip()
        if not content:
            raise LLMError("Ollama returned an empty assistant message.")
        return content
