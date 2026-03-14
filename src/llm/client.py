"""Thin optional OpenAI-compatible client used for answer synthesis."""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, ValidationError

from src.app.settings import settings


class LLMError(Exception):
    """Raised when a remote LLM call fails."""


class LLMResponse(BaseModel):
    """Normalized response metadata from the remote LLM."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None


class LLMClient:
    """Small HTTPX-based client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.temperature = settings.openai_temperature if temperature is None else temperature
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def is_configured(self) -> bool:
        """Return whether the client has enough configuration to make calls."""

        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        """Return HTTP headers for the configured backend."""

        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_completion(
        self,
        messages: list[dict[str, object]],
        *,
        response_format: dict[str, str] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Call the remote chat completion endpoint and normalize the response."""

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        body = response.json()
        choice = body["choices"][0]
        usage = body.get("usage", {})
        message_content = choice["message"]["content"]
        if isinstance(message_content, list):
            message_content = "\n".join(
                item.get("text", "") for item in message_content if isinstance(item, dict)
            )

        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.total_completion_tokens += int(usage.get("completion_tokens", 0))

        return LLMResponse(
            content=str(message_content or ""),
            model=str(body.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            finish_reason=choice.get("finish_reason"),
        )

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a plain-text completion."""

        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._chat_completion(messages, temperature=temperature).content

    def generate_structured(
        self,
        prompt: str,
        *,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        """Request JSON output and validate it against a Pydantic model."""

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        rendered_system_prompt = (
            system_prompt or "You are a helpful assistant."
        ) + f"\n\nReturn JSON matching this schema exactly:\n{schema_json}"

        response = self._chat_completion(
            [
                {"role": "system", "content": rendered_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        try:
            payload = json.loads(response.content)
            return response_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(f"Structured LLM response could not be validated: {exc}") from exc

    def usage_stats(self) -> dict[str, int]:
        """Return cumulative token usage for the client."""

        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for one or more texts using the configured backend."""

        if not texts:
            return []
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": model or settings.openai_embedding_model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Embedding request failed: {exc}") from exc

        body = response.json()
        rows = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        embeddings = [list(map(float, row.get("embedding", []))) for row in rows]
        if len(embeddings) != len(texts):
            raise LLMError("Embedding response count did not match the request count.")
        return embeddings
