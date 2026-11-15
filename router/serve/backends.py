"""LLM backends: offline mock, OpenAI-compatible, and Anthropic Messages API.

The "openai" backend is OpenAI-compatible: any provider exposing ``/chat/completions``
(OpenAI, Gemini's OpenAI-compatible endpoint, Groq, Together, vLLM, Ollama) works by
pointing CHEAP_BASE_URL / STRONG_BASE_URL at it.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import httpx

from router.serve.engine import _content_text
from router.serve.settings import Settings

TIMEOUT_S = 60.0
OPENAI_PARAMS = (
    "temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty",
    "frequency_penalty", "response_format", "tools", "tool_choice",
)
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024


class BackendError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"backend error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / 4)) if text else 0


def _messages_text(messages: list[dict]) -> str:
    return "\n".join(_content_text(m.get("content")) for m in messages)


class MockBackend:
    """Deterministic offline backend; token counts estimated as ceil(chars / 4)."""

    def __init__(self, latency_s: float = 0.0):
        self.latency_s = float(latency_s)

    async def complete(self, model: str, messages: list[dict], **params: Any) -> ChatResult:
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = _content_text(m.get("content"))
                break
        text = f"[mock:{model}] {last_user[:80]}"
        return ChatResult(
            text=text,
            prompt_tokens=estimate_tokens(_messages_text(messages)),
            completion_tokens=estimate_tokens(text),
            model=model,
        )


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            return str(err.get("message", err))
        return str(err)
    except Exception:
        return resp.text[:500] or resp.reason_phrase


async def _post(url: str, headers: dict, payload: dict, transport: httpx.AsyncBaseTransport | None) -> dict:
    try:
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT_S) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise BackendError(502, f"upstream request failed: {exc!r}") from exc
    if resp.status_code < 200 or resp.status_code >= 300:
        raise BackendError(resp.status_code, _error_message(resp))
    try:
        return resp.json()
    except ValueError as exc:
        raise BackendError(502, "upstream returned a non-JSON body") from exc


class OpenAICompatBackend:
    def __init__(self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport

    async def complete(self, model: str, messages: list[dict], **params: Any) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": messages}
        for key in OPENAI_PARAMS:
            if params.get(key) is not None:
                payload[key] = params[key]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = await _post(f"{self.base_url}/chat/completions", headers, payload, self.transport)
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BackendError(502, "malformed upstream response: missing choices[0].message") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", estimate_tokens(_messages_text(messages)))),
            completion_tokens=int(usage.get("completion_tokens", estimate_tokens(text))),
            model=model,
        )


class AnthropicBackend:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.transport = transport

    async def complete(self, model: str, messages: list[dict], **params: Any) -> ChatResult:
        system_parts = [
            _content_text(m.get("content")) for m in messages if m.get("role") == "system"
        ]
        chat = [
            {"role": m["role"], "content": _content_text(m.get("content"))}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": chat,
            "max_tokens": int(params.get("max_tokens") or ANTHROPIC_DEFAULT_MAX_TOKENS),
        }
        if system_parts:
            payload["system"] = "\n\n".join(p for p in system_parts if p)
        for key in ("temperature", "top_p"):
            if params.get(key) is not None:
                payload[key] = params[key]
        stop = params.get("stop")
        if stop:
            payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = await _post(f"{self.base_url}/v1/messages", headers, payload, self.transport)
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise BackendError(502, "malformed upstream response: missing content blocks")
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            prompt_tokens=int(usage.get("input_tokens", estimate_tokens(_messages_text(messages)))),
            completion_tokens=int(usage.get("output_tokens", estimate_tokens(text))),
            model=model,
        )


def build_backends(settings: Settings) -> tuple[Any, Any]:
    """Return (cheap_backend, strong_backend) for the configured provider."""
    if settings.backend == "mock":
        return MockBackend(), MockBackend()
    if settings.backend == "openai":
        return (
            OpenAICompatBackend(settings.cheap_base_url, settings.cheap_api_key or ""),
            OpenAICompatBackend(settings.strong_base_url, settings.strong_api_key or ""),
        )
    if settings.backend == "anthropic":
        return (
            AnthropicBackend(settings.cheap_api_key or "", settings.cheap_base_url),
            AnthropicBackend(settings.strong_api_key or "", settings.strong_base_url),
        )
    raise ValueError(f"unknown backend {settings.backend!r}")
