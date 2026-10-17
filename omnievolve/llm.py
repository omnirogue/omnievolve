# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The OmniEvolve Authors
"""A small bring-your-own-key client for OpenAI, Anthropic, Gemini and Ollama.

Standard library only (``urllib``). Keys are read from environment variables by name at call
time and are never logged, stored in the ledger or included in error messages. ``openai``
also covers any OpenAI-compatible endpoint (vLLM, LM Studio, OpenRouter and others) through
``base_url``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

PROVIDERS = ("openai", "anthropic", "gemini", "ollama")
DEFAULT_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": (),
}
DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}
BASE_URL_ENV = {"openai": "OPENAI_BASE_URL", "ollama": "OLLAMA_HOST"}
ANTHROPIC_VERSION = "2023-06-01"

# transport(url, headers, payload, timeout) -> parsed JSON response
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class LLMError(RuntimeError):
    """A provider call failed. The message never contains the API key."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    retries: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown LLM setting(s): {', '.join(unknown)}")
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
        if not self.model:
            raise ValueError("an LLM model id is required (for example the provider's model name)")

    def describe(self) -> dict[str, Any]:
        """Settings safe to persist: the key variable's name, never its value."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def _urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {exc.code} from {_host(url)}: {detail}") from None
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {_host(url)}: {exc.reason}") from None
    except (TimeoutError, OSError) as exc:
        raise LLMError(f"request to {_host(url)} failed: {exc}") from None
    except json.JSONDecodeError:
        raise LLMError(f"{_host(url)} returned a response that is not JSON") from None


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or url


class LLMClient:
    """Sends one system + user prompt and returns the reply text."""

    def __init__(self, config: LLMConfig, transport: Transport | None = None):
        config.validate()
        self.config = config
        self.transport = transport or _urllib_transport

    def complete(self, system: str, user: str) -> str:
        url, headers, payload = self.build_request(system, user)
        attempts = max(1, self.config.retries + 1)
        for attempt in range(attempts):
            try:
                data = self.transport(url, headers, payload, self.config.timeout_seconds)
                return self.parse_response(self.config.provider, data)
            except LLMError as exc:
                retryable = any(f"HTTP {code}" in str(exc) for code in (429, 500, 502, 503, 529))
                if attempt + 1 >= attempts or not retryable:
                    raise
                time.sleep(min(8.0, 1.5 * 2**attempt))
        raise LLMError("unreachable")  # pragma: no cover

    # -- request building -------------------------------------------------------------------

    def _api_key(self) -> str | None:
        names = (self.config.api_key_env,) if self.config.api_key_env else ()
        names = names or DEFAULT_KEY_ENV[self.config.provider]
        for name in names:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        if names:
            raise LLMError(f"set the {' or '.join(names)} environment variable")
        return None

    def _base_url(self) -> str:
        provider = self.config.provider
        base = self.config.base_url or os.environ.get(BASE_URL_ENV.get(provider, ""), "")
        base = (base or DEFAULT_BASE_URL[provider]).rstrip("/")
        if "://" not in base:
            base = "http://" + base
        return base

    def build_request(self, system: str, user: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        cfg = self.config
        base = self._base_url()
        headers = {"content-type": "application/json"}
        key = self._api_key()
        if cfg.provider == "openai":
            payload: dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            # OpenAI's own API takes max_completion_tokens; compatible servers take max_tokens.
            limit = "max_completion_tokens" if "api.openai.com" in base else "max_tokens"
            payload[limit] = cfg.max_tokens
            if cfg.temperature is not None:
                payload["temperature"] = cfg.temperature
            if key:
                headers["authorization"] = f"Bearer {key}"
            return f"{base}/chat/completions", headers, payload
        if cfg.provider == "anthropic":
            payload = {
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if cfg.temperature is not None:
                payload["temperature"] = cfg.temperature
            headers["anthropic-version"] = ANTHROPIC_VERSION
            if key:
                headers["x-api-key"] = key
            return f"{base}/v1/messages", headers, payload
        if cfg.provider == "gemini":
            generation: dict[str, Any] = {"maxOutputTokens": cfg.max_tokens}
            if cfg.temperature is not None:
                generation["temperature"] = cfg.temperature
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": generation,
            }
            if key:
                headers["x-goog-api-key"] = key  # a header, so the key never appears in a URL
            return f"{base}/models/{cfg.model}:generateContent", headers, payload
        # ollama
        options: dict[str, Any] = {"num_predict": cfg.max_tokens}
        if cfg.temperature is not None:
            options["temperature"] = cfg.temperature
        payload = {
            "model": cfg.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
        }
        return f"{base}/api/chat", headers, payload

    @staticmethod
    def parse_response(provider: str, data: dict[str, Any]) -> str:
        try:
            if provider == "openai":
                text = data["choices"][0]["message"]["content"]
            elif provider == "anthropic":
                text = "".join(
                    block.get("text", "")
                    for block in data["content"]
                    if block.get("type") == "text"
                )
            elif provider == "gemini":
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts)
            else:
                text = data["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected {provider} response shape") from None
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"{provider} returned an empty reply")
        return text
