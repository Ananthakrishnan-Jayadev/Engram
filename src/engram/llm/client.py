"""Qwen LLM client over the DashScope OpenAI-compatible endpoint."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from engram.config import Settings, get_settings

# Embeddings sometimes fail on DashScope's OpenAI compatible-mode depending on
# the chosen model. If that happens, switch `model_embed` to a compatible model
# or call the native DashScope embeddings endpoint. Native fallback sketch:
#
#     import dashscope
#     from dashscope import TextEmbedding
#     dashscope.api_key = settings.dashscope_api_key
#     resp = TextEmbedding.call(model=settings.model_embed, input=texts)
#     return [r["embedding"] for r in resp.output["embeddings"]]

_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0
_TIMEOUT_SECONDS = 30.0


class QwenClient:
    """Thin wrapper around the OpenAI SDK pointed at DashScope (Qwen)."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the client; reads settings if none are supplied."""
        self._settings = settings or get_settings()
        self._client = OpenAI(
            api_key=self._settings.dashscope_api_key,
            base_url=self._settings.base_url,
            timeout=_TIMEOUT_SECONDS,
        )

    def _with_retry(self, fn: Any) -> Any:
        """Call `fn` with up to 3 attempts and exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - retry then re-raise below
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BASE_BACKOFF_SECONDS * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def chat(self, messages: list[dict[str, str]], model: str | None = None, **kw: Any) -> str:
        """Send a chat completion and return the assistant's text content.

        Defaults to the flash model for high-frequency operations.
        """
        chosen = model or self._settings.model_flash

        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=chosen,
                messages=messages,  # type: ignore[arg-type]
                **kw,
            )
            return resp.choices[0].message.content or ""

        return self._with_retry(_call)

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed `texts` and return one vector per input.

        Raises a clear error if compatible-mode embeddings fail, pointing the
        user at `model_embed` or the native DashScope embeddings endpoint.
        """
        chosen = model or self._settings.model_embed

        def _call() -> list[list[float]]:
            resp = self._client.embeddings.create(model=chosen, input=texts)
            return [item.embedding for item in resp.data]

        try:
            return self._with_retry(_call)
        except Exception as exc:  # noqa: BLE001 - re-raise with guidance
            raise RuntimeError(
                f"Embeddings call failed for model '{chosen}' on DashScope "
                "compatible-mode. Switch `model_embed` to a compatible model, or "
                "use the native DashScope embeddings endpoint (see the commented "
                "fallback in engram/llm/client.py)."
            ) from exc
