"""Deterministic offline fakes for benchmark tests (no network).

A bag-of-words hash embedder gives same-topic memories real cosine similarity,
and a keyword-routed fake client makes salience/supersession/recheck behave
deterministically (keyed on the generator's marker tokens).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from eval.generator import STALE_MARKER, SUPERSEDE_MARKER

_DIM = 96


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words embedding over a hashed vocabulary (deterministic)."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _DIM
        for token in re.findall(r"[a-zA-Z_]+", text.lower()):
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        if not any(vec):
            vec[0] = 1.0  # avoid an all-zero vector
        vectors.append(vec)
    return vectors


def _supersession_verdicts(text: str) -> list[dict[str, Any]]:
    """Supersede candidates marked stale, but only for a superseding new memory."""
    match = re.search(r"body:\s*(.*?)\n\nCANDIDATES", text, re.DOTALL)
    new_body = match.group(1) if match else ""
    if SUPERSEDE_MARKER not in new_body:
        return []
    after = text.split("CANDIDATES", 1)[1] if "CANDIDATES" in text else ""
    verdicts: list[dict[str, Any]] = []
    for chunk in after.split("- id: ")[1:]:
        candidate_id = chunk.split()[0]
        if STALE_MARKER in chunk:
            verdicts.append(
                {
                    "target_id": candidate_id,
                    "relation": "supersedes",
                    "confidence": 0.95,
                    "rationale": "new version supersedes the stale one",
                }
            )
    return verdicts


class FakeClient:
    """Keyword-routed offline LLM stand-in."""

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Return a scripted reply based on prompt keywords."""
        text = " ".join(m["content"] for m in messages)
        if "reusability" in text:
            return json.dumps({"score": 0.6, "rationale": "scripted"})
        if "CURRENT SOURCE" in text:
            return json.dumps({"relation": "needs_update", "confidence": 0.9, "rationale": "x"})
        if "relation" in text and "supersede" in text:
            return json.dumps(_supersession_verdicts(text))
        return "[]"

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Deterministic bag-of-words embeddings."""
        return fake_embed(texts)
