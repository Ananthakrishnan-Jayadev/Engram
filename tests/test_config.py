"""Tests for settings loading."""

from __future__ import annotations

import pytest

from engram.config import Settings, get_settings


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings pick up env vars and apply documented defaults."""
    get_settings.cache_clear()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-123")
    # Avoid reading a developer's real .env during the test.
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.dashscope_api_key == "sk-test-123"
    assert settings.base_url.endswith("/compatible-mode/v1")
    assert settings.model_flash == "qwen3.5-flash"
    assert settings.model_heavy == "qwen3-max"
    assert settings.model_embed == "text-embedding-v3"
    assert settings.chroma_path == "./chroma_db"
    assert settings.sqlite_path == "./engram.sqlite"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings returns the same cached instance."""
    get_settings.cache_clear()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-123")
    assert get_settings() is get_settings()
    get_settings.cache_clear()
