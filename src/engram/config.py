"""Application configuration loaded from environment / `.env`."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for Engram, sourced from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DashScope / Qwen credentials and endpoint.
    dashscope_api_key: str
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # Model selection (see engram.llm.models for the single source of truth).
    model_flash: str = "qwen3.5-flash"
    model_heavy: str = "qwen3-max"
    model_embed: str = "text-embedding-v3"

    # Local store locations.
    chroma_path: str = "./chroma_db"
    sqlite_path: str = "./engram.sqlite"


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    return Settings()  # type: ignore[call-arg]
