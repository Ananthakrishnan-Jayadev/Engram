"""Read-only HTTP API over Engram's stores (powers the dashboard)."""

from engram.api.app import create_app

__all__ = ["create_app"]
