"""Simple in-memory inventory for the sample project."""

from __future__ import annotations

_STOCK: dict[str, int] = {"apple": 10, "banana": 5, "cherry": 0}


def in_stock(sku: str) -> bool:
    """Return True if `sku` has at least one unit available."""
    return _STOCK.get(sku, 0) > 0


def units_available(sku: str) -> int:
    """Return the number of units available for `sku` (0 if unknown)."""
    return _STOCK.get(sku, 0)
