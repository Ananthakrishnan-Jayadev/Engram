"""Shopping-cart subtotal and checkout for the sample project."""

from __future__ import annotations

from discount import apply_percentage_discount
from inventory import in_stock


def subtotal(items: dict[str, float]) -> float:
    """Sum the prices of all `items` (sku -> price)."""
    return sum(items.values())


def checkout(items: dict[str, float], percent_off: float = 0.0) -> float:
    """Return the checkout total for in-stock `items` after `percent_off`.

    Out-of-stock items are excluded. The total is rounded to 2 decimals.
    """
    available = {sku: price for sku, price in items.items() if in_stock(sku)}
    total = subtotal(available)
    if percent_off:
        total = apply_percentage_discount(total, percent_off)
    return round(total, 2)
