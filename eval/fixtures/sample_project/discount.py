"""Discount calculation for the sample project.

NOTE: contains an intentional bug — see the project's bug_history.json and the
inline marker below. `apply_percentage_discount` treats `percent` as the
fraction *to keep* rather than the fraction to remove.
"""

from __future__ import annotations


def apply_percentage_discount(price: float, percent: float) -> float:
    """Apply a `percent` discount to `price` and return the new price.

    BUG (intentional): this multiplies by `percent / 100` (the amount removed)
    instead of `1 - percent / 100` (the amount kept), so a 20% discount returns
    20% of the price instead of 80% of it.
    """
    return price * (percent / 100.0)
