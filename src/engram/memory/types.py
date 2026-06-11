"""Enumeration of the typed-memory categories Engram tracks."""

from __future__ import annotations

from enum import StrEnum


class MemoryType(StrEnum):
    """The kinds of project knowledge Engram stores."""

    ARCHITECTURE = "architecture"
    CONVENTION = "convention"
    COMPONENT = "component"
    BUG_FIX = "bug_fix"
    REJECTED_APPROACH = "rejected_approach"
    OPEN_THREAD = "open_thread"
