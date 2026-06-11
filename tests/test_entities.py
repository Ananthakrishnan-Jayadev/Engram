"""Tests for Python code-entity parsing."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from engram.code.entities import parse_file, scan_project

SAMPLE = """\
def top_level():
    return 1


class Service:
    def method(self):
        return 2
"""


def _write(tmp: str, name: str, text: str) -> Path:
    """Write `text` to `tmp/name` and return the path."""
    path = Path(tmp) / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_file_extracts_expected_entities() -> None:
    """A module yields module + class + function/method entities with stable keys."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = _write(tmp, "svc.py", SAMPLE)
        by_key = {e.entity_key: e for e in parse_file(path, project_id="p1", root=tmp)}

        assert by_key["svc.py"].kind == "module"
        assert by_key["svc.py::top_level"].kind == "function"
        assert by_key["svc.py::Service"].kind == "class"
        assert by_key["svc.py::Service.method"].kind == "function"
        assert by_key["svc.py::Service.method"].qualname == "Service.method"


def test_source_hash_is_stable_and_localized() -> None:
    """Re-parsing is stable; editing one function changes only its hash."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = _write(tmp, "svc.py", SAMPLE)
        first = {e.entity_key: e.source_hash for e in parse_file(path, root=tmp)}
        again = {e.entity_key: e.source_hash for e in parse_file(path, root=tmp)}
        assert first == again

        _write(tmp, "svc.py", SAMPLE.replace("return 1", "return 99"))
        edited = {e.entity_key: e.source_hash for e in parse_file(path, root=tmp)}
        assert edited["svc.py::top_level"] != first["svc.py::top_level"]
        assert edited["svc.py::Service.method"] == first["svc.py::Service.method"]


def test_scan_project_skips_vendor_dirs() -> None:
    """scan_project walks source files but skips .venv and friends."""
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        _write(tmp, "a.py", "def f():\n    return 1\n")
        venv = Path(tmp) / ".venv"
        venv.mkdir()
        (venv / "junk.py").write_text("def g():\n    return 2\n", encoding="utf-8")

        paths = {e.path for e in scan_project(tmp, project_id="p1")}
        assert "a.py" in paths
        assert not any(".venv" in p for p in paths)
