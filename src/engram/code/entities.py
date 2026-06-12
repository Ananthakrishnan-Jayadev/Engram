"""Parse source files into code entities for the knowledge graph.

The parser is pluggable by language; Phase 3 implements Python only, via the
stdlib `ast` module. Each entity has a stable `entity_key` and a `source_hash`
so later syncs can detect when its source changed.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

# Directories never walked when scanning a project.
SKIP_DIRS = {".venv", "venv", ".git", "node_modules", "__pycache__", "chroma_db"}

# Maximum entity matches returned for a single hint.
ENTITY_MATCH_CAP = 3


class CodeEntity(BaseModel):
    """A parsed code entity (module, class, or function/method)."""

    entity_key: str
    project_id: str
    path: str
    qualname: str
    kind: str  # module | class | function
    source_hash: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _hash(text: str) -> str:
    """Return the sha256 hex digest of `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relpath(path: Path, root: Path | None) -> str:
    """Return `path` relative to `root` as a posix string (or its basename)."""
    if root is None:
        return path.name
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.name


def _iter_defs(node: ast.AST, prefix: str) -> Iterator[tuple[ast.AST, str, str]]:
    """Yield (node, qualname, kind) for nested classes/functions under `node`."""
    for child in getattr(node, "body", []):
        if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            qualname = f"{prefix}.{child.name}" if prefix else child.name
            kind = "class" if isinstance(child, ast.ClassDef) else "function"
            yield child, qualname, kind
            yield from _iter_defs(child, qualname)


def _arg_names(args: ast.arguments) -> str:
    """Render a function's parameter names (no defaults/annotations)."""
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return ", ".join(names)


def _signature(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a one-line name+signature for a class/function def (no body)."""
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases})"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({_arg_names(node.args)}){returns}"


def _module_structure(source: str, tree: ast.Module) -> str:
    """Render a module's structural surface: top-level code + child signatures.

    Function/class *bodies* are excluded, so editing a body does not change the
    module hash — only adding/removing/renaming entities or module-level code does.
    """
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(_signature(node))
        else:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                parts.append(segment)
    return "\n".join(parts)


def parse_file(
    path: str | Path, project_id: str = "default", root: str | Path | None = None
) -> list[CodeEntity]:
    """Parse a Python file into a module entity plus its classes/functions.

    `entity_key` is the posix relpath for the module, and ``relpath::qualname``
    for nested definitions. Relpaths are computed against `root` when given. The
    module hash is structural (see `_module_structure`).
    """
    file_path = Path(path)
    source = file_path.read_text(encoding="utf-8")
    rel = _relpath(file_path, Path(root) if root is not None else None)
    now = datetime.now(UTC)
    tree = ast.parse(source)

    entities = [
        CodeEntity(
            entity_key=rel,
            project_id=project_id,
            path=rel,
            qualname="",
            kind="module",
            source_hash=_hash(_module_structure(source, tree)),
            updated_at=now,
        )
    ]
    for node, qualname, kind in _iter_defs(tree, ""):
        segment = ast.get_source_segment(source, node) or ""
        entities.append(
            CodeEntity(
                entity_key=f"{rel}::{qualname}",
                project_id=project_id,
                path=rel,
                qualname=qualname,
                kind=kind,
                source_hash=_hash(segment),
                updated_at=now,
            )
        )
    return entities


def scan_project(root: str | Path, project_id: str = "default") -> list[CodeEntity]:
    """Walk `root` for *.py files (skipping SKIP_DIRS) and parse each."""
    root_path = Path(root)
    entities: list[CodeEntity] = []
    for py in sorted(root_path.rglob("*.py")):
        if any(part in SKIP_DIRS for part in py.relative_to(root_path).parts):
            continue
        try:
            entities.extend(parse_file(py, project_id=project_id, root=root_path))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return entities


def ancestor_keys(entity_key: str) -> list[str]:
    """Return the containment-ancestor keys for `entity_key`, module last.

    A function/class is contained by its enclosing class(es) and its module, so:
      "a.py::Service.method" -> ["a.py::Service", "a.py"]
      "a.py::func"           -> ["a.py"]
      "a.py"                 -> []  (a module has no parent)
    """
    if "::" not in entity_key:
        return []
    path, qualname = entity_key.split("::", 1)
    parts = qualname.split(".")
    ancestors = [f"{path}::{'.'.join(parts[:i])}" for i in range(len(parts) - 1, 0, -1)]
    ancestors.append(path)  # the module
    return ancestors


def entity_source(root: str | Path, entity: CodeEntity) -> str | None:
    """Return the current source text for `entity` under `root`, or None.

    Modules return the whole file; classes/functions return their source
    segment. None means the file (or the definition) no longer exists.
    """
    path = Path(root) / entity.path
    if not path.is_file():
        return None
    source = path.read_text(encoding="utf-8", errors="ignore")
    if entity.kind == "module" or not entity.qualname:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node, qualname, _kind in _iter_defs(tree, ""):
        if qualname == entity.qualname:
            return ast.get_source_segment(source, node)
    return None


def match_entities(hint: str, entities: list[CodeEntity], cap: int = ENTITY_MATCH_CAP) -> list[str]:
    """Resolve a free-text `hint` to entity keys (exact > symbol > path).

    Tries, in order: exact entity_key, qualname or trailing symbol, then full
    path or path basename. Returns at most `cap` keys.
    """
    for entity in entities:
        if entity.entity_key == hint:
            return [entity.entity_key]

    symbol_matches = [
        e.entity_key
        for e in entities
        if e.qualname and (e.qualname == hint or e.qualname.rsplit(".", 1)[-1] == hint)
    ]
    if symbol_matches:
        return symbol_matches[:cap]

    path_matches = [
        e.entity_key for e in entities if e.path == hint or e.path.rsplit("/", 1)[-1] == hint
    ]
    return path_matches[:cap]
