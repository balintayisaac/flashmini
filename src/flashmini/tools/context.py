"""Execution context and workspace jail for tools.

Every tool receives a ``ToolContext`` and must route every path through
``resolve()``. Anything that escapes the workspace root raises
``WorkspaceViolation``; the registry converts that into a ``denied``
envelope. Writes inside ``.git/`` are blocked as well.
"""

from dataclasses import dataclass
from pathlib import Path


class WorkspaceViolation(Exception):
    """Raised when a tool tries to touch a path outside its workspace."""


@dataclass
class ToolContext:
    """Per-instance tool configuration; no global mutable state."""

    workspace: Path
    max_read_bytes: int = 100_000
    max_output_bytes: int = 200_000
    default_timeout_s: float = 120.0
    test_timeout_s: float = 300.0
    allow_write: bool = True

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()

    def resolve(self, raw: str, *, must_exist: bool = False, for_write: bool = False) -> Path:
        """Resolve ``raw`` against the workspace and enforce the jail."""
        if not raw or raw == ".":
            candidate = self.workspace
        else:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
        resolved = candidate.resolve() if must_exist else _resolve_lenient(candidate)
        if resolved != self.workspace and not resolved.is_relative_to(self.workspace):
            raise WorkspaceViolation(f"path escapes the workspace: {raw!r}")
        if for_write and not self.allow_write:
            raise WorkspaceViolation("this tool context is read-only (writes disabled)")
        if for_write and _inside_git(self.workspace, resolved):
            raise WorkspaceViolation("writes inside .git/ are blocked")
        return resolved


def _resolve_lenient(candidate: Path) -> Path:
    """Resolve as much of the path as exists, keeping a non-existent suffix."""
    if candidate.exists():
        return candidate.resolve()
    parent = candidate.parent
    if parent == candidate:
        return candidate
    return _resolve_lenient(parent) / candidate.name


def _inside_git(workspace: Path, resolved: Path) -> bool:
    try:
        rel = resolved.relative_to(workspace)
    except ValueError:
        return True
    return ".git" in rel.parts
