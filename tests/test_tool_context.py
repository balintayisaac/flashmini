"""Tests for the workspace jail."""

from pathlib import Path

import pytest

from flashmini.tools import ToolContext, WorkspaceViolation


def test_resolves_relative_paths(ctx: ToolContext) -> None:
    assert ctx.resolve("pkg/mod.py") == ctx.workspace / "pkg" / "mod.py"
    assert ctx.resolve(".") == ctx.workspace


def test_rejects_parent_escape(ctx: ToolContext) -> None:
    with pytest.raises(WorkspaceViolation):
        ctx.resolve("../outside.txt")


def test_rejects_absolute_outside(ctx: ToolContext) -> None:
    with pytest.raises(WorkspaceViolation):
        ctx.resolve("/etc/passwd", must_exist=True)


def test_rejects_symlink_escape(ctx: ToolContext) -> None:
    link = ctx.workspace / "escape"
    link.symlink_to("/etc")
    with pytest.raises(WorkspaceViolation):
        ctx.resolve("escape/passwd", must_exist=True)


def test_read_only_context_blocks_writes(ctx: ToolContext) -> None:
    ro = ToolContext(workspace=ctx.workspace, allow_write=False)
    with pytest.raises(WorkspaceViolation):
        ro.resolve("file.txt", for_write=True)
    assert ro.resolve("file.txt") == ctx.workspace / "file.txt"


def test_git_writes_blocked(ctx: ToolContext) -> None:
    (ctx.workspace / ".git").mkdir()
    with pytest.raises(WorkspaceViolation):
        ctx.resolve(".git/config", for_write=True)
    assert ctx.resolve(".git/config") == ctx.workspace / ".git" / "config"


def test_lenient_resolution_of_missing_paths(ctx: ToolContext) -> None:
    resolved = ctx.resolve("a/b/c.txt")
    assert resolved == ctx.workspace / "a" / "b" / "c.txt"


def test_workspace_is_resolved_in_post_init(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path / "sub" / "..")
    assert ctx.workspace == tmp_path.resolve()
