"""Shared fixtures for the tool-layer tests."""

from pathlib import Path

import pytest

from flashmini.tools import ToolContext


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


@pytest.fixture()
def sample(ctx: ToolContext) -> ToolContext:
    """A small workspace with one package and one markdown file."""
    (ctx.workspace / "pkg").mkdir()
    (ctx.workspace / "pkg" / "mod.py").write_text(
        "def alpha():\n    return 1\n\n\nclass Beta:\n    pass\n"
    )
    (ctx.workspace / "README.md").write_text("# demo\nline two\n")
    return ctx
