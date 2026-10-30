"""Tests for the registry dispatcher."""

from flashmini.tools import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolStatus,
    build_default_registry,
)


def test_default_registry_names(ctx: ToolContext) -> None:
    reg = build_default_registry(ctx)
    assert reg.names() == [
        "apply_patch",
        "git_ops",
        "grep",
        "lint",
        "read_file",
        "run_tests",
        "symbol_search",
        "write_file",
    ]


def test_unknown_tool(ctx: ToolContext) -> None:
    reg = ToolRegistry(ctx)
    result = reg.call("nope")
    assert result.status is ToolStatus.ERROR
    assert "unknown tool" in result.output


def test_invalid_arguments(ctx: ToolContext) -> None:
    reg = build_default_registry(ctx)
    result = reg.call("read_file", bogus=1)
    assert result.status is ToolStatus.ERROR
    assert "invalid arguments" in result.output


def test_crash_is_contained(ctx: ToolContext) -> None:
    reg = ToolRegistry(ctx)

    def boom() -> ToolResult:
        raise RuntimeError("kaboom")

    reg.register("boom", boom)
    result = reg.call("boom")
    assert result.status is ToolStatus.ERROR
    assert "RuntimeError" in result.output


def test_workspace_violation_becomes_denied(ctx: ToolContext) -> None:
    reg = build_default_registry(ctx)
    result = reg.call("read_file", path="../../etc/passwd")
    assert result.status is ToolStatus.DENIED


def test_conformance_missing_args_return_envelope(ctx: ToolContext) -> None:
    reg = build_default_registry(ctx)
    for name in ("read_file", "write_file", "apply_patch", "grep", "symbol_search", "git_ops"):
        result = reg.call(name)
        assert isinstance(result, ToolResult), name
        assert result.tool == name
        assert result.status is ToolStatus.ERROR
