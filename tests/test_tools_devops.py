"""Tests for git_ops / run_tests / lint."""

import subprocess
import sys

from flashmini.tools import ToolContext, ToolStatus, git_ops, lint, run_tests, write_file


def _init_repo(ctx: ToolContext) -> None:
    subprocess.run(["git", "init", "-q"], cwd=ctx.workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=ctx.workspace, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=ctx.workspace, check=True)


def test_git_status_outside_repo(ctx: ToolContext) -> None:
    result = git_ops(ctx, "status")
    assert result.status is ToolStatus.ERROR


def test_git_status_in_repo(ctx: ToolContext) -> None:
    _init_repo(ctx)
    result = git_ops(ctx, "status", ["--short"])
    assert result.status is ToolStatus.OK


def test_git_commit_flow(ctx: ToolContext) -> None:
    _init_repo(ctx)
    write_file(ctx, "f.txt", "x\n")
    assert git_ops(ctx, "add", ["f.txt"]).status is ToolStatus.OK
    assert git_ops(ctx, "commit", ["-m", "init"]).status is ToolStatus.OK
    log = git_ops(ctx, "log", ["--oneline"])
    assert log.status is ToolStatus.OK
    assert "init" in log.output


def test_destructive_ops_denied(ctx: ToolContext) -> None:
    for op in ("push", "reset", "checkout", "clean"):
        result = git_ops(ctx, op)
        assert result.status is ToolStatus.DENIED, op


def test_forbidden_arguments_denied(ctx: ToolContext) -> None:
    assert git_ops(ctx, "log", ["--output=/tmp/x"]).status is ToolStatus.DENIED
    assert git_ops(ctx, "add", ["../outside"]).status is ToolStatus.DENIED
    assert git_ops(ctx, "show", ["/etc/passwd"]).status is ToolStatus.DENIED
    assert git_ops(ctx, "commit", ["--amend"]).status is ToolStatus.DENIED


def test_branch_flags_restricted(ctx: ToolContext) -> None:
    _init_repo(ctx)
    assert git_ops(ctx, "branch", ["-D", "nope"]).status is ToolStatus.DENIED
    assert git_ops(ctx, "branch", ["-a"]).status is ToolStatus.OK


def test_run_tests_custom_command(ctx: ToolContext) -> None:
    command = f"{sys.executable} -c \"print('3 passed')\""
    result = run_tests(ctx, command)
    assert result.status is ToolStatus.OK
    assert "3 passed" in (result.next_action_hint or "")


def test_run_tests_reports_failures(ctx: ToolContext) -> None:
    command = f"{sys.executable} -c \"print('1 passed, 2 failed')\""
    result = run_tests(ctx, command)
    hint = result.next_action_hint or ""
    assert "1 passed" in hint
    assert "2 failed" in hint


def test_lint_custom_command(ctx: ToolContext) -> None:
    command = f"{sys.executable} -c \"print('clean')\""
    result = lint(ctx, command)
    assert result.status is ToolStatus.OK
    assert "clean" in result.output


def test_run_tests_default_degrades_without_project(ctx: ToolContext) -> None:
    """A bare workspace (no pyproject) must still get a working pytest run."""
    tests_dir = ctx.workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = run_tests(ctx)
    assert result.status is ToolStatus.OK
    assert "1 passed" in (result.next_action_hint or "")
