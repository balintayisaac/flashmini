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
    (ctx.workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = run_tests(ctx, f"{sys.executable} -m pytest -q test_ok.py")
    assert result.status is ToolStatus.OK
    assert "1 passed" in (result.next_action_hint or "")


def test_run_tests_explicit_pytest_gets_failure_visibility(ctx: ToolContext) -> None:
    """Model-supplied commands must not bypass failure visibility (v12 probe:
    command='pytest' produced a verbose header + trimmed-out assertion diffs,
    and the model oscillated on guesses). Force -q/--tb=line/-ra whenever the
    command runs pytest; non-pytest commands stay untouched."""
    (ctx.workspace / "test_x.py").write_text("def test_fail():\n    assert 1 == 2\n")
    command = f"{sys.executable} -m pytest"
    result = run_tests(ctx, command)
    out = result.output
    assert "test session starts" not in out  # -q applied
    assert "short test summary info" in out  # -ra applied
    assert "FAILED" in out  # one-line failure visible


def test_run_tests_reports_failures(ctx: ToolContext) -> None:
    (ctx.workspace / "test_x.py").write_text("def test_fail():\n    assert 1 == 2\n")
    result = run_tests(ctx, f"{sys.executable} -m pytest -q test_x.py")
    hint = result.next_action_hint or ""
    assert "1 failed" in hint


def test_lint_custom_command(ctx: ToolContext) -> None:
    result = lint(ctx, f"{sys.executable} -m ruff check .")
    assert result.status is ToolStatus.OK


class TestCommandOverrideAllowlist:
    """command= is restricted to allowlisted test/lint runners (security C1).

    Before the allowlist, `run_tests(command="bash -c 'env'")` executed
    arbitrary argv as the user with the full parent environment."""

    def test_shell_runner_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, "bash -c 'echo LEAK: $FLASHMINI_API_KEY'")
        assert result.status is ToolStatus.DENIED
        assert "runner not allowed" in result.output

    def test_python_dash_c_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, f"{sys.executable} -c \"print('3 passed')\"")
        assert result.status is ToolStatus.DENIED

    def test_network_runner_denied(self, ctx: ToolContext) -> None:
        result = lint(ctx, "curl --version")
        assert result.status is ToolStatus.DENIED

    def test_absolute_selector_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, f"{sys.executable} -m pytest /etc/passwd")
        assert result.status is ToolStatus.DENIED

    def test_traversal_selector_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, f"{sys.executable} -m pytest ../outside")
        assert result.status is ToolStatus.DENIED

    def test_disallowed_flag_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, f"{sys.executable} -m pytest -p no:cacheprovider")
        assert result.status is ToolStatus.DENIED

    def test_bare_uv_denied(self, ctx: ToolContext) -> None:
        result = run_tests(ctx, "uv sync")
        assert result.status is ToolStatus.DENIED

    def test_ruff_requires_subcommand(self, ctx: ToolContext) -> None:
        result = lint(ctx, "ruff")
        assert result.status is ToolStatus.DENIED

    def test_allowlisted_forms_pass_validation(self, ctx: ToolContext) -> None:
        (ctx.workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        for command in (
            "pytest -q test_ok.py",
            "uv run pytest -q test_ok.py",
        ):
            result = run_tests(ctx, command)
            assert result.status is ToolStatus.OK, command
        result = lint(ctx, "ruff check .")
        assert result.status is ToolStatus.OK


def test_run_tests_default_degrades_without_project(ctx: ToolContext) -> None:
    """A bare workspace (no pyproject) must still get a working pytest run."""
    tests_dir = ctx.workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    result = run_tests(ctx)
    assert result.status is ToolStatus.OK
    assert "1 passed" in (result.next_action_hint or "")
