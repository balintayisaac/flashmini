"""Development tools: allowlisted git, project tests, and linting.

``git_ops`` only exposes read-mostly verbs; destructive operations
(reset/push/force/amend) are refused with a ``denied`` result. ``run_tests``
and ``lint`` shell out through the shared runner with generous timeouts.
"""

from __future__ import annotations

import re
import shlex
import shutil
import sys
import time

from flashmini.exec import elapsed_ms, run_command
from flashmini.tools.context import ToolContext
from flashmini.tools.types import ToolResult

_GIT_OPS = {"status", "diff", "log", "show", "add", "commit", "rev-parse", "branch"}

_BRANCH_SAFE_FLAGS = {"-a", "--all", "-v", "--verbose", "-l", "--list", "--show-current"}

_GIT_ENV = {
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_EDITOR": "true",
    "LC_ALL": "C.UTF-8",
}

_FORBIDDEN_ARG_PARTS = ("..", "/", "--output", "--git-dir", "--work-tree", "--amend", "--force")

_RESULT_RE = re.compile(r"(\d+)\s+passed|(\d+)\s+failed")


def git_ops(ctx: ToolContext, op: str, args: list[str] | None = None) -> ToolResult:
    """Run one allowlisted git command inside the workspace."""
    started = time.monotonic()
    tool = "git_ops"
    args = list(args or [])
    if op not in _GIT_OPS:
        return ToolResult.denied(
            tool,
            f"git op not allowed: {op!r} (allowed: {', '.join(sorted(_GIT_OPS))})",
            duration_ms=elapsed_ms(started),
            next_action_hint="use read-only inspection or 'commit' for changes",
        )
    if shutil.which("git") is None:
        return ToolResult.error(
            tool,
            "git is not installed",
            duration_ms=elapsed_ms(started),
            next_action_hint="install git and retry",
        )
    for arg in args:
        if any(part in arg for part in _FORBIDDEN_ARG_PARTS):
            return ToolResult.denied(
                tool,
                f"argument not allowed: {arg!r}",
                duration_ms=elapsed_ms(started),
                next_action_hint="paths must be relative to the workspace",
            )
    if op == "branch":
        for arg in args:
            if arg.startswith("-") and arg not in _BRANCH_SAFE_FLAGS:
                return ToolResult.denied(
                    tool,
                    f"branch flag not allowed: {arg!r}",
                    duration_ms=elapsed_ms(started),
                    next_action_hint="branch is list-only; creating/deleting is not supported",
                )
    result = run_command(
        ["git", op, *args],
        cwd=ctx.workspace,
        timeout_s=ctx.default_timeout_s,
        max_output_bytes=ctx.max_output_bytes,
        env=_GIT_ENV,
    )
    output = result.stdout if result.stdout else "(no output)"
    return _finish(tool, started, result.returncode, result.timed_out, output, result.stderr)


def run_tests(ctx: ToolContext, command: str | None = None) -> ToolResult:
    """Run the project's test suite (pytest by default)."""
    return _run_project_command(
        ctx,
        tool="run_tests",
        command=command,
        default=["uv", "run", "pytest", "-q"],
        fallback=[sys.executable, "-m", "pytest", "-q"],
        timeout_s=ctx.test_timeout_s,
        parse_summary=True,
    )


def lint(ctx: ToolContext, command: str | None = None) -> ToolResult:
    """Run the project's linter (ruff by default)."""
    return _run_project_command(
        ctx,
        tool="lint",
        command=command,
        default=["uv", "run", "ruff", "check", "."],
        fallback=[sys.executable, "-m", "ruff", "check", "."],
        timeout_s=ctx.default_timeout_s,
        parse_summary=False,
    )


def _run_project_command(
    ctx: ToolContext,
    *,
    tool: str,
    command: str | None,
    default: list[str],
    fallback: list[str],
    timeout_s: float,
    parse_summary: bool,
) -> ToolResult:
    started = time.monotonic()
    if command:
        argv = shlex.split(command)
        if not argv:
            return ToolResult.error(
                tool,
                "empty command",
                duration_ms=elapsed_ms(started),
                next_action_hint="pass a command string or omit it for the default",
            )
    else:
        argv = _default_argv(ctx, default, fallback)
    result = run_command(
        argv,
        cwd=ctx.workspace,
        timeout_s=timeout_s,
        max_output_bytes=ctx.max_output_bytes,
    )
    note = ""
    if command is None and argv != fallback and _spawn_failed(result.stderr):
        # The project-managed runner could not start (uv env without the tool
        # installed); degrade to the fallback interpreter and say so.
        result = run_command(
            fallback,
            cwd=ctx.workspace,
            timeout_s=timeout_s,
            max_output_bytes=ctx.max_output_bytes,
        )
        note = f"[flashmini] {argv[0]} could not spawn the command; retried with python -m.\n"
    output = note + (result.stdout if result.stdout else "(no output)")
    hint = None
    if parse_summary:
        passed = sum(int(g) for g in re.findall(r"(\d+)\s+passed", result.stdout))
        failed = sum(int(g) for g in re.findall(r"(\d+)\s+failed", result.stdout))
        if passed or failed:
            hint = f"{passed} passed, {failed} failed"
            if failed:
                hint += "; inspect the first failure and fix it"
    return _finish(tool, started, result.returncode, result.timed_out, output, result.stderr, hint)


def _default_argv(ctx: ToolContext, default: list[str], fallback: list[str]) -> list[str]:
    """Prefer the project-managed runner only where a project exists to manage."""
    if shutil.which(default[0]) and (ctx.workspace / "pyproject.toml").exists():
        return default
    return fallback


_SPAWN_FAILURE_MARKERS = ("Failed to spawn", "No pyproject.toml")


def _spawn_failed(stderr: str) -> bool:
    return any(marker in stderr for marker in _SPAWN_FAILURE_MARKERS)


def _finish(
    tool: str,
    started: float,
    returncode: int | None,
    timed_out: bool,
    output: str,
    stderr: str,
    hint: str | None = None,
) -> ToolResult:
    duration = elapsed_ms(started)
    if timed_out:
        return ToolResult.timeout(
            tool,
            output,
            duration_ms=duration,
            next_action_hint=hint or "command timed out; narrow its scope",
        )
    if returncode == 0:
        return ToolResult.ok(
            tool, output, stderr=stderr, duration_ms=duration, next_action_hint=hint
        )
    return ToolResult.error(
        tool,
        output,
        stderr=stderr,
        duration_ms=duration,
        next_action_hint=hint or f"command exited with code {returncode}",
    )
