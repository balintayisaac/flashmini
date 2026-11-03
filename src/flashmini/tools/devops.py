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
from pathlib import Path

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

_FORBIDDEN_ARG_PARTS = (
    "..",
    "/",
    "--output",
    "--git-dir",
    "--work-tree",
    "--amend",
    "--force",
    "--ext-diff",
    "--textconv",
    "--no-index",
)

_RESULT_RE = re.compile(r"(\d+)\s+passed|(\d+)\s+failed")

# --- command= override allowlist (security audit C1) -----------------------
# A model-supplied `command` used to be arbitrary argv execution (verified:
# `bash -c 'echo LEAK: $FLASHMINI_API_KEY'` ran as the user).  Overrides are
# now restricted to allowlisted test/lint runners with a fixed flag set.
# NOTE: pytest still executes workspace test code by design — this closes the
# shell/env/network one-liner hole, not a VM sandbox.
_PYTEST_FLAGS = frozenset(
    {
        "-q",
        "--quiet",
        "-v",
        "-vv",
        "--tb=line",
        "--tb=short",
        "--tb=long",
        "--tb=no",
        "-ra",
        "-rA",
        "-x",
        "-s",
        "--no-header",
        "--lf",
        "--ff",
        "--disable-warnings",
        "--collect-only",
        "--setup-show",
    }
)
_PYTEST_VALUE_FLAGS = frozenset({"-k", "-m", "--maxfail", "--durations", "-W"})
_RUFF_SUBCOMMANDS = frozenset({"check", "format"})
_RUFF_FLAGS = frozenset({"-q", "-v", "--statistics", "--no-cache"})
_RUFF_VALUE_FLAGS = frozenset({"--select", "--extend-select", "--ignore", "--output-format"})


def _selector_ok(tok: str) -> bool:
    """A positional arg must be a workspace-relative test/lint selector."""
    return (
        tok != "" and not tok.startswith("-") and not tok.startswith(("/", "~")) and ".." not in tok
    )


def _validate_override(tool: str, argv: list[str], started: float) -> ToolResult | None:
    """Return a ``denied`` result unless argv is an allowlisted override."""
    exe = Path(argv[0]).name  # overrides may use absolute interpreter paths
    if exe in ("python", "python3"):
        if len(argv) >= 3 and argv[1] == "-m" and argv[2] in ("pytest", "ruff"):
            runner, rest = argv[2], argv[3:]
        else:
            runner, rest = "", argv[1:]
    elif exe == "uv":
        if len(argv) >= 3 and argv[1] == "run" and argv[2] in ("pytest", "ruff"):
            runner, rest = argv[2], argv[3:]
        else:
            runner, rest = "", argv[1:]
    elif exe in ("pytest", "ruff"):
        runner, rest = exe, argv[1:]
    else:
        runner, rest = "", argv[1:]
    if runner == "":
        return ToolResult.denied(
            tool,
            f"command runner not allowed: {argv[0]!r}",
            duration_ms=elapsed_ms(started),
            next_action_hint=(
                "use pytest, python -m pytest, uv run pytest, ruff, or "
                "uv run ruff; omit command= for the default suite"
            ),
        )
    if runner == "ruff":
        if not rest or rest[0] not in _RUFF_SUBCOMMANDS:
            return ToolResult.denied(
                tool,
                "ruff override must start with 'check' or 'format'",
                duration_ms=elapsed_ms(started),
                next_action_hint="e.g. command='ruff check src'",
            )
        rest = rest[1:]
        ok_flags, value_flags = _RUFF_FLAGS, _RUFF_VALUE_FLAGS
    else:
        ok_flags, value_flags = _PYTEST_FLAGS, _PYTEST_VALUE_FLAGS
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok.startswith("-"):
            if tok in value_flags:
                if i + 1 >= len(rest):
                    return ToolResult.denied(
                        tool,
                        f"flag {tok!r} requires a value",
                        duration_ms=elapsed_ms(started),
                    )
                i += 2
                continue
            if tok not in ok_flags:
                return ToolResult.denied(
                    tool,
                    f"flag not allowed: {tok!r}",
                    duration_ms=elapsed_ms(started),
                    next_action_hint=(
                        "see the pytest/ruff docs; only a fixed set of "
                        "reporting and selection flags is permitted"
                    ),
                )
        elif not _selector_ok(tok):
            return ToolResult.denied(
                tool,
                f"selector not allowed: {tok!r}",
                duration_ms=elapsed_ms(started),
                next_action_hint="use workspace-relative paths without '..'",
            )
        i += 1
    return None


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
        default=["uv", "run", "pytest", "-q", "--tb=line"],
        fallback=[sys.executable, "-m", "pytest", "-q", "--tb=line"],
        timeout_s=ctx.test_timeout_s,
        parse_summary=True,
        ensure_flags=("-q", "--tb=line", "-ra"),
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
    ensure_flags: tuple[str, ...] = (),
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
        denial = _validate_override(tool, argv, started)
        if denial is not None:
            return denial
    else:
        argv = _default_argv(ctx, default, fallback)
    # Model-supplied commands are runner choices, not verbosity choices: keep
    # the failure-visibility flags on every pytest invocation path (v12 probe:
    # command="pytest" bypassed --tb=line, assertion diffs were trimmed away
    # by the envelope head+tail cut, and the model oscillated blindly).
    if any(Path(a).name == "pytest" for a in argv):
        argv = argv + [f for f in ensure_flags if f not in argv]
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
