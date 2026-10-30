"""Search tools: ripgrep-backed ``grep`` and AST-based ``symbol_search``.

``grep`` prefers a real ripgrep binary (env override, PATH, or a vendored
copy under ``vendor/bin/rg``) and falls back to a pure-Python scanner that
is flagged as degraded in the result hint. ``symbol_search`` uses the
Python AST for ``.py`` files until the tree-sitter engine lands in the
code-intelligence phase.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import time
from pathlib import Path

from flashmini.exec import elapsed_ms, run_command
from flashmini.tools.context import ToolContext
from flashmini.tools.types import ToolResult

_SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".flashmini",
    ".mypy_cache",
}

_RG_DEGRADED_HINT = "python fallback scan (slower); install ripgrep for full speed"


def find_ripgrep(ctx: ToolContext) -> str | None:
    """Locate a ripgrep binary: env override, PATH, then vendored copy."""
    env_rg = os.environ.get("FLASHMINI_RG")
    if env_rg and Path(env_rg).is_file():
        return env_rg
    which = shutil.which("rg")
    if which:
        return which
    vendored = ctx.workspace / "vendor" / "bin" / "rg"
    if vendored.is_file():
        return str(vendored)
    return None


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            files.append(Path(current) / name)
    return files


def grep(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_sensitive: bool = True,
    max_results: int = 200,
) -> ToolResult:
    """Search file contents for a regular expression."""
    started = time.monotonic()
    tool = "grep"
    target = ctx.resolve(path, must_exist=True)
    rg = find_ripgrep(ctx)
    if rg is not None:
        argv = [rg, "--line-number", "--no-heading", "--color", "never"]
        if not case_sensitive:
            argv.append("--case-insensitive")
        if glob:
            argv.extend(["--glob", glob])
        argv.extend(["--regexp", pattern, str(target)])
        result = run_command(
            argv,
            cwd=ctx.workspace,
            timeout_s=ctx.default_timeout_s,
            max_output_bytes=ctx.max_output_bytes,
        )
        if result.timed_out:
            return ToolResult.timeout(
                tool,
                result.stdout,
                duration_ms=elapsed_ms(started),
                next_action_hint="search timed out; narrow the path or pattern",
            )
        if result.returncode not in (0, 1):
            return ToolResult.error(
                tool,
                result.stderr[:2000] or "ripgrep failed",
                duration_ms=elapsed_ms(started),
                next_action_hint="check the regex pattern syntax",
            )
        lines = result.stdout.splitlines()
    else:
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return ToolResult.error(
                tool,
                f"invalid regex: {exc}",
                duration_ms=elapsed_ms(started),
                next_action_hint="check the regex pattern syntax",
            )
        lines = []
        for file in _iter_files(target if target.is_dir() else target.parent):
            if not target.is_dir() and file != target:
                continue
            if glob and not file.match(glob):
                continue
            try:
                content = file.read_bytes()
            except OSError:
                continue
            if b"\x00" in content[:8192]:
                continue
            rel = file.relative_to(ctx.workspace)
            text = content.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    lines.append(f"{rel}:{number}:{line}")
    truncated = len(lines) > max_results
    selected = lines[:max_results]
    output = "\n".join(selected) if selected else "no matches"
    if rg is None:
        hint = _RG_DEGRADED_HINT
    elif not selected:
        hint = "pattern not found; try a broader regex or a different path"
    else:
        hint = None
    return ToolResult.ok(
        tool,
        output,
        duration_ms=elapsed_ms(started),
        truncated=truncated,
        next_action_hint=hint,
    )


def symbol_search(
    ctx: ToolContext,
    query: str,
    path: str = ".",
    max_results: int = 100,
) -> ToolResult:
    """Find Python definitions (functions/classes) whose name matches a regex."""
    started = time.monotonic()
    tool = "symbol_search"
    target = ctx.resolve(path, must_exist=True)
    try:
        name_re = re.compile(query, re.IGNORECASE)
    except re.error as exc:
        return ToolResult.error(
            tool,
            f"invalid name pattern: {exc}",
            duration_ms=elapsed_ms(started),
            next_action_hint="symbol queries are matched as regexes against names",
        )
    results: list[str] = []
    fallback_re = re.compile(r"^\s*(?:async\s+def|def|class)\s+(\w+)")
    for file in _iter_files(target if target.is_dir() else target.parent):
        if file.suffix != ".py":
            continue
        if not target.is_dir() and file != target:
            continue
        rel = file.relative_to(ctx.workspace)
        try:
            source = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parsed = False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            parsed = True
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
                    name_re.search(node.name)
                ):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    results.append(f"{rel}:{node.lineno}: {kind} {node.name}")
        if not parsed:
            for number, line in enumerate(source.splitlines(), 1):
                match = fallback_re.match(line)
                if match and name_re.search(match.group(1)):
                    results.append(f"{rel}:{number}: definition {match.group(1)}")
    results.sort()
    truncated = len(results) > max_results
    output = "\n".join(results[:max_results]) if results else "no symbols found"
    return ToolResult.ok(
        tool,
        output,
        duration_ms=elapsed_ms(started),
        truncated=truncated,
        next_action_hint="python-AST engine; tree-sitter symbol index arrives in Phase 3",
    )
