"""File tools: bounded reads, atomic writes, and automatic backups.

``read_file`` never returns more than ``ctx.max_read_bytes`` and refuses
binaries. ``write_file`` replaces files atomically (temp file + rename) and
drops a copy of the previous content into ``.flashmini/backups/`` first.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from flashmini.exec import elapsed_ms
from flashmini.tools.context import ToolContext
from flashmini.tools.types import ToolResult

BACKUP_ROOT = ".flashmini/backups"
_BINARY_SAMPLE = 8192
_DISPLAY_LINE_RE = re.compile(r"^\s*\d+[\t:]")


def backup_file(ctx: ToolContext, target: Path) -> Path | None:
    """Copy an existing file into the backup store; return the copy's path."""
    if not target.is_file():
        return None
    rel = target.relative_to(ctx.workspace)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = ctx.workspace / BACKUP_ROOT / stamp / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, dest)
    return dest


def atomic_write(target: Path, content: str) -> None:
    """Replace ``target`` with ``content`` atomically, preserving its mode."""
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".flashmini-tmp-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        if target.exists():
            # mkstemp creates 0600; a rewritten file must keep its prior mode
            # (incl. exec bits) instead of silently dropping to owner-only.
            os.chmod(tmp, target.stat().st_mode & 0o777)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _looks_like_display_copy(content: str) -> bool:
    """True when content visibly starts with read_file's line-number annotations."""
    lines = [line for line in content.splitlines() if line.strip()]
    return sum(1 for line in lines[:3] if _DISPLAY_LINE_RE.match(line)) >= 2


def read_file(
    ctx: ToolContext,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolResult:
    """Read a text file inside the workspace, line-numbered and bounded."""
    started = time.monotonic()
    tool = "read_file"
    target = ctx.resolve(path)
    if target.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return ToolResult.ok(
            tool,
            "\n".join(entries),
            duration_ms=elapsed_ms(started),
            next_action_hint="directory listing; pass a file path to read contents",
        )
    if not target.is_file():
        return ToolResult.error(
            tool,
            f"file not found: {path}",
            duration_ms=elapsed_ms(started),
            next_action_hint=(
                "if the task asks you to create this file, write it with "
                "write_file; otherwise pass its parent directory to read_file "
                "to list the directory"
            ),
        )
    # Read through a bounded file handle: read_bytes() would slurp the whole
    # file into memory before the slice (OOM DoS on a multi-GiB file).
    with target.open("rb") as fh:
        raw = fh.read(ctx.max_read_bytes + 1)
    if b"\x00" in raw[:_BINARY_SAMPLE]:
        return ToolResult.error(
            tool,
            f"binary file (not text): {path}",
            duration_ms=elapsed_ms(started),
            next_action_hint="only text files can be read",
        )
    truncated = len(raw) > ctx.max_read_bytes
    text = raw[: ctx.max_read_bytes].decode("utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    start = max(start_line, 1)
    end = end_line if end_line is not None else total
    selected = lines[start - 1 : end]
    numbered = "\n".join(f"{n:>6}: {line}" for n, line in enumerate(selected, start=start))
    if not selected:
        hint = f"no lines in the requested range; file has {total} line(s)"
    elif truncated:
        hint = f"file truncated at {ctx.max_read_bytes} bytes; showing lines {start}+"
    elif start != 1 or end < total:
        hint = f"showing lines {start}-{start + len(selected) - 1} of {total}"
    else:
        hint = None
    return ToolResult.ok(
        tool,
        numbered,
        duration_ms=elapsed_ms(started),
        truncated=truncated,
        next_action_hint=hint,
    )


def write_file(
    ctx: ToolContext,
    path: str,
    content: str,
    *,
    create_parents: bool = False,
    backup: bool = True,
) -> ToolResult:
    """Write a text file atomically, keeping a backup of the old content."""
    started = time.monotonic()
    tool = "write_file"
    target = ctx.resolve(path, for_write=True)
    if target.is_dir():
        return ToolResult.error(
            tool,
            f"path is a directory: {path}",
            duration_ms=elapsed_ms(started),
            next_action_hint="choose a file path, not a directory",
        )
    if not target.parent.is_dir():
        if not create_parents:
            return ToolResult.error(
                tool,
                f"parent directory does not exist: {target.parent}",
                duration_ms=elapsed_ms(started),
                next_action_hint="pass create_parents=True to create missing directories",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
    if _looks_like_display_copy(content):
        return ToolResult.error(
            tool,
            "content looks like read_file display output (line numbers at line start); "
            "write only the raw file text",
            duration_ms=elapsed_ms(started),
            next_action_hint="remove the line-number prefixes from every line and retry",
        )
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            existing = None
        if existing == content:
            return ToolResult.ok(
                tool,
                f"no changes: {path} content is identical to what was written",
                duration_ms=elapsed_ms(started),
                next_action_hint=(
                    "your write changed nothing; if you intended a fix, the old "
                    "content is still in place — make the actual edit"
                ),
            )
    backup_path = backup_file(ctx, target) if backup else None
    atomic_write(target, content)
    output = f"wrote {len(content)} chars to {path}"
    if backup_path is not None:
        output += f" (backup: {backup_path.relative_to(ctx.workspace)})"
    return ToolResult.ok(tool, output, duration_ms=elapsed_ms(started))
