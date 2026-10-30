"""File tools: bounded reads, atomic writes, and automatic backups.

``read_file`` never returns more than ``ctx.max_read_bytes`` and refuses
binaries. ``write_file`` replaces files atomically (temp file + rename) and
drops a copy of the previous content into ``.flashmini/backups/`` first.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from flashmini.exec import elapsed_ms
from flashmini.tools.context import ToolContext
from flashmini.tools.types import ToolResult

BACKUP_ROOT = ".flashmini/backups"
_BINARY_SAMPLE = 8192


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
    """Replace ``target`` with ``content`` atomically."""
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".flashmini-tmp-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


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
            next_action_hint="list the directory first to find the right path",
        )
    raw = target.read_bytes()[: ctx.max_read_bytes + 1]
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
    numbered = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(selected, start=start))
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
    backup_path = backup_file(ctx, target) if backup else None
    atomic_write(target, content)
    output = f"wrote {len(content)} chars to {path}"
    if backup_path is not None:
        output += f" (backup: {backup_path.relative_to(ctx.workspace)})"
    return ToolResult.ok(tool, output, duration_ms=elapsed_ms(started))
