"""Codex-style apply_patch engine.

Format (compatible with the format proven in OpenAI's Codex CLI):

    *** Begin Patch
    *** Add File: path/to/new.py
    +new line
    *** Update File: path/to/existing.py
    @@ context anchor
     unchanged line
    -removed line
    +added line
    *** Delete File: path/to/gone.py
    *** End Patch

Matching is line-exact first; a whitespace-insensitive fallback is allowed
but flagged in the summary. The whole patch is parsed and planned in memory
before a single byte is written: if any hunk does not match, nothing
changes on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from flashmini.exec import elapsed_ms
from flashmini.tools.context import ToolContext
from flashmini.tools.files import atomic_write, backup_file
from flashmini.tools.types import ToolResult


class PatchError(Exception):
    """Raised when a patch cannot be parsed or does not match the files."""


@dataclass
class Hunk:
    """One ``@@``-anchored block of ``+``/``-``/context lines."""

    context: str | None
    lines: list[tuple[str, str]]  # (prefix, text); prefix in {" ", "+", "-"}


@dataclass
class FileOp:
    """One file section of a patch."""

    kind: str  # "add" | "update" | "delete"
    path: str
    added_lines: list[str] = field(default_factory=list)  # for "add"
    hunks: list[Hunk] = field(default_factory=list)  # for "update"


@dataclass
class _PlannedWrite:
    target: Path
    content: str
    action: str
    delete: bool = False


_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File: "
_UPDATE = "*** Update File: "
_DELETE = "*** Delete File: "
_MOVE = "*** Move to: "


def parse_patch(text: str) -> list[FileOp]:
    """Parse the Codex patch format into a list of file operations."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != _BEGIN:
        raise PatchError("patch must start with '*** Begin Patch'")
    i += 1

    ops: list[FileOp] = []
    current: FileOp | None = None
    current_hunk: Hunk | None = None
    saw_end = False

    def ensure_hunk() -> Hunk:
        nonlocal current_hunk
        if current is None:
            raise PatchError("content outside any file section")
        if current_hunk is None:
            current_hunk = Hunk(context=None, lines=[])
            current.hunks.append(current_hunk)
        return current_hunk

    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip() == _END:
            saw_end = True
            break
        if line.startswith(_MOVE):
            raise PatchError("'*** Move to:' is not supported by this tool")
        if line.startswith(_ADD):
            current = FileOp(kind="add", path=line[len(_ADD) :].strip())
            ops.append(current)
            current_hunk = None
            continue
        if line.startswith(_UPDATE):
            current = FileOp(kind="update", path=line[len(_UPDATE) :].strip())
            ops.append(current)
            current_hunk = None
            continue
        if line.startswith(_DELETE):
            current = FileOp(kind="delete", path=line[len(_DELETE) :].strip())
            ops.append(current)
            current_hunk = None
            continue
        if line.startswith("*** "):
            raise PatchError(f"unexpected patch marker: {line!r}")
        if current is None:
            raise PatchError(f"content outside any file section: {line!r}")
        if current.kind == "delete":
            raise PatchError(f"delete section for {current.path!r} must contain no lines")
        if current.kind == "add":
            if line.startswith("+"):
                current.added_lines.append(line[1:])
            elif line == "":
                current.added_lines.append("")
            else:
                raise PatchError(f"add-file lines must start with '+': {line!r}")
            continue
        # update section
        if line.startswith("@@"):
            current_hunk = Hunk(context=line[2:].strip() or None, lines=[])
            current.hunks.append(current_hunk)
            continue
        if line and line[0] in " +-":
            ensure_hunk().lines.append((line[0], line[1:]))
            continue
        if line == "":
            ensure_hunk().lines.append((" ", ""))
            continue
        raise PatchError(f"cannot parse patch line: {line!r}")

    if not saw_end:
        raise PatchError("patch is missing '*** End Patch'")
    if not ops:
        raise PatchError("patch contains no file sections")
    for op in ops:
        if not op.path:
            raise PatchError(f"empty path in a {op.kind} section")
        if op.kind == "update" and not op.hunks:
            raise PatchError(f"update section for {op.path!r} has no hunks")
    return ops


def _split_lines(text: str) -> tuple[list[str], str, bool]:
    """Split into lines, detect newline style, and track trailing newline."""
    newline = "\r\n" if "\r\n" in text else "\n"
    parts = text.split(newline)
    trailing = False
    if parts and parts[-1] == "":
        parts.pop()
        trailing = True
    return parts, newline, trailing


def _find_block(lines: list[str], block: list[str], start_at: int) -> tuple[int, bool] | None:
    """Locate ``block`` at/after ``start_at``; exact first, then rstrip-fuzz."""
    n = len(block)
    for idx in range(start_at, len(lines) - n + 1):
        if lines[idx : idx + n] == block:
            return idx, False
    stripped = [b.rstrip() for b in block]
    for idx in range(start_at, len(lines) - n + 1):
        if [line.rstrip() for line in lines[idx : idx + n]] == stripped:
            return idx, True
    return None


def _apply_update(op: FileOp, original: str) -> tuple[str, bool, int, int]:
    """Apply all hunks of one update operation in memory."""
    lines, newline, trailing = _split_lines(original)
    cursor = 0
    fuzz = False
    added = removed = 0
    for hunk in op.hunks:
        old_block = [text for prefix, text in hunk.lines if prefix in (" ", "-")]
        new_block = [text for prefix, text in hunk.lines if prefix in (" ", "+")]
        if not old_block:
            raise PatchError(f"hunk in {op.path!r} has no context or removal lines to anchor on")
        found = _find_block(lines, old_block, cursor)
        if found is None:
            anchor = f" (anchor: {hunk.context!r})" if hunk.context else ""
            raise PatchError(
                f"hunk does not match {op.path!r}{anchor}: no exact or whitespace-fuzzy match found"
            )
        idx, used_fuzz = found
        fuzz = fuzz or used_fuzz
        lines[idx : idx + len(old_block)] = new_block
        cursor = idx + len(new_block)
        added += sum(1 for prefix, _ in hunk.lines if prefix == "+")
        removed += sum(1 for prefix, _ in hunk.lines if prefix == "-")
    result = newline.join(lines)
    if trailing:
        result += newline
    return result, fuzz, added, removed


def apply_patch(ctx: ToolContext, patch: str) -> ToolResult:
    """Validate the whole patch, then apply it atomically (all or nothing)."""
    started = time.monotonic()
    tool = "apply_patch"
    try:
        ops = parse_patch(patch)
        planned: list[_PlannedWrite] = []
        for op in ops:
            target = ctx.resolve(op.path, for_write=True)
            if op.kind == "add":
                if target.exists():
                    raise PatchError(f"cannot add {op.path!r}: it already exists")
                content = "\n".join(op.added_lines)
                if op.added_lines:
                    content += "\n"
                planned.append(
                    _PlannedWrite(target, content, f"A {op.path} (+{len(op.added_lines)} lines)")
                )
            elif op.kind == "delete":
                if not target.is_file():
                    raise PatchError(f"cannot delete {op.path!r}: file not found")
                planned.append(_PlannedWrite(target, "", f"D {op.path}", delete=True))
            else:
                if not target.is_file():
                    raise PatchError(f"cannot update {op.path!r}: file not found")
                original = target.read_text(encoding="utf-8", errors="replace")
                updated, fuzz, added, removed = _apply_update(op, original)
                tag = " (fuzzy match)" if fuzz else ""
                planned.append(
                    _PlannedWrite(target, updated, f"M {op.path} (+{added}/-{removed}{tag})")
                )
        # Every hunk matched and every path was validated -- only now write.
        for item in planned:
            backup_file(ctx, item.target)
            if item.delete:
                item.target.unlink()
            else:
                atomic_write(item.target, item.content)
    except PatchError as exc:
        return ToolResult.error(
            tool,
            str(exc),
            duration_ms=elapsed_ms(started),
            next_action_hint="fix the hunks and retry; no files were changed",
        )
    return ToolResult.ok(
        tool,
        "\n".join(item.action for item in planned),
        duration_ms=elapsed_ms(started),
        next_action_hint="review the changes with git diff before committing",
    )
