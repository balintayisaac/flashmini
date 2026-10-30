"""Tests for the apply_patch engine."""

import pytest

from flashmini.tools import ToolContext, ToolStatus, apply_patch, write_file
from flashmini.tools.patch import PatchError, parse_patch


def make_patch(*sections: str) -> str:
    return "*** Begin Patch\n" + "\n".join(sections) + "\n*** End Patch\n"


def test_add_file(ctx: ToolContext) -> None:
    result = apply_patch(ctx, make_patch("*** Add File: new.txt", "+alpha", "+beta"))
    assert result.status is ToolStatus.OK
    assert (ctx.workspace / "new.txt").read_text() == "alpha\nbeta\n"
    assert "A new.txt" in result.output


def test_update_file(ctx: ToolContext) -> None:
    write_file(ctx, "target.txt", "one\ntwo\nthree\n")
    patch = make_patch(
        "*** Update File: target.txt",
        "@@",
        " one",
        "-two",
        "+TWO",
        " three",
    )
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.OK
    assert (ctx.workspace / "target.txt").read_text() == "one\nTWO\nthree\n"


def test_multi_hunk_update(ctx: ToolContext) -> None:
    write_file(ctx, "target.txt", "a\nb\nc\nd\n")
    patch = make_patch(
        "*** Update File: target.txt",
        "@@",
        "-b",
        "+B",
        "@@",
        "-d",
        "+D",
    )
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.OK
    assert (ctx.workspace / "target.txt").read_text() == "a\nB\nc\nD\n"


def test_delete_file_with_backup(ctx: ToolContext) -> None:
    write_file(ctx, "gone.txt", "bye\n")
    result = apply_patch(ctx, make_patch("*** Delete File: gone.txt"))
    assert result.status is ToolStatus.OK
    assert not (ctx.workspace / "gone.txt").exists()
    backups = list((ctx.workspace / ".flashmini" / "backups").rglob("gone.txt"))
    assert backups and backups[-1].read_text() == "bye\n"


def test_mismatch_leaves_file_untouched(ctx: ToolContext) -> None:
    write_file(ctx, "a.txt", "keep\n")
    patch = make_patch("*** Update File: a.txt", "-missing", "+changed")
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.ERROR
    assert (ctx.workspace / "a.txt").read_text() == "keep\n"


def test_multi_file_atomicity(ctx: ToolContext) -> None:
    write_file(ctx, "a.txt", "alpha\n")
    patch = make_patch(
        "*** Add File: b.txt",
        "+new",
        "*** Update File: missing.txt",
        "-x",
        "+y",
    )
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.ERROR
    assert not (ctx.workspace / "b.txt").exists()
    assert (ctx.workspace / "a.txt").read_text() == "alpha\n"


def test_fuzzy_match_is_flagged(ctx: ToolContext) -> None:
    (ctx.workspace / "fuzzy.txt").write_text("alpha  \nbeta\n")
    patch = make_patch(
        "*** Update File: fuzzy.txt",
        "@@",
        "-alpha",
        "+ALPHA",
    )
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.OK
    assert "(fuzzy match)" in result.output
    assert (ctx.workspace / "fuzzy.txt").read_text() == "ALPHA\nbeta\n"


def test_add_existing_file_rejected(ctx: ToolContext) -> None:
    write_file(ctx, "already.txt", "x")
    result = apply_patch(ctx, make_patch("*** Add File: already.txt", "+y"))
    assert result.status is ToolStatus.ERROR
    assert "already exists" in result.output


def test_update_missing_file_rejected(ctx: ToolContext) -> None:
    result = apply_patch(ctx, make_patch("*** Update File: ghost.txt", "-a", "+b"))
    assert result.status is ToolStatus.ERROR
    assert "not found" in result.output


def test_pure_addition_without_context_rejected(ctx: ToolContext) -> None:
    write_file(ctx, "anchor.txt", "content\n")
    patch = make_patch("*** Update File: anchor.txt", "@@ note", "+added")
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.ERROR
    assert "anchor" in result.output


def test_move_not_supported(ctx: ToolContext) -> None:
    write_file(ctx, "src.txt", "x\n")
    patch = make_patch(
        "*** Update File: src.txt",
        "*** Move to: dst.txt",
        "-x",
        "+y",
    )
    result = apply_patch(ctx, patch)
    assert result.status is ToolStatus.ERROR
    assert "not supported" in result.output


def test_parser_requires_markers() -> None:
    with pytest.raises(PatchError):
        parse_patch("no markers here")
    with pytest.raises(PatchError):
        parse_patch("*** Begin Patch\n*** Add File: x.txt\n+y\n")
