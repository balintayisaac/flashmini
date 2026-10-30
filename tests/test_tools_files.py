"""Tests for read_file / write_file."""

from flashmini.tools import ToolContext, ToolStatus, read_file, write_file


def test_write_read_roundtrip(ctx: ToolContext) -> None:
    written = write_file(ctx, "note.txt", "hello\nworld\n")
    assert written.status is ToolStatus.OK
    read = read_file(ctx, "note.txt")
    assert read.status is ToolStatus.OK
    assert "1\thello" in read.output
    assert "2\tworld" in read.output


def test_overwrite_creates_backup(ctx: ToolContext) -> None:
    write_file(ctx, "note.txt", "old")
    write_file(ctx, "note.txt", "new")
    backups = list((ctx.workspace / ".flashmini" / "backups").rglob("note.txt"))
    assert backups
    assert backups[-1].read_text() == "old"
    assert (ctx.workspace / "note.txt").read_text() == "new"


def test_read_missing_file(ctx: ToolContext) -> None:
    result = read_file(ctx, "missing.txt")
    assert result.status is ToolStatus.ERROR
    assert "not found" in result.output


def test_read_binary_refused(ctx: ToolContext) -> None:
    (ctx.workspace / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    result = read_file(ctx, "blob.bin")
    assert result.status is ToolStatus.ERROR
    assert "binary" in result.output


def test_read_truncates_at_cap(ctx: ToolContext) -> None:
    small = ToolContext(workspace=ctx.workspace, max_read_bytes=100)
    write_file(ctx, "big.txt", "x" * 500)
    result = read_file(small, "big.txt")
    assert result.truncated is True
    assert len(result.output) < 500


def test_read_directory_listing(ctx: ToolContext) -> None:
    (ctx.workspace / "sub").mkdir()
    (ctx.workspace / "a.txt").write_text("a")
    result = read_file(ctx, ".")
    assert result.status is ToolStatus.OK
    assert "a.txt" in result.output
    assert "sub/" in result.output


def test_read_line_range(ctx: ToolContext) -> None:
    write_file(ctx, "lines.txt", "one\ntwo\nthree\nfour\n")
    result = read_file(ctx, "lines.txt", start_line=2, end_line=3)
    assert "2\ttwo" in result.output
    assert "3\tthree" in result.output
    assert "one" not in result.output


def test_write_requires_parents_flag(ctx: ToolContext) -> None:
    blocked = write_file(ctx, "deep/nested/file.txt", "x")
    assert blocked.status is ToolStatus.ERROR
    created = write_file(ctx, "deep/nested/file.txt", "x", create_parents=True)
    assert created.status is ToolStatus.OK
    assert (ctx.workspace / "deep" / "nested" / "file.txt").read_text() == "x"
