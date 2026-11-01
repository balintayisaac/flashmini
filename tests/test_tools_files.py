"""Tests for read_file / write_file."""

from flashmini.tools import ToolContext, ToolStatus, read_file, write_file


def test_write_read_roundtrip(ctx: ToolContext) -> None:
    written = write_file(ctx, "note.txt", "hello\nworld\n")
    assert written.status is ToolStatus.OK
    read = read_file(ctx, "note.txt")
    assert read.status is ToolStatus.OK
    assert "1: hello" in read.output
    assert "2: world" in read.output


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
    assert "2: two" in result.output
    assert "3: three" in result.output
    assert "one" not in result.output


def test_write_requires_parents_flag(ctx: ToolContext) -> None:
    blocked = write_file(ctx, "deep/nested/file.txt", "x")
    assert blocked.status is ToolStatus.ERROR
    created = write_file(ctx, "deep/nested/file.txt", "x", create_parents=True)
    assert created.status is ToolStatus.OK
    assert (ctx.workspace / "deep" / "nested" / "file.txt").read_text() == "x"


def test_write_rejects_display_copy(ctx: ToolContext) -> None:
    copied = "     1\talpha\n     2\tbeta\n"
    result = write_file(ctx, "copied.txt", copied)
    assert result.status is ToolStatus.ERROR
    assert "display" in result.output
    assert not (ctx.workspace / "copied.txt").exists()


def test_write_allows_single_numbered_line(ctx: ToolContext) -> None:
    result = write_file(ctx, "note.txt", "42: the answer\n")
    assert result.status is ToolStatus.OK


def test_write_file_identical_content_is_noop(tmp_path) -> None:
    """Round-6 trace: the model rewrote the typo back unchanged ("wrote 20 chars").

    A no-op write must say so instead of pretending progress was made.
    """
    ctx = ToolContext(workspace=tmp_path)
    (tmp_path / "a.txt").write_text("same\n")
    result = write_file(ctx, "a.txt", "same\n")
    assert result.status == ToolStatus.OK
    assert "identical" in result.output
    assert result.next_action_hint is not None
    assert "changed nothing" in result.next_action_hint
