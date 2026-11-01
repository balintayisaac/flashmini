"""Regression test: the ripgrep argv must use --ignore-case (not --case-insensitive)."""

from __future__ import annotations

from pathlib import Path

from flashmini.tools import ToolContext, ToolStatus, grep
from flashmini.tools import search as search_mod


def _fake_rg(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake rg that dumps its argv to a file and reports 'no matches'."""
    argv_file = tmp_path / "rg.argv"
    script = tmp_path / "rg"
    script.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_file}"\nexit 1\n')
    script.chmod(0o755)
    return script, argv_file


def test_case_insensitive_uses_ignore_case(monkeypatch, tmp_path: Path) -> None:
    script, argv_file = _fake_rg(tmp_path)
    monkeypatch.setattr(search_mod, "find_ripgrep", lambda *args, **kwargs: script)
    ctx = ToolContext(workspace=tmp_path)

    result = grep(ctx, "needle", case_sensitive=False)
    assert result.status == ToolStatus.OK
    args = argv_file.read_text().splitlines()
    assert "--ignore-case" in args
    assert "--case-insensitive" not in args

    result = grep(ctx, "needle", case_sensitive=True)
    assert result.status == ToolStatus.OK
    args = argv_file.read_text().splitlines()
    assert "--ignore-case" not in args


def test_no_match_hint_leads_with_shorter_pattern(monkeypatch, tmp_path: Path) -> None:
    """The no-match hint must lead with the bare-identifier/shorter-pattern advice.

    Probe #4 showed models burning steps on over-specific grep patterns (full
    import statements) while case-insensitivity is only occasionally the issue,
    so the shorter-pattern advice comes first.
    """
    script, _argv_file = _fake_rg(tmp_path)
    monkeypatch.setattr(search_mod, "find_ripgrep", lambda *args, **kwargs: script)
    ctx = ToolContext(workspace=tmp_path)

    result = grep(ctx, "needle", case_sensitive=True)
    hint = result.next_action_hint or ""
    assert "shorter pattern" in hint
    assert "bare" in hint
    assert hint.index("shorter pattern") < hint.index("case_sensitive")
