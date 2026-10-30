"""Tests for grep / symbol_search (Python fallback paths)."""

import pytest

import flashmini.tools.search as search_mod
from flashmini.tools import ToolContext, ToolStatus, grep, symbol_search


@pytest.fixture()
def no_rg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_mod, "find_ripgrep", lambda ctx: None)


def test_grep_finds_matches(sample: ToolContext, no_rg: None) -> None:
    result = grep(sample, "alpha")
    assert result.status is ToolStatus.OK
    assert "pkg/mod.py" in result.output
    assert "def alpha" in result.output


def test_grep_no_matches(sample: ToolContext, no_rg: None) -> None:
    result = grep(sample, "zzz_not_there")
    assert result.status is ToolStatus.OK
    assert "no matches" in result.output


def test_grep_max_results(sample: ToolContext, no_rg: None) -> None:
    (sample.workspace / "many.txt").write_text("hit\n" * 10)
    result = grep(sample, "hit", max_results=3)
    assert result.truncated is True
    assert len(result.output.splitlines()) == 3


def test_grep_glob_filter(sample: ToolContext, no_rg: None) -> None:
    result = grep(sample, "line", glob="*.md")
    assert "README.md" in result.output
    assert "mod.py" not in result.output


def test_grep_case_insensitive(sample: ToolContext, no_rg: None) -> None:
    result = grep(sample, "ALPHA", case_sensitive=False)
    assert "def alpha" in result.output


def test_grep_invalid_regex(sample: ToolContext, no_rg: None) -> None:
    result = grep(sample, "([unclosed")
    assert result.status is ToolStatus.ERROR
    assert "invalid regex" in result.output


def test_symbol_search_finds_function_and_class(sample: ToolContext) -> None:
    functions = symbol_search(sample, "^alpha$")
    assert "function alpha" in functions.output
    classes = symbol_search(sample, "^Beta$")
    assert "class Beta" in classes.output


def test_symbol_search_no_match(sample: ToolContext) -> None:
    result = symbol_search(sample, "nope_nothing")
    assert result.status is ToolStatus.OK
    assert "no symbols found" in result.output
