"""Tests for the ToolResult envelope."""

from flashmini.tools import ToolResult, ToolStatus


def test_status_values() -> None:
    assert ToolStatus.OK == "ok"
    assert ToolStatus.ERROR == "error"
    assert ToolStatus.TIMEOUT == "timeout"
    assert ToolStatus.DENIED == "denied"


def test_ok_constructor() -> None:
    result = ToolResult.ok("demo", "out", next_action_hint="next")
    assert result.tool == "demo"
    assert result.status is ToolStatus.OK
    assert result.output == "out"
    assert result.next_action_hint == "next"
    assert result.duration_ms == 0
    assert result.truncated is False


def test_error_constructor_keeps_stderr() -> None:
    result = ToolResult.error("demo", "out", stderr="boom")
    assert result.status is ToolStatus.ERROR
    assert result.stderr == "boom"


def test_timeout_and_denied_defaults() -> None:
    timeout = ToolResult.timeout("demo")
    denied = ToolResult.denied("demo", "nope")
    assert timeout.status is ToolStatus.TIMEOUT
    assert denied.status is ToolStatus.DENIED
    assert denied.output == "nope"


def test_serialization_roundtrip() -> None:
    result = ToolResult.ok("demo", "out", truncated=True)
    payload = result.model_dump()
    assert set(payload) == {
        "tool",
        "status",
        "output",
        "stderr",
        "duration_ms",
        "truncated",
        "next_action_hint",
    }
    assert ToolResult.model_validate(payload) == result
