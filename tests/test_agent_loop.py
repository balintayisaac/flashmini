"""Tests for the Flash agent loop (scripted client, no model required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flashmini.agent import AgentLoop, build_system_prompt, parse_step, tool_call_schema
from flashmini.tools import ToolContext, ToolStatus, build_default_registry


class FakeClient:
    """Scripted replacement for ChatClient: pops replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []
        self.kwargs_seen: list[dict[str, Any]] = []

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.requests.append(list(messages))
        self.kwargs_seen.append(kwargs)
        if not self.replies:
            return '{"final": "out of scripted replies"}'
        return self.replies.pop(0)


def _registry(tmp_path: Path):
    return build_default_registry(ToolContext(workspace=tmp_path))


class TestParseStep:
    def test_plain_json(self) -> None:
        assert parse_step('{"final": "x"}') == {"final": "x"}

    def test_tolerates_surrounding_text(self) -> None:
        assert parse_step('here you go:\n{"tool": "grep", "args": {}}\n') == {
            "tool": "grep",
            "args": {},
        }

    def test_tolerates_code_fences(self) -> None:
        assert parse_step('```json\n{"final": "ok"}\n```') == {"final": "ok"}

    def test_garbage_returns_none(self) -> None:
        assert parse_step("no json here") is None
        assert parse_step("") is None
        assert parse_step("[1, 2, 3]") is None


class TestSchemas:
    def test_schema_enumerates_tool_names(self) -> None:
        names = ["grep", "read_file"]
        schema = tool_call_schema(names)
        assert schema["properties"]["tool"]["enum"] == names

    def test_system_prompt_lists_every_tool(self, tmp_path: Path) -> None:
        names = _registry(tmp_path).names()
        prompt = build_system_prompt(names)
        for name in names:
            assert f"- {name}(" in prompt
        assert '"final"' in prompt
        assert "Patch format for apply_patch" in prompt


class TestAgentLoop:
    def test_final_only_finishes_without_tools(self, tmp_path: Path) -> None:
        client = FakeClient(['{"final": "hello"}'])
        run = AgentLoop(_registry(tmp_path), client).run("say hi")
        assert run.finished is True
        assert run.final == "hello"
        assert run.tool_calls == 0
        assert run.steps[0].kind == "final"
        assert client.kwargs_seen[0]["json_schema"]["type"] == "object"

    def test_tool_call_then_final(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello world\n")
        client = FakeClient(
            [
                '{"tool": "read_file", "args": {"path": "a.txt"}}',
                '{"final": "done"}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client).run("read a.txt")
        assert run.finished is True
        assert run.tool_calls == 1
        step = run.steps[0]
        assert step.result is not None
        assert step.result.status == ToolStatus.OK
        assert "hello world" in step.result.output
        follow_up = client.requests[1]
        assert "Tool result:" in follow_up[-1]["content"]

    def test_unknown_tool_returns_envelope_and_continues(self, tmp_path: Path) -> None:
        client = FakeClient(['{"tool": "nope", "args": {}}', '{"final": "ok"}'])
        run = AgentLoop(_registry(tmp_path), client).run("try a bad tool")
        assert run.finished is True
        first = run.steps[0]
        assert first.result is not None
        assert first.result.status == ToolStatus.ERROR

    def test_bad_tool_args_return_envelope_and_continue(self, tmp_path: Path) -> None:
        client = FakeClient(
            [
                '{"tool": "read_file", "args": {"bogus_kwarg": 1}}',
                '{"final": "ok"}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client).run("bad args")
        assert run.finished is True
        first = run.steps[0]
        assert first.result is not None
        assert first.result.status == ToolStatus.ERROR

    def test_repeated_failing_call_gets_warning(self, tmp_path: Path) -> None:
        call = '{"tool": "read_file", "args": {"path": "missing.txt"}}'
        client = FakeClient([call, call, '{"final": "gave up"}'])
        run = AgentLoop(_registry(tmp_path), client).run("retry test")
        assert run.finished is True
        third_request = client.requests[2][-1]["content"]
        assert "WARNING: identical call already failed" in third_request

    def test_protocol_violation_gets_reminder_then_recovers(self, tmp_path: Path) -> None:
        client = FakeClient(["just some prose", '{"final": "ok"}'])
        run = AgentLoop(_registry(tmp_path), client).run("task")
        assert run.finished is True
        assert run.steps[0].kind == "protocol"
        reminder = client.requests[1][-1]["content"]
        assert "ONE JSON object" in reminder

    def test_budget_exhaustion_is_honest(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x\n")
        call = '{"tool": "read_file", "args": {"path": "a.txt"}}'
        client = FakeClient([call, call, call])
        run = AgentLoop(_registry(tmp_path), client, max_steps=2).run("loop forever")
        assert run.finished is False
        assert run.final is None
        assert run.tool_calls == 2

    def test_denied_path_arrives_as_envelope(self, tmp_path: Path) -> None:
        client = FakeClient(
            [
                '{"tool": "read_file", "args": {"path": "../../etc/passwd"}}',
                '{"final": "cannot read that"}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client).run("escape attempt")
        first = run.steps[0]
        assert first.result is not None
        assert first.result.status == ToolStatus.DENIED
        assert run.finished is True
