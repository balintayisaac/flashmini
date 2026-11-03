"""v11: context-overflow handling (round 7 exposed it: 4447-token request
against a 4096-token context crashed implement-spec with tools=[])."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from flashmini.agent.loop import AgentLoop, envelope_message
from flashmini.runtime.server import ServerError
from flashmini.tools import ToolContext, ToolResult, ToolStatus, build_default_registry

_SPEC = importlib.util.spec_from_file_location(
    "flashmini_tests_test_v11",
    Path(__file__).resolve(),
)


def _registry(tmp_path: Path):
    return build_default_registry(ToolContext(workspace=tmp_path))


class OverflowClient:
    """FakeClient that raises exceed-context ServerError on the Nth call."""

    def __init__(self, replies: list[str], fail_on_call: int) -> None:
        self.replies = list(replies)
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.requests: list[list[dict[str, Any]]] = []

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls += 1
        self.requests.append(list(messages))
        if self.calls == self.fail_on_call:
            raise ServerError(
                "HTTP 400 from llama-server /v1/chat/completions: request "
                "(4447 tokens) exceeds the available context size (4096 tokens)"
            )
        if not self.replies:
            return '{"final": "out of scripted replies"}'
        return self.replies.pop(0)

    def chat_result(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, str | None]:
        return self.chat_text(messages, **kwargs), "stop"


class TestEnvelopeTrim:
    def test_long_output_keeps_head_and_tail_with_marker(self) -> None:
        result = ToolResult(
            tool="run_tests",
            status=ToolStatus.OK,
            output="HEADMARKER" + "x" * 5000 + "TAILMARKER",
        )
        envelope = json.loads(envelope_message(result))
        out = envelope["output"]
        assert "[3020 chars omitted]" in out  # 5020 total - 2000 kept
        assert out.startswith("HEADMARKER")
        assert out.endswith("TAILMARKER")
        assert len(out) < 2400

    def test_short_output_untouched(self) -> None:
        result = ToolResult(tool="read_file", status=ToolStatus.OK, output="tiny")
        envelope = json.loads(envelope_message(result))
        assert envelope["output"] == "tiny"
        assert "omitted" not in envelope["output"]


class TestContextCompaction:
    def test_overflow_compacts_older_tool_results_and_retries(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello\n")
        (tmp_path / "b.txt").write_text("world\n")
        client = OverflowClient(
            [
                '{"tool": "read_file", "args": {"path": "a.txt"}}',
                '{"tool": "read_file", "args": {"path": "b.txt"}}',
                '{"final": "done"}',
            ],
            fail_on_call=3,
        )
        run = AgentLoop(_registry(tmp_path), client).run("read both files then finish")
        assert run.finished is True
        assert run.final == "done"
        assert client.calls == 4
        retried = client.requests[3]
        tool_results = [
            m["content"]
            for m in retried
            if m["role"] == "user" and "Tool result" in str(m["content"])
        ]
        assert any("earlier tool result omitted" in c for c in tool_results)
        # exactly one compaction: the most recent tool result stays intact
        placeholders = [c for c in tool_results if "earlier tool result omitted" in c]
        assert len(placeholders) == 1

    def test_non_context_server_error_propagates(self, tmp_path: Path) -> None:
        client = OverflowClient([], fail_on_call=1)
        client.fail_on_call = 1
        # make the error message unrelated to context size
        original = OverflowClient.chat_text

        def other_error(self: OverflowClient, messages: Any, **kw: Any) -> str:
            self.calls += 1
            self.requests.append(list(messages))
            raise ServerError("HTTP 500 from llama-server: backend crashed")

        OverflowClient.chat_text = other_error  # type: ignore[method-assign]
        try:
            with pytest.raises(ServerError):
                AgentLoop(_registry(tmp_path), client).run("anything")
        finally:
            OverflowClient.chat_text = original  # type: ignore[method-assign]
