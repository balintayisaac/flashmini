"""v10 guards: per-run repeat counters + no-op-write escalation.

Probe evidence (v9 probe, 20261101-154411):
- refactor-sig: the model re-wrote identical content to app/calc.py four times
  (no-op writes interleaved with reads) and never touched the other files that
  still used the old name.
- implement-spec: the model re-read a nonexistent app/slug.py ten times; the
  "file not found" hint never told it the file must be CREATED with write_file.
The consecutive-only repeat guards reset whenever any other call intervenes,
so they never fired in either trace. These tests pin per-run counters and a
redirecting no-op escalation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flashmini.agent import AgentLoop
from flashmini.tools import ToolContext, ToolStatus, build_default_registry, read_file


class FakeClient:
    """Scripted replacement for ChatClient: pops replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.requests.append(list(messages))
        if not self.replies:
            return '{"final": "out of scripted replies"}'
        return self.replies.pop(0)

    def chat_result(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, str | None]:
        return self.chat_text(messages, **kwargs), "stop"


def _registry(tmp_path: Path):
    return build_default_registry(ToolContext(workspace=tmp_path))


def test_identical_failure_warns_even_non_consecutive(tmp_path: Path) -> None:
    """Two identical failing calls with an OK call in between must warn."""
    (tmp_path / "a.txt").write_text("hello")
    client = FakeClient(
        [
            '{"tool": "read_file", "args": {"path": "missing.txt"}}',
            '{"tool": "read_file", "args": {"path": "a.txt"}}',
            '{"tool": "read_file", "args": {"path": "missing.txt"}}',
        ]
    )
    loop = AgentLoop(_registry(tmp_path), client)
    loop.run("task")
    last = client.requests[-1][-1]["content"]
    assert "failed 2 times" in last


def test_noop_writes_escalate_to_look_elsewhere(tmp_path: Path) -> None:
    """A second no-op write to the same path must redirect to the other files."""
    (tmp_path / "a.txt").write_text("X")
    write_call = '{"tool": "write_file", "args": {"path": "a.txt", "content": "X"}}'
    client = FakeClient(
        [
            write_call,  # no-op #1 (content already identical)
            '{"tool": "read_file", "args": {"path": "a.txt"}}',
            write_call,  # no-op #2 → escalation
        ]
    )
    loop = AgentLoop(_registry(tmp_path), client)
    loop.run("task")
    last = client.requests[-1][-1]["content"]
    # v7 wording: if tests still fail on this file, the repeated content is
    # NOT the fix — change the implementation; otherwise look elsewhere.
    assert "THIS CONTENT IS NOT THE FIX" in last
    assert "run_tests" in last
    assert "a.txt" in last


def test_identical_ok_calls_warn_non_consecutively(tmp_path: Path) -> None:
    """Three identical OK calls with different calls in between must warn."""
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    read_a = '{"tool": "read_file", "args": {"path": "a.txt"}}'
    read_b = '{"tool": "read_file", "args": {"path": "b.txt"}}'
    client = FakeClient([read_a, read_b, read_a, read_b, read_a])
    loop = AgentLoop(_registry(tmp_path), client)
    loop.run("task")
    last = client.requests[-1][-1]["content"]
    assert "3 times with identical results" in last


def test_read_file_missing_hint_suggests_create_or_list(tmp_path: Path) -> None:
    """implement-spec trace: the model re-read a file it was supposed to CREATE;
    the hint must point at write_file (creation) and directory listing."""
    result = read_file(ToolContext(workspace=tmp_path), "app/slug.py")
    assert result.status == ToolStatus.ERROR
    hint = result.next_action_hint or ""
    assert "write_file" in hint
    assert "directory" in hint
