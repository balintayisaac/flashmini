"""Regression tests for the AgentBench v1 fix batch: planning guard + patch teaching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flashmini.agent import AgentLoop
from flashmini.tools import ToolContext, ToolStatus, apply_patch, build_default_registry


class FakeClient:
    """Scripted replacement for ChatClient: pops replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.requests.append(list(messages))
        return self.replies.pop(0) if self.replies else '{"final": "out of scripted replies"}'


def _registry(tmp_path: Path):
    return build_default_registry(ToolContext(workspace=tmp_path))


class TestPlanningGuard:
    def test_planning_final_is_rejected_and_reminder_given(self, tmp_path: Path) -> None:
        client = FakeClient(
            ['{"final": "Let me first read the file to check its contents."}', '{"final": "done"}']
        )
        run = AgentLoop(_registry(tmp_path), client).run("do something")
        assert run.steps[0].kind == "protocol"
        assert run.finished is True
        assert run.final == "done"
        assert "plan, not a completed answer" in client.requests[1][-1]["content"]

    def test_completed_final_is_accepted_immediately(self, tmp_path: Path) -> None:
        client = FakeClient(['{"final": "The bug is fixed and the tests pass."}'])
        run = AgentLoop(_registry(tmp_path), client).run("t")
        assert run.finished is True
        assert run.steps[0].kind == "final"


class TestPatchTeaching:
    def test_display_prefix_in_hunk_is_explained(self, tmp_path: Path) -> None:
        ctx = ToolContext(workspace=tmp_path)
        target = tmp_path / "m.py"
        target.write_text("def f():\n    return 1\n")
        patch = (
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            "     1\tdef f():\n     2\t    return 1\n"
            "-    return 1\n+    return 2\n*** End Patch"
        )
        result = apply_patch(ctx, patch)
        assert result.status == ToolStatus.ERROR
        assert "display formatting" in result.output
        assert "Expected format:" in result.output
        assert target.read_text() == "def f():\n    return 1\n"

    def test_mismatch_shows_nearest_raw_content(self, tmp_path: Path) -> None:
        ctx = ToolContext(workspace=tmp_path)
        target = tmp_path / "m.py"
        target.write_text("def f():\n        return 1\n")
        patch = (
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            "  return 1\n-  return 1\n+  return 2\n*** End Patch"
        )
        result = apply_patch(ctx, patch)
        assert result.status == ToolStatus.ERROR
        assert "Nearest raw file content" in result.output
        assert "        return 1" in result.output
        assert "Leading whitespace differs" in result.output
        assert target.read_text() == "def f():\n        return 1\n"


class TestPatchFallbackWarning:
    def test_two_patch_failures_suggest_write_file(self, tmp_path: Path) -> None:
        client = FakeClient(
            [
                '{"tool": "apply_patch", "args": {"patch": "not a patch"}}',
                '{"tool": "apply_patch", "args": {"patch": "still not a patch"}}',
                '{"final": "done"}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client).run("t")
        assert run.finished is True
        second_envelope = client.requests[2][-1]["content"]
        assert "switch approach" in second_envelope
        assert "write_file" in second_envelope
