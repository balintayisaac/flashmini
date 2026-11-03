"""Regression tests for the 4-way audit remediation (Nov 3, 2026).

Each test pins a finding: patch duplicate-sections, CRLF preservation,
decorated-tool-result compaction, planning-guard false positives, unseen-name
evidence from reads, invented-path errors, truncation reminders, and the
TypeError mislabel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flashmini.agent.loop import TRUNCATION_REMINDER, AgentLoop
from flashmini.runtime.server import ServerError
from flashmini.tools import ToolContext, ToolResult, ToolStatus, build_default_registry
from flashmini.tools.patch import PatchError, parse_patch
from test_agent_loop import FakeClient


def _registry(tmp_path: Path):
    return build_default_registry(ToolContext(workspace=tmp_path))


class TestPatchDuplicateSections:
    def test_duplicate_update_sections_rejected(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: a.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** Update File: a.txt\n"
            "@@\n"
            "-two\n"
            "+TWO\n"
            "*** End Patch\n"
        )
        try:
            parse_patch(patch)
            raise AssertionError("expected PatchError")
        except PatchError as exc:
            assert "duplicate" in str(exc)

    def test_delete_then_update_rejected(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: a.txt\n"
            "*** Update File: a.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** End Patch\n"
        )
        try:
            parse_patch(patch)
            raise AssertionError("expected PatchError")
        except PatchError as exc:
            assert "duplicate" in str(exc)


class TestCrlfPreservation:
    def test_crlf_file_stays_crlf(self, tmp_path: Path) -> None:
        target = tmp_path / "cfg.txt"
        target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
        registry = _registry(tmp_path)
        patch = "*** Begin Patch\n*** Update File: cfg.txt\n@@\n-beta\n+BETA\n*** End Patch\n"
        result = registry.call("apply_patch", patch=patch)
        assert result.status is ToolStatus.OK, result.output
        assert b"BETA\r\n" in target.read_bytes()
        assert b"\r\n" in target.read_bytes()


class TestCompactionCatchesDecoratedResults:
    def test_warning_decorated_results_are_compacted(self, tmp_path: Path) -> None:
        # A tool result with a WARNING prepended used to be invisible to the
        # compaction selector (startswith "Tool result"), leaving overflow
        # unfixable (audit #3).
        from flashmini.agent.loop import _compact_messages

        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": '{"tool": "grep", "args": {}}'},
            {"role": "user", "content": "WARNING: x\nTool result:\nbody-one"},
            {"role": "assistant", "content": '{"tool": "grep", "args": {}}'},
            {"role": "user", "content": "Tool result:\nbody-two"},
        ]
        compact = _compact_messages(messages)
        assert "(earlier tool result omitted" in compact[3]["content"]
        assert compact[5]["content"] == "Tool result:\nbody-two"  # latest kept


class TestPlanningGuardEvidence:
    def test_completion_report_after_work_is_accepted(self, tmp_path: Path) -> None:
        # "Done. I will not touch the other files" used to be discarded as a
        # plan even though work had landed (audit #4).
        (tmp_path / "f.txt").write_text("x\n")
        client = FakeClient(
            [
                '{"tool": "write_file", "args": {"path": "out.txt", "content": "hi\\n"}}',
                '{"final": "Done. I will not touch the other files."}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client, max_steps=4).run("task")
        assert run.finished
        assert run.final == "Done. I will not touch the other files."

    def test_pure_plan_before_any_work_is_still_rejected(self, tmp_path: Path) -> None:
        client = FakeClient(['{"final": "First, I will read the spec."}'])
        run = AgentLoop(_registry(tmp_path), client, max_steps=4).run("task")
        # The plan reply is rejected at step 0 (PLANNING_REMINDER sent back)…
        planning = [
            m
            for m in client.requests[1]
            if m["role"] == "user" and "not a completed answer" in m["content"]
        ]
        assert planning
        # …and the run only finishes on later scripted replies.
        assert run.finished  # '{"final": "out of scripted replies"}' is honest


class TestUnseenNameEvidence:
    def test_name_from_read_file_is_not_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "lib.py").write_text("def MAX_RETRIES_SET():\n    return 1\n")
        client = FakeClient(
            [
                '{"tool": "read_file", "args": {"path": "lib.py"}}',
                (
                    '{"tool": "write_file", "args": {"path": "use.py", '
                    '"content": "x = MAX_RETRIES_SET()\\n"}}'
                ),
                '{"final": "done"}',
            ]
        )
        run = AgentLoop(_registry(tmp_path), client, max_steps=6).run("task")
        warn_steps = [s for s in run.steps if "WARNING" in (getattr(s.result, "output", "") or "")]
        assert not warn_steps


class TestInventedPath:
    def test_grep_on_missing_path_is_error_with_hint(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)
        result = registry.call("grep", pattern="needle", path="ghost_dir/ghost.py")
        assert result.status is ToolStatus.ERROR
        assert "not found" in result.output.lower() + " " + result.next_action_hint


class TestTruncationReminder:
    def test_finish_reason_length_sends_truncation_hint(self, tmp_path: Path) -> None:
        class LengthCappedClient(FakeClient):
            def chat_result(self, messages, **kwargs):  # noqa: ANN001, ANN202, ANN003
                return self.chat_text(messages, **kwargs), "length"

        client = LengthCappedClient(["this is not json at all", '{"final": "ok"}'])
        run = AgentLoop(_registry(tmp_path), client, max_steps=4).run("task")
        reminder_msgs = [
            m
            for m in client.requests[1]
            if m["role"] == "user" and TRUNCATION_REMINDER in m["content"]
        ]
        assert reminder_msgs
        assert run.finished


class TestRegistryTypeErrorMessage:
    def test_internal_type_error_is_not_invalid_arguments(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)

        def broken(**kwargs: object) -> ToolResult:
            raise TypeError("cannot concatenate str to int")

        registry.register("broken", broken)
        result = registry.call("broken")
        assert "tool crashed" in result.output

    def test_binding_error_is_invalid_arguments(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)
        result = registry.call("read_file", nonexistent_kwarg="x")
        assert "invalid arguments" in result.output


class TestEnvAllowlist:
    def test_secrets_are_withheld_from_children(self, tmp_path: Path, monkeypatch: Any) -> None:
        from flashmini.exec.runner import _child_env

        monkeypatch.setenv("FLASHMINI_API_KEY", "nvapi-secret123456789")
        monkeypatch.setenv("FLASHMINI_RG", "/usr/bin/rg")
        monkeypatch.setenv("HOME", "/home/x")
        env = _child_env(None)
        assert "FLASHMINI_API_KEY" not in env
        assert env["FLASHMINI_RG"] == "/usr/bin/rg"
        assert env["HOME"] == "/home/x"

    def test_scrub_redacts_key_from_output(self, monkeypatch: Any) -> None:
        from flashmini.exec.runner import _scrub_secrets

        monkeypatch.setenv("FLASHMINI_API_KEY", "nvapi-secret123456789")
        out = _scrub_secrets("key=nvapi-secret123456789 end")
        assert "nvapi-secret123456789" not in out

    def test_scrub_redacts_bare_nvapi_pattern(self) -> None:
        from flashmini.exec.runner import _scrub_secrets

        out = _scrub_secrets("see nvapi-abcdefghijk123 below")
        assert "abcdefghijk123" not in out


def _unused_json_helper() -> str:  # keep json import meaningful for future edits
    return json.dumps({"ok": True})


class TestServerErrorFromConnectionLoss:
    def test_urlerror_retried_then_server_error(self, monkeypatch: Any) -> None:
        # Audit I11: connection resets now retry like 429/500 instead of
        # failing the trajectory on the first drop.
        import flashmini.runtime.server as srv

        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001, ANN202
            calls["n"] += 1
            raise OSError("connection reset by peer")

        monkeypatch.setattr(srv.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(srv, "_RETRY_DELAYS", (0.0, 0.0))
        client = srv.ChatClient("http://127.0.0.1:1/v1", model="m", api_key="k")
        with __import__("pytest").raises(ServerError):
            client.chat_text([{"role": "user", "content": "hi"}])
        assert calls["n"] == 3  # 2 retries then final raise
