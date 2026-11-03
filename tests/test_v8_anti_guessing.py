"""v8: anti-guessing teachings — (a) repeated test failures demand derivation
instead of value-guessing (round 6 debug-cross: TAX_RATE guessed 3×);
(b) repeated invented names escalate to a user-role reminder (round 6
typo-hunt: 'frobnicate' written although only 'FROBNICATER' was seen)."""

from __future__ import annotations

import json
from pathlib import Path

from flashmini.agent import AgentLoop
from flashmini.tools import ToolContext, build_default_registry
from test_agent_loop import FakeClient


def test_second_failed_run_tests_demands_derivation(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "m.py").write_text("RATE = 1\n")
    calls = [
        json.dumps({"tool": "write_file", "args": {"path": "app/m.py", "content": "RATE = 2\n"}}),
        json.dumps({"tool": "run_tests", "args": {}}),
        json.dumps({"tool": "write_file", "args": {"path": "app/m.py", "content": "RATE = 3\n"}}),
        json.dumps({"tool": "run_tests", "args": {}}),
        json.dumps({"final": "done"}),
    ]
    # No real test suite: run_tests will fail (no pytest project) — the point
    # is the escalation warning after the 2nd failed run_tests call.
    client = FakeClient(calls)
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("fix the rate")

    last = client.requests[-1][-1]["content"]
    assert "Stop" in last and "guessing values by trial and error" in last
    assert "DERIVE" in last
    # v9: observability teach — print() + pytest -s to see actual values.
    assert "pytest -s" in last


def test_derive_warning_suppresses_generic_repeat_warnings(tmp_path: Path) -> None:
    """Audit I2: when the derive warning fires for run_tests, the generic
    'identical call already failed' / 'this exact call has now failed'
    warnings must NOT stack on top (warning wall)."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "m.py").write_text("RATE = 1\n")
    client = FakeClient(
        [
            json.dumps({"tool": "run_tests", "args": {}}),
            json.dumps({"tool": "run_tests", "args": {}}),
            json.dumps({"final": "done"}),
        ]
    )
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("fix the rate")

    last = client.requests[-1][-1]["content"]
    assert "DERIVE" in last  # v8 warning fired (2nd failed run_tests)
    assert "identical call already failed" not in last
    assert "this exact call has now failed" not in last


def test_derive_counter_resets_after_green_run(tmp_path: Path) -> None:
    """Audit M1: a passing run_tests clears the trial-and-error counter —
    after green, a single new failure must NOT claim a stale streak."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "m.py").write_text("RATE = 1\n")
    client = FakeClient(
        [
            json.dumps({"tool": "run_tests", "args": {}}),  # fail 1 (no tests: rc 5)
            json.dumps({"tool": "run_tests", "args": {}}),  # fail 2 → derive fires
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "test_ok.py", "content": "def test_ok():\n    assert True\n"},
                }
            ),
            json.dumps({"tool": "run_tests", "args": {}}),  # green → reset
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "test_ok.py", "content": "def test_ok():\n    assert False\n"},
                }
            ),
            json.dumps({"tool": "run_tests", "args": {}}),  # new streak fail 1
            json.dumps({"final": "done"}),
        ]
    )
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("fix the rate")

    last = client.requests[-1][-1]["content"]
    assert "guessing values by trial and error" not in last, (
        "stale counter must not fire the derive warning after a green run"
    )


def test_second_unseen_name_write_escalates(tmp_path: Path) -> None:
    (tmp_path / "lib.py").write_text("KNOWN_NAME = True\n")
    client = FakeClient(
        [
            # 1st modification with an invented name (vocab lacks it)
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "lib.py", "content": "KNOWN_NAME = True\nINVENTED_A = 1\n"},
                }
            ),
            json.dumps({"tool": "read_file", "args": {"path": "lib.py"}}),
            # 2nd modification with another invented name → escalation
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "lib.py", "content": "KNOWN_NAME = True\nINVENTED_B = 2\n"},
                }
            ),
            json.dumps({"final": "done"}),
        ]
    )
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("update lib.py")

    escalations = [
        m
        for req in client.requests
        for m in req
        if m["role"] == "user" and m["content"].startswith("STOP guessing spellings")
    ]
    assert escalations, "expected user-role invented-name reminder on 2nd unseen write"
    assert "2 times" in escalations[0]["content"]
    assert "grep" in escalations[0]["content"]


def test_escalation_stops_once_names_are_known(tmp_path: Path) -> None:
    """Once the model writes clean content, no further invented-name
    reminders may be appended (spam guard)."""
    (tmp_path / "lib.py").write_text("KNOWN_NAME = True\n")
    client = FakeClient(
        [
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "lib.py", "content": "KNOWN_NAME = True\nINVENTED_A = 1\n"},
                }
            ),
            json.dumps({"tool": "read_file", "args": {"path": "lib.py"}}),
            json.dumps(
                {
                    "tool": "write_file",
                    "args": {"path": "lib.py", "content": "KNOWN_NAME = True\nINVENTED_B = 2\n"},
                }
            ),
            # Model corrected itself: clean write → must stay quiet
            json.dumps(
                {"tool": "write_file", "args": {"path": "lib.py", "content": "KNOWN_NAME = True\n"}}
            ),
            json.dumps({"tool": "read_file", "args": {"path": "lib.py"}}),
            json.dumps({"final": "done"}),
        ]
    )
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("update lib.py")

    # The reminder from the 2nd unseen write legitimately stays in history;
    # clean steps must NOT append new ones (spam guard). Final request's
    # history must contain exactly one such message.
    reminders = [
        m for m in client.requests[-1] if m["content"].startswith("STOP guessing spellings")
    ]
    assert len(reminders) == 1, f"expected 1 reminder in history, got {len(reminders)}"
