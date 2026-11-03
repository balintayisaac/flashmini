"""v7: doom-loop escalation — repeated identical no-op writes must trigger
a user-role reminder demanding a different implementation (round 5:
implement-spec burned 6 steps re-writing byte-identical content)."""

from __future__ import annotations

import json
from pathlib import Path

from flashmini.agent import AgentLoop
from flashmini.tools import ToolContext, build_default_registry
from test_agent_loop import FakeClient

SAME = "def f():\n    return 1\n"
_REPLIES = [
    # 1st write lands (creates the file)
    json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
    # 2nd write: no-op #1 (prefix warning at >=2 next turn)
    json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
    # 3rd write: no-op #2 (>=2 prefix warning)
    json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
    # 4th write: no-op #3 (doom-loop reminder fires)
    json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
    json.dumps({"final": "done"}),
]


def test_third_identical_noop_write_injects_doom_loop_reminder(tmp_path: Path) -> None:
    client = FakeClient(list(_REPLIES))
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("fix m.py")

    # Find the user-role reminder appended after tool-result messages.
    doom = [
        m
        for m in client.requests[-1]
        if m["role"] == "user" and m["content"].startswith("STOP re-writing the same file")
    ]
    assert doom, "expected a user-role doom-loop reminder after 3 identical writes"
    text = doom[0]["content"]
    assert "m.py" in text
    assert "3 times" in text
    # It demands a different implementation, not a repeat.
    assert "DIFFERENT implementation" in text


def test_single_noop_write_does_not_trigger_reminder(tmp_path: Path) -> None:
    client = FakeClient(
        [
            json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
            json.dumps({"tool": "write_file", "args": {"path": "m.py", "content": SAME}}),
            json.dumps({"final": "done"}),
        ]
    )
    loop = AgentLoop(build_default_registry(ToolContext(workspace=tmp_path)), client)
    loop.run("fix m.py")

    for req in client.requests:
        for m in req:
            assert not m["content"].startswith("STOP re-writing the same file")
