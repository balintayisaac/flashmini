"""Tests for kaggle/kaggle_kernel.py — cloud SFT pipeline helpers.

The module imports cleanly without torch/datasets (heavy imports live inside
function bodies, and TRACES_PATH resolves lazily), so it can be spec-loaded
by path the same way tests/test_kaggle_ultra.py loads train/kaggle_ultra.py.

Focus: `_valid_messages()` — the guard that keeps UltraData rows whose
`messages` field is a plain string (which apply_chat_template accepts as a
single user turn) out of the blended dataset. Without it, mixing those rows
with our list-of-dict traces poisons Dataset.from_list with
`pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values`
(the v5 Kaggle kernel failure).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "kaggle_kernel.py"
_spec = importlib.util.spec_from_file_location("kaggle_kernel", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
kaggle_kernel = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kaggle_kernel
_spec.loader.exec_module(kaggle_kernel)


class TestValidMessages:
    def test_rejects_non_lists(self) -> None:
        for bad in (None, "hello", {"role": "user"}, [123], 42):
            assert not kaggle_kernel._valid_messages(bad), bad

    def test_rejects_empty_string_inside_list(self) -> None:
        assert not kaggle_kernel._valid_messages([])
        assert not kaggle_kernel._valid_messages(["hi"])

    def test_rejects_items_without_role(self) -> None:
        assert not kaggle_kernel._valid_messages([{"content": "hi"}])
        assert not kaggle_kernel._valid_messages(
            [{"role": "user", "content": "hi"}, {"content": "missing role"}]
        )

    def test_accepts_well_formed_turns(self) -> None:
        assert kaggle_kernel._valid_messages([{"role": "user", "content": "hi"}])
        # content optional: tool-call turns carry role + tool_calls
        assert kaggle_kernel._valid_messages([{"role": "assistant"}])

    def test_accepts_two_turn_dialogue(self) -> None:
        convo = [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "type": "function"}],
            },
            {"role": "tool", "content": "ok"},
        ]
        assert kaggle_kernel._valid_messages(convo)
