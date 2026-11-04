"""Tests for kaggle/kaggle_kernel.py — cloud SFT pipeline helpers.

The module imports cleanly without torch/datasets (heavy imports live
inside function bodies). _clean_messages is the v7 crash fix: it
canonicalizes UltraData chat rows to uniform {role, content, tool_calls}
messages (subfield type mixing crashed Dataset.from_list with ArrowInvalid
in the v5/v6 kernels).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "kaggle_kernel.py"
_spec = importlib.util.spec_from_file_location("kaggle_kernel", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
kaggle_kernel = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kaggle_kernel
_spec.loader.exec_module(kaggle_kernel)

_clean = kaggle_kernel._clean_messages


class TestCleanMessages:
    def test_rejects_non_lists(self):
        for bad in (None, "hello", {"role": "user"}, [123], 42):
            assert _clean(bad) is None

    def test_rejects_empty_list(self):
        assert _clean([]) is None

    def test_rejects_missing_or_non_str_role(self):
        assert _clean([{"content": "hi"}]) is None
        assert _clean([{"role": 7, "content": "x"}]) is None

    def test_rejects_untrainable_turns(self):
        assert _clean([{"role": "assistant", "content": None}]) is None
        assert _clean([{"role": "assistant", "content": "   "}]) is None
        assert _clean([{"role": "user", "content": [{"type": "image"}]}]) is None

    def test_accepts_plain_turn(self):
        clean = _clean([{"role": "user", "content": "hi"}])
        assert clean == [{"role": "user", "content": "hi", "tool_calls": None}]

    def test_strips_extra_keys(self):
        clean = _clean(
            [{"role": "assistant", "content": "x", "reasoning_content": "r", "loss": "False"}]
        )
        assert clean == [{"role": "assistant", "content": "x", "tool_calls": None}]

    def test_extracts_text_parts_from_list_content(self):
        clean = _clean(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "url": "u"},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        )
        assert clean == [{"role": "user", "content": "hello world", "tool_calls": None}]

    def test_keeps_empty_content_when_tool_calls_present(self):
        clean = _clean(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "read_file", "arguments": {"file_path": "/a"}},
                        }
                    ],
                }
            ]
        )
        assert clean == [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"file_path": "/a"}'},
                    }
                ],
            }
        ]

    def test_arguments_normalization(self):
        def one(args):
            return _clean(
                [
                    {
                        "role": "assistant",
                        "content": "x",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "f", "arguments": args}}
                        ],
                    }
                ]
            )[0]["tool_calls"][0]["function"]["arguments"]

        assert one({"a": [1, 2]}) == '{"a": [1, 2]}'
        assert one([1, 2]) == "[1, 2]"
        assert one(None) == ""
        assert one(5) == "5"
        assert one("raw") == "raw"

    def test_empty_tool_calls_list_treated_as_absent(self):
        clean = _clean([{"role": "user", "content": "x", "tool_calls": []}])
        assert clean == [{"role": "user", "content": "x", "tool_calls": None}]
        assert _clean([{"role": "user", "content": "", "tool_calls": []}]) is None

    def test_rejects_malformed_tool_calls(self):
        base = {"role": "assistant", "content": "x"}
        assert _clean([{**base, "tool_calls": {"name": "f"}}]) is None
        assert _clean([{**base, "tool_calls": [{"type": "function"}]}]) is None
        assert _clean([{**base, "tool_calls": [{"function": {"arguments": "{}"}}]}]) is None
        assert _clean([{**base, "tool_calls": ["nope"]}]) is None

    def test_accepts_multi_turn_dialogue_with_tool_calls(self):
        clean = _clean(
            [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "result"},
            ]
        )
        assert [m["role"] for m in clean or []] == ["system", "assistant", "tool"]


class TestArrowUniformity:
    """Reproduce the v5/v6 Kaggle crash shapes and the fix (needs datasets)."""

    @staticmethod
    def _from_list(rows):
        datasets = pytest.importorskip("datasets")
        return datasets.Dataset.from_list(rows)

    def test_mixed_tool_calls_shape_crashes(self):
        rows = [
            {"messages": [{"role": "assistant", "content": "x", "tool_calls": [{"name": "f"}]}]},
            {"messages": [{"role": "assistant", "content": "y", "tool_calls": {"name": "f"}}]},
        ]
        with pytest.raises(Exception, match="mix list and non-list"):
            self._from_list(rows)

    def test_mixed_content_shape_crashes(self):
        rows = [
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        ]
        with pytest.raises(Exception, match="mix list and non-list"):
            self._from_list(rows)

    def test_cleaned_rows_build_uniformly(self):
        cleaned = [
            _clean([{"role": "user", "content": "a"}]),
            _clean(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
                        ],
                    }
                ]
            ),
            _clean([{"role": "tool", "content": "r"}]),
        ]
        rows = [{"messages": c} for c in cleaned if c is not None]
        ds = self._from_list(rows)
        assert ds.num_rows == 3
        assert ds[0]["messages"][0]["content"] == "a"
        assert ds[1]["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
        assert ds[2]["messages"][0]["tool_calls"] is None

    def test_uniform_str_content_builds(self):
        ds = self._from_list(
            [
                {"messages": [{"role": "user", "content": "hi"}]},
                {"messages": [{"role": "assistant", "content": "yo"}]},
            ]
        )
        assert ds.num_rows == 2
