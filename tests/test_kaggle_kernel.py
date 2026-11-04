"""Tests for kaggle/kaggle_kernel.py — cloud SFT pipeline helpers.

The module imports cleanly without torch/datasets (heavy imports live
inside function bodies). _clean_messages is the v7 crash fix: it
canonicalizes UltraData chat rows to uniform {role, content, tool_calls}
messages (subfield type mixing crashed Dataset.from_list with ArrowInvalid
in the v5/v6 kernels). _msg_features + from_list(features=...) is the v8
crash fix: raw trace rows (no tool_calls key) and cleaned corpus rows
inferred divergent messages features, which made concatenate_datasets
raise "features can't be aligned" in the v7 kernel.
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


class TestBlendAlignment:
    """The v7 crash: concatenate_datasets requires identical features.

    Our traces carry messages WITHOUT a tool_calls key (tool use is
    serialized inside content), so the raw traces part inferred
    List({role, content}) while cleaned corpus parts inferred
    List({role, content, tool_calls: List(...)}) — "features can't be
    aligned". Fix (v8): every part is built with explicit canonical
    features from _msg_features(). (Needs datasets.)
    """

    @staticmethod
    def _canon():
        pytest.importorskip("datasets")
        return kaggle_kernel._msg_features()

    @staticmethod
    def _traces_like_rows():
        """Mirror the real 98-trace shape: {role, content} only."""
        return [
            {
                "messages": [
                    {"role": "system", "content": "You are FlashMini."},
                    {"role": "user", "content": "implement spec"},
                    {"role": "assistant", "content": '{"tool": "read_file", "args": {}}'},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "fix the bug"},
                    {"role": "assistant", "content": "done"},
                ]
            },
        ]

    @staticmethod
    def _corpus_like_rows():
        """Mirror plan-B-cleaned corpus rows: tool_calls always present."""
        cleaned = _clean(
            [
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
                    ],
                },
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "done"},
            ]
        )
        assert cleaned is not None
        return [{"messages": cleaned}]

    def test_v7_alignment_crash_reproduced(self):
        datasets = pytest.importorskip("datasets")
        traces = datasets.Dataset.from_list(self._traces_like_rows())
        corpus = datasets.Dataset.from_list(self._corpus_like_rows())
        with pytest.raises(Exception, match="can't be aligned"):
            datasets.concatenate_datasets([traces, corpus])

    def test_canonical_features_align_traces_and_corpus(self):
        datasets = pytest.importorskip("datasets")
        canon = self._canon()
        traces = datasets.Dataset.from_list(self._traces_like_rows(), features=canon)
        corpus = datasets.Dataset.from_list(self._corpus_like_rows(), features=canon)
        blended = datasets.concatenate_datasets([traces, corpus]).shuffle(seed=42)
        assert blended.num_rows == 3
        # trace messages gain the tool_calls key (None) via canonicalization
        assert blended[0]["messages"][0]["tool_calls"] is None
        assert set(blended[0]["messages"][0].keys()) == {"role", "content", "tool_calls"}
        # corpus messages keep their tool_calls structure
        has_tc = any(m["tool_calls"] is not None for row in blended for m in row["messages"])
        assert has_tc

    def test_all_none_tool_calls_casts_to_canonical(self):
        """The traces part has zero tool_calls; from_list must still build
        the List(struct) schema instead of Value('null')."""
        datasets = pytest.importorskip("datasets")
        canon = self._canon()
        traces = datasets.Dataset.from_list(self._traces_like_rows(), features=canon)
        tc_feature = traces.features["messages"].feature["tool_calls"]
        assert not isinstance(tc_feature, datasets.Value)
