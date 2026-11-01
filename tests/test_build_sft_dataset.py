"""TDD for the verified-trace SFT dataset builder (Track B phase 0).

The builder converts passing AgentBench trajectories into chat-SFT
samples whose message structure matches the runtime loop exactly:
system → user(task) → (assistant(raw JSON) → user(tool-result))* →
assistant(final JSON).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_sft_dataset.py"
_spec = importlib.util.spec_from_file_location("build_sft_dataset", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder
_spec.loader.exec_module(builder)


def _write_results(tmp_path: Path, results: list[dict]) -> None:
    payload = {
        "stamp": "20261101-000000",
        "passed": sum(1 for r in results if r["passed"]),
        "total": len(results),
        "results": results,
    }
    (tmp_path / "agentbench-20261101-000000.json").write_text(json.dumps(payload))


def _tool_step(index: int, tool: str, args: dict, output: str) -> dict:
    return {
        "index": index,
        "kind": "tool",
        "tool": tool,
        "args": args,
        "status": "ok",
        "output": output,
        "hint": None,
    }


def test_passing_trajectory_becomes_messages(tmp_path: Path) -> None:
    steps = [
        _tool_step(0, "grep", {"pattern": "needle"}, "b.py:1:needle"),
        {"index": 1, "kind": "final", "raw": '{"final": "b.py"}'},
    ]
    _write_results(tmp_path, [{"name": "find-text", "passed": True, "steps": steps}])

    samples, stats = builder.collect_samples(tmp_path)
    assert stats["passed_runs"] == 1
    assert stats["excluded_failed"] == 0
    assert len(samples) == 1

    msgs = samples[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "FlashMini" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == builder.load_task_prompts()["find-text"]
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == json.dumps(
        {"tool": "grep", "args": {"pattern": "needle"}}, ensure_ascii=False
    )
    assert msgs[3]["role"] == "user"
    assert msgs[3]["content"].startswith("Tool result:\n")
    envelope = json.loads(msgs[3]["content"].removeprefix("Tool result:\n"))
    assert envelope["tool"] == "grep"
    assert envelope["status"] == "ok"
    assert envelope["output"] == "b.py:1:needle"
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == '{"final": "b.py"}'


def test_failed_and_protocol_runs_excluded(tmp_path: Path) -> None:
    good = [
        {
            "name": "find-text",
            "passed": True,
            "steps": [
                {"index": 0, "kind": "final", "raw": '{"final": "ok"}'},
            ],
        }
    ]
    failed = [
        {
            "name": "find-text",
            "passed": False,
            "steps": [
                {"index": 0, "kind": "final", "raw": '{"final": "gave up"}'},
            ],
        }
    ]
    protocol = [
        {
            "name": "find-text",
            "passed": True,
            "steps": [
                {"index": 0, "kind": "protocol", "raw": "I plan to..."},
                {"index": 1, "kind": "final", "raw": '{"final": "ok"}'},
            ],
        }
    ]
    _write_results(tmp_path, good + failed + protocol)

    samples, stats = builder.collect_samples(tmp_path)
    assert len(samples) == 1
    assert stats["excluded_failed"] == 1
    assert stats["excluded_protocol"] == 1


def test_truncated_final_excluded(tmp_path: Path) -> None:
    _write_results(
        tmp_path,
        [
            {
                "name": "find-text",
                "passed": True,
                "steps": [
                    {"index": 0, "kind": "final", "raw": "f" * 300},
                ],
            }
        ],
    )

    samples, stats = builder.collect_samples(tmp_path)
    assert samples == []
    assert stats["excluded_truncated_final"] == 1


def test_identical_trajectories_deduplicated(tmp_path: Path) -> None:
    run = {
        "name": "find-text",
        "passed": True,
        "steps": [
            {"index": 0, "kind": "final", "raw": '{"final": "same"}'},
        ],
    }
    _write_results(tmp_path, [run, dict(run)])

    samples, stats = builder.collect_samples(tmp_path)
    assert len(samples) == 1
    assert stats["duplicates_dropped"] == 1
