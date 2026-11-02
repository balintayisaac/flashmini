"""Teacher-solve harness: prompt builder + bench-schema serializer contract.

The dataset builder (scripts/data/build_sft_dataset.py) only absorbs runs
whose results JSON matches the AgentBench step schema — these tests pin that
contract for distill runs (serializer output shape, truncation limits).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from flashmini.agent.loop import AgentRun, Step
from flashmini.tools import ToolResult, ToolStatus

_SPEC = importlib.util.spec_from_file_location(
    "teacher_solve",
    Path(__file__).resolve().parents[1] / "scripts" / "distill" / "teacher_solve.py",
)
assert _SPEC and _SPEC.loader
ts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ts
_SPEC.loader.exec_module(ts)


class TestTaskPrompt:
    def test_prompt_is_implement_spec_shaped(self) -> None:
        prompt = ts.task_prompt("slugify")
        assert "app/slugify.py" in prompt
        assert "SPEC.md" in prompt
        assert "make sure they pass" in prompt


class TestSerializeSteps:
    def test_tool_step_matches_bench_schema(self) -> None:
        result = ToolResult(
            tool="read_file",
            status=ToolStatus.OK,
            output="x" * 900,
            next_action_hint="try again",
        )
        run = AgentRun(
            task="t",
            final=None,
            finished=True,
            steps=[Step(index=0, kind="tool", tool="read_file", args={}, result=result)],
        )
        steps = ts.serialize_steps(run)
        assert len(steps) == 1
        s = steps[0]
        assert s["index"] == 0
        assert s["kind"] == "tool"
        assert s["tool"] == "read_file"
        assert s["status"] == "ok"
        assert len(s["output"]) == ts.STEP_OUTPUT_CHARS  # truncated like the bench
        assert s["hint"] == "try again"
        assert "raw" not in s

    def test_final_step_keeps_truncated_raw(self) -> None:
        run = AgentRun(
            task="t",
            final="done",
            finished=True,
            steps=[Step(index=1, kind="final", raw="y" * 400)],
        )
        steps = ts.serialize_steps(run)
        assert steps[0]["kind"] == "final"
        assert len(steps[0]["raw"]) == ts.STEP_RAW_CHARS
        assert "tool" not in steps[0]
