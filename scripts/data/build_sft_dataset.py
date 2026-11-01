#!/usr/bin/env python3
"""Build the FlashMini Ultra SFT dataset from verified AgentBench traces.

Track B phase 0 (frontier-plan.md): ONLY trajectories whose graders
confirmed success become training data. Each sample replicates the
runtime loop's exact message structure:

    system(build_system_prompt) → user(task prompt) →
    (assistant(raw tool-call JSON) → user("Tool result:\n" + envelope))* →
    assistant(final JSON)

Approximations (documented, acceptable for v0):
- Tool-step assistant raw is reconstructed as {"tool": ..., "args": ...}
  (the harness stores parsed tool/args; the grammar-constrained model
  emits exactly this shape).
- Envelopes are rebuilt from stored fields with stderr="" — stored
  output/hint are what the model saw (bounded at 600 chars by the
  harness).
- Runs containing protocol-violation steps are EXCLUDED entirely: a
  protocol step means the assistant message at that turn was a known
  bad example, and clean data beats volume.

Usage:
    uv run python scripts/data/build_sft_dataset.py \
        [--results-dir bench/results] [--out data/sft/flashmini_traces.jsonl]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from flashmini.agent.loop import envelope_message  # noqa: E402
from flashmini.agent.schemas import build_system_prompt  # noqa: E402
from flashmini.tools import ToolContext, ToolResult, ToolStatus, build_default_registry  # noqa: E402

TRUNCATED_FINAL_LEN = 300  # harness stores raw[:300] — equality means cut off


def _load_agentbench() -> ModuleType:
    path = REPO / "scripts" / "bench" / "agentbench.py"
    spec = importlib.util.spec_from_file_location("_agentbench_for_sft", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_task_prompts() -> dict[str, str]:
    """Task name → prompt text, from the live bench definitions."""
    agentbench = _load_agentbench()
    return {t.name: t.prompt for t in agentbench.TASKS}


def load_system_prompt() -> str:
    """The exact system prompt the runtime loop sends."""
    registry = build_default_registry(ToolContext(workspace=REPO))
    return build_system_prompt(registry.names())


def _rebuild_envelope(status: str, tool: str, output: str, hint: str | None) -> str:
    result = ToolResult(
        tool=tool,
        status=ToolStatus(status),
        output=output,
        next_action_hint=hint,
    )
    return "Tool result:\n" + envelope_message(result)


def _assistant_raw(step: dict[str, object]) -> str:
    raw = step.get("raw")
    if isinstance(raw, str):
        return raw
    return json.dumps({"tool": step.get("tool"), "args": step.get("args")}, ensure_ascii=False)


def build_sample(
    task: str, steps: list[dict[str, object]], task_prompts: dict[str, str], system: str
) -> dict[str, Any] | None:
    if task not in task_prompts:
        return None
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_prompts[task]},
    ]
    for step in steps:
        kind = step.get("kind")
        if kind == "tool":
            messages.append({"role": "assistant", "content": _assistant_raw(step)})
            messages.append(
                {
                    "role": "user",
                    "content": _rebuild_envelope(
                        str(step.get("status", "error")),
                        str(step.get("tool", "")),
                        str(step.get("output", "")),
                        step.get("hint"),  # type: ignore[arg-type]
                    ),
                }
            )
        elif kind == "final":
            messages.append({"role": "assistant", "content": _assistant_raw(step)})
        # protocol steps never reach here (filtered by caller)
    return {"messages": messages, "task": task}


def collect_samples(
    results_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    task_prompts = load_task_prompts()
    system = load_system_prompt()
    stats = {
        "passed_runs": 0,
        "excluded_failed": 0,
        "excluded_protocol": 0,
        "excluded_truncated_final": 0,
        "duplicates_dropped": 0,
        "samples": 0,
    }
    samples: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    for path in sorted(results_dir.glob("agentbench-*.json")):
        payload = json.loads(path.read_text())
        for result in payload.get("results", []):
            if not result.get("passed"):
                stats["excluded_failed"] += 1
                continue
            stats["passed_runs"] += 1
            steps = list(result.get("steps", []))
            if any(s.get("kind") == "protocol" for s in steps):
                stats["excluded_protocol"] += 1
                continue
            finals = [s for s in steps if s.get("kind") == "final"]
            if finals and len(str(finals[-1].get("raw", ""))) == TRUNCATED_FINAL_LEN:
                stats["excluded_truncated_final"] += 1
                continue
            task = str(result.get("name", ""))
            sample = build_sample(task, steps, task_prompts, system)
            if sample is None:
                continue
            key = (
                task,
                "\n".join(
                    m["content"]
                    for m in sample["messages"]
                    if m["role"] == "assistant"  # type: ignore[index,union-attr]
                ),
            )
            if key in seen:
                stats["duplicates_dropped"] += 1
                continue
            seen.add(key)
            sample["source"] = path.name
            samples.append(sample)

    stats["samples"] = len(samples)
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO / "bench" / "results")
    parser.add_argument(
        "--out", type=Path, default=REPO / "data" / "sft" / "flashmini_traces.jsonl"
    )
    args = parser.parse_args()

    samples, stats = collect_samples(args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
