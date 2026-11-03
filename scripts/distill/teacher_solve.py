#!/usr/bin/env python3
"""Teacher-solve harness: run a frontier teacher on the synthetic spec corpus.

Depth-first distillation, phase 2 (see docs/decisions.md). For each spec-corpus
template this script builds the fixture workspace, runs the FlashMini agent
loop against a remote teacher (NIM OpenAI-compatible endpoint), grades the
result with the corpus grader, and writes results in the SAME schema as
AgentBench results JSONs so scripts/data/build_sft_dataset.py absorbs passing
trajectories as verified samples.

Secrets never leave this machine: the teacher key is read from the
FLASHMINI_API_KEY environment variable; nothing is written into the results
file except the endpoint URL and model name.

Usage:
  FLASHMINI_API_KEY=... uv run python scripts/distill/teacher_solve.py \
      [--model deepseek-ai/deepseek-v4.1-flash] [--max-steps 12] [--limit 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from flashmini.agent.loop import AgentLoop, AgentRun  # noqa: E402
from flashmini.runtime.server import ChatClient  # noqa: E402
from flashmini.tools import ToolContext, build_default_registry  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "distill"))
import spec_corpus  # noqa: E402

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
MAX_STEPS = 12
STEP_OUTPUT_CHARS = 600
STEP_RAW_CHARS = 300


def task_prompt(name: str) -> str:
    """implement-spec-shaped task prompt for a spec-corpus template."""
    return (
        f"Read SPEC.md, then create the file app/{name}.py with the function "
        f"exactly as the spec describes. Run the tests and make sure they pass. "
        f"Do not change the tests or the spec."
    )


def serialize_steps(run: AgentRun) -> list[dict[str, Any]]:
    """Serialize an AgentRun into the AgentBench step schema."""
    steps: list[dict[str, Any]] = []
    for step in run.steps:
        if step.kind == "tool" and step.result is not None:
            steps.append(
                {
                    "index": step.index,
                    "kind": "tool",
                    "tool": step.tool,
                    "args": step.args,
                    "status": step.result.status.value,
                    "output": step.result.output[:STEP_OUTPUT_CHARS],
                    "hint": step.result.next_action_hint,
                }
            )
        else:
            steps.append(
                {
                    "index": step.index,
                    "kind": step.kind,
                    "raw": (step.raw or "")[:STEP_RAW_CHARS],
                }
            )
    return steps


def solve_one(name: str, client: ChatClient, max_steps: int, tmp_root: Path) -> dict[str, Any]:
    """Run the teacher-agent on one template; return a bench-schema result row."""
    ws = tmp_root / name
    spec_corpus.make_workspace(ws, name)
    registry = build_default_registry(ToolContext(workspace=ws))
    loop = AgentLoop(registry, client, max_steps=max_steps)
    started = time.monotonic()
    run = loop.run(task_prompt(name))
    wall_s = round(time.monotonic() - started, 1)
    passed = spec_corpus.grade(ws)
    detail = "spec-corpus pytest grade"
    tools = [s.tool for s in run.steps if s.kind == "tool"]
    return {
        "name": f"distill-{name}",
        "passed": passed,
        "detail": detail,
        "tool_calls": len(tools),
        "tools_used": tools,
        "wall_s": wall_s,
        "final": run.final,
        "steps": serialize_steps(run),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--limit", type=int, default=len(spec_corpus.TEMPLATES))
    ap.add_argument("--only", default="", help="comma-separated template names to (re)run")
    args = ap.parse_args()

    api_key = os.environ.get("FLASHMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("FLASHMINI_API_KEY is not set — refusing to run")

    names = spec_corpus.all_template_names()[: args.limit]
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        names = [n for n in names if n in wanted]
    client = ChatClient(base_url=args.base_url, model=args.model, api_key=api_key)
    tmp_root = Path(tempfile.mkdtemp(prefix="spec-corpus-", dir="/tmp/opencode"))

    results = []
    for i, name in enumerate(names, 1):
        print(f"[teacher_solve] ({i}/{len(names)}) {name} ...", flush=True)
        try:
            row = None
            # Two attempts: transient upstream errors (HTTP 502/503, timeouts)
            # are common on hosted endpoints — retry once before recording FAIL.
            for attempt in (1, 2):
                try:
                    row = solve_one(name, client, args.max_steps, tmp_root)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    print(
                        f"[teacher_solve]   attempt {attempt} failed "
                        f"({type(exc).__name__}: {str(exc)[:120]}) — retrying",
                        flush=True,
                    )
                    time.sleep(10)
        except Exception as exc:  # noqa: BLE001 — record and continue the batch
            row = {
                "name": f"distill-{name}",
                "passed": False,
                "detail": f"crashed: {type(exc).__name__}: {str(exc)[:300]}",
                "tool_calls": 0,
                "tools_used": [],
                "wall_s": 0.0,
                "final": None,
                "steps": [],
            }
        assert row is not None  # unreachable: both loop paths assign it
        results.append(row)
        print(
            f"[teacher_solve]   {'PASS' if row['passed'] else 'FAIL'} "
            f"{row['wall_s']}s tools={row['tools_used']}",
            flush=True,
        )

    passed = sum(1 for r in results if r["passed"])
    out = {
        "stamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "endpoint": args.base_url,
        "model": args.model,
        "model_note": f"teacher: {args.model} (distill batch)",
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    dest = REPO / "bench" / "results" / f"distill-{datetime.now():%Y%m%d-%H%M%S}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[teacher_solve] {passed}/{len(results)} — wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
