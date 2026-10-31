"""AgentBench v0: live, deterministic, scoped tasks for the Flash loop.

Runs a small suite of verifiable tasks through the real agent loop against
the real model (or any OpenAI-compatible endpoint), grades each task with
deterministic checks, and records scores as JSON so the numbers can be
tracked over time and re-run after every improvement.

Usage:
    uv run python scripts/bench/agentbench.py                  # local winner model
    uv run python scripts/bench/agentbench.py --base-url URL   # any OAI-compatible endpoint
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from flashmini.agent import AgentLoop, AgentRun
from flashmini.runtime import ChatClient, LlamaServer
from flashmini.tools import ToolContext, build_default_registry

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "bench" / "results"
MAX_STEPS = 8
LOCAL_MODEL_NOTE = "Qwen3.5-4B MTP UD-Q4_K_XL; --spec-type draft-mtp --spec-draft-n-max 2"


# ---------------------------------------------------------------------------
# Helpers


def write(ws: Path, rel: str, text: str) -> None:
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def used(run: AgentRun, tool: str, needle: str | None = None) -> bool:
    """True if the run called `tool`; with `needle`, any arg must contain it."""
    for step in run.steps:
        if step.kind != "tool" or step.tool != tool:
            continue
        if needle is None or needle.lower() in json.dumps(step.args).lower():
            return True
    return False


def tools_used(run: AgentRun) -> list[str]:
    return [step.tool for step in run.steps if step.kind == "tool"]


def steps_summary(run: AgentRun) -> list[dict[str, object]]:
    """Compact per-step record for the results JSON (keeps runs self-diagnosing)."""
    out: list[dict[str, object]] = []
    for step in run.steps:
        entry: dict[str, object] = {"index": step.index, "kind": step.kind}
        if step.kind == "tool":
            entry["tool"] = step.tool
            entry["args"] = step.args
            if step.result is not None:
                entry["status"] = step.result.status.value
                entry["output"] = step.result.output[:600]
                entry["hint"] = step.result.next_action_hint
        else:
            entry["raw"] = step.raw[:300]
        out.append(entry)
    return out


def pytest_ok(ws: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=180,
    )
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return proc.returncode == 0, f"pytest rc={proc.returncode} ({tail})"


# ---------------------------------------------------------------------------
# Task definitions


@dataclass
class Task:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    grade: Callable[[Path, AgentRun], tuple[bool, str]]


def _qa_setup(ws: Path) -> None:
    write(ws, "NOTES.md", "# Notes\n\nThe deploy code for this project is ORANGE-42.\n")


def _qa_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return "ORANGE-42" in (run.final or ""), f"final={run.final!r}"


def _sym_setup(ws: Path) -> None:
    write(ws, "lib/ops.py", "def compute_cost(items):\n    return sum(items) * 2\n")
    write(ws, "lib/other.py", "def reset_all():\n    return None\n")


def _sym_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok = "ops.py" in (run.final or "") and used(run, "symbol_search")
    return ok, f"used_symbol_search={used(run, 'symbol_search')} final={run.final!r}"


def _grep_setup(ws: Path) -> None:
    write(ws, "a.py", "value = 1\n")
    write(ws, "b.py", "token = 'ZEBRA-19'\n")
    write(ws, "c.md", "# doc\n")
    write(ws, "d.txt", "nothing here\n")


def _grep_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok = "b.py" in (run.final or "") and used(run, "grep", "ZEBRA-19")
    return ok, f"used_grep={used(run, 'grep', 'ZEBRA-19')} final={run.final!r}"


def _fix_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["."]\n')
    write(ws, "app/__init__.py", "")
    write(ws, "app/mathx.py", "def add(a, b):\n    return a - b\n")
    test_src = "from app.mathx import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    write(ws, "tests/test_mathx.py", test_src)


def _fix_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    flags = [t for t in ("run_tests", "apply_patch", "write_file") if used(run, t)]
    return ok, f"{detail}; agent tools={flags}"


def _patch_setup(ws: Path) -> None:
    write(ws, "config.py", "MODE = 'dev'\nLEVEL = 1\nDEBUG = True\n")


def _patch_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    content = (ws / "config.py").read_text()
    ok = (
        "MODE = 'prod'" in content
        and "'dev'" not in content
        and "LEVEL = 1" in content
        and "DEBUG = True" in content
        and used(run, "apply_patch")
    )
    return ok, f"file={content!r}"


def _proto_read_setup(ws: Path) -> None:
    write(ws, "README.md", "# Demo\nHello.\n")


def _proto_read_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "read_file", "README.md"), f"tools={tools_used(run)}"


def _proto_grep_setup(ws: Path) -> None:
    write(ws, "data/notes.txt", "bananas are yellow\n")


def _proto_grep_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "grep", "banana"), f"tools={tools_used(run)}"


def _proto_test_setup(ws: Path) -> None:
    write(ws, "tests/test_ok.py", "def test_ok():\n    assert True\n")


def _proto_test_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "run_tests"), f"tools={tools_used(run)}"


TASKS: list[Task] = [
    Task(
        "qa-read",
        "Read NOTES.md in this workspace, then tell me the deploy code it mentions.",
        _qa_setup,
        _qa_grade,
    ),
    Task(
        "find-symbol",
        "Find which file defines the function compute_cost. Use symbol_search, "
        "then report just the file path in your final answer.",
        _sym_setup,
        _sym_grade,
    ),
    Task(
        "find-text",
        "Use grep to find which file contains the string ZEBRA-19, "
        "then report that file's name in your final answer.",
        _grep_setup,
        _grep_grade,
    ),
    Task(
        "fix-test",
        "The project tests are failing: run them, find the bug in app/mathx.py, "
        "fix it so the tests pass. Do not change the tests. Then say what you fixed.",
        _fix_setup,
        _fix_grade,
    ),
    Task(
        "patch-update",
        "Use apply_patch to change MODE from 'dev' to 'prod' in config.py. Change nothing else.",
        _patch_setup,
        _patch_grade,
    ),
    Task(
        "proto-read",
        "Read the file README.md in this workspace.",
        _proto_read_setup,
        _proto_read_grade,
    ),
    Task(
        "proto-grep",
        "Search the workspace for the word banana using grep.",
        _proto_grep_setup,
        _proto_grep_grade,
    ),
    Task(
        "proto-test",
        "Run the workspace test suite with the run_tests tool.",
        _proto_test_setup,
        _proto_test_grade,
    ),
]


# ---------------------------------------------------------------------------
# Runner


@dataclass
class TaskResult:
    name: str
    passed: bool
    detail: str
    tool_calls: int
    tools_used: list[str]
    wall_s: float
    final: str | None
    steps: list[dict[str, object]] = field(default_factory=list)


def run_suite(
    client: ChatClient, only: list[str] | None = None, verbose: bool = False
) -> list[TaskResult]:
    tasks = [t for t in TASKS if not only or t.name in only]
    results: list[TaskResult] = []
    for task in tasks:
        with tempfile.TemporaryDirectory(
            prefix=f"agentbench-{task.name}-", dir="/tmp/opencode"
        ) as td:
            ws = Path(td)
            task.setup(ws)
            registry = build_default_registry(ToolContext(workspace=ws))
            loop = AgentLoop(registry, client, max_steps=MAX_STEPS)
            run: AgentRun | None = None
            start = time.monotonic()
            try:
                run = loop.run(task.prompt)
                passed, detail = task.grade(ws, run)
            except Exception as exc:  # noqa: BLE001 — a crashed task is a failed task
                passed, detail = False, f"crashed: {type(exc).__name__}: {exc}"
            wall = time.monotonic() - start
            tools = tools_used(run) if run else []
            results.append(
                TaskResult(
                    name=task.name,
                    passed=passed,
                    detail=detail,
                    tool_calls=len(tools),
                    tools_used=tools,
                    wall_s=round(wall, 1),
                    final=run.final if run else None,
                    steps=steps_summary(run) if run else [],
                )
            )
            mark = "PASS" if passed else "FAIL"
            print(f"{mark} {task.name:12s} {wall:6.1f}s tools={tools} :: {detail}")
            if verbose and run:
                for s in run.steps:
                    if s.kind == "tool" and s.result is not None:
                        args_s = json.dumps(s.args)[:160]
                        print(f"    [{s.index}] {s.tool} args={args_s}")
                        out_s = s.result.output[:220]
                        hint_s = s.result.next_action_hint
                        print(f"        -> {s.result.status.value}: {out_s!r} hint={hint_s!r}")
                    else:
                        print(f"    [{s.index}] {s.kind}: {s.raw[:160]!r}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentBench v0 — live scoped tasks")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible endpoint; default = local llama-server"
    )
    parser.add_argument("--model-note", default="", help="label for the model under test")
    parser.add_argument("--only", default="", help="comma-separated task names to run")
    parser.add_argument("--verbose", action="store_true", help="print every agent step")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.base_url:
        note = args.model_note or "remote endpoint"
        print(f"AgentBench v0 — endpoint: {args.base_url} ({note})")
        results = run_suite(ChatClient(base_url=args.base_url), only, args.verbose)
        endpoint = args.base_url
    else:
        note = args.model_note or LOCAL_MODEL_NOTE
        print(f"AgentBench v0 — local winner model ({note})")
        with LlamaServer(workspace=REPO) as server:
            results = run_suite(ChatClient(base_url=server.base_url), only, args.verbose)
        endpoint = "local llama-server"

    passed = sum(1 for r in results if r.passed)
    summary = {
        "stamp": stamp,
        "endpoint": endpoint,
        "model_note": note,
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3),
        "results": [r.__dict__ for r in results],
    }
    out = RESULTS_DIR / f"agentbench-{stamp}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"AgentBench v0: {passed}/{len(results)} passed ({summary['pass_rate']:.0%})")
    print(f"results: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
