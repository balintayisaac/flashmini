"""Command line interface — the user-facing entry point for FlashMini.

Run:
    python -m flashmini.cli run "task description" [--max-steps N]
    python -m flashmini.cli fix [--max-steps N]

Both start the local model server, run the agent loop, and print every step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flashmini.agent import AgentLoop, Step
from flashmini.runtime import ChatClient, LlamaServer
from flashmini.tools import ToolContext, ToolStatus, build_default_registry

DEFAULT_RUN_STEPS = 8
DEFAULT_FIX_STEPS = 12

FIX_TASK = (
    "Make this project's test suite pass. Start by running the tests with the "
    "run_tests tool, find the cause of any failure, fix the code, and re-run "
    "the tests until they pass. Verify before you answer: only report success "
    "if the final test run shows the tests passing."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashmini", description="Local-first coding agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one task with the agent loop")
    run.add_argument("task", help="the task, in plain words")
    run.add_argument("--max-steps", type=int, default=DEFAULT_RUN_STEPS)
    run.add_argument("--workspace", default=".")

    fix = sub.add_parser("fix", help="make the project's tests pass")
    fix.add_argument("--max-steps", type=int, default=DEFAULT_FIX_STEPS)
    fix.add_argument("--workspace", default=".")

    return parser


def format_step(step: Step) -> str:
    """One readable line per agent step."""
    if step.kind == "tool" and step.result is not None:
        return f"[{step.index}] {step.tool} -> {step.result.status.value}"
    if step.kind == "final":
        return f"[{step.index}] final"
    return f"[{step.index}] {step.kind}"


def run_loop(task: str, workspace: Path, max_steps: int) -> int:
    ctx = ToolContext(workspace=workspace)
    registry = build_default_registry(ctx)

    with LlamaServer(workspace=workspace) as server:
        client = ChatClient(base_url=server.base_url)
        loop = AgentLoop(registry, client, max_steps=max_steps)

        def on_step(step: Step) -> None:
            print(format_step(step), flush=True)
            if (
                step.kind == "tool"
                and step.result is not None
                and step.result.status is not ToolStatus.OK
            ):
                snippet = (step.result.output or step.result.stderr)[:200].strip()
                if snippet:
                    print(f"    {snippet}", flush=True)

        run = loop.run(task, on_step=on_step)

    if run.finished:
        print()
        print(run.final)
        return 0
    print("\nAgent stopped before finishing (step budget exhausted).", flush=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    task = args.task if args.command == "run" else FIX_TASK
    return run_loop(task, workspace, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
