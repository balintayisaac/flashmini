"""Live end-to-end smoke: the real model drives the agent loop against this repo.

Starts the winner configuration (Qwen3.5-4B MTP, --spec-type draft-mtp,
--spec-draft-n-max 2) under llama-server, then asks the agent to complete a
small read-and-report task using the tool layer.

Run: uv run python scripts/smoke_agent.py   (make smoke-agent)
"""

from __future__ import annotations

from pathlib import Path

from flashmini.agent import AgentLoop
from flashmini.runtime import ChatClient, LlamaServer
from flashmini.tools import ToolContext, build_default_registry

REPO = Path(__file__).resolve().parents[1]
TASK = (
    "Read the file docs/STATUS.md with the read_file tool, then report in one "
    "sentence which phase FlashMini is currently in."
)


def main() -> int:
    print(f"repo: {REPO}")
    ctx = ToolContext(workspace=REPO)
    registry = build_default_registry(ctx)

    with LlamaServer(workspace=REPO) as server:
        print(f"server ready at {server.base_url}")
        client = ChatClient(base_url=server.base_url)
        run = AgentLoop(registry, client, max_steps=4).run(TASK)

    print()
    for step in run.steps:
        if step.kind == "tool":
            status = step.result.status.value if step.result else "?"
            print(f"[{step.index}] tool {step.tool} -> {status}")
        else:
            print(f"[{step.index}] {step.kind}: {step.raw[:120]!r}")

    print()
    print(f"finished={run.finished} tool_calls={run.tool_calls}")
    print(f"final: {run.final}")

    ok = run.finished and run.tool_calls >= 1 and bool(run.final)
    print("SMOKE-AGENT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
