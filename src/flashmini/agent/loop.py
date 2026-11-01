"""The Flash loop: task in, bounded tool-use steps, verified final answer out.

Each turn the model must produce one JSON object (enforced server-side):
either a tool call or a final answer. Tool results come back as normalized
envelopes; the loop never lets a crash escape and never retries forever.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from flashmini.agent.schemas import build_system_prompt, tool_call_schema
from flashmini.runtime.server import ChatClient
from flashmini.tools import ToolRegistry, ToolResult, ToolStatus

MAX_STEPS_DEFAULT = 8
MAX_TOKENS_DEFAULT = 700
OUTPUT_CHARS = 2000
STDERR_CHARS = 500

PROTOCOL_REMINDER = (
    'Reply with exactly ONE JSON object: {"tool": "<name>", "args": {...}} '
    'or {"final": "<answer>"}.'
)

PLANNING_REMINDER = (
    "That is a plan, not a completed answer. Do the first step NOW by calling a tool: "
    '{"tool": "<name>", "args": {...}}. Use "final" only for the finished, verified result.'
)

_PLANNING_RE = re.compile(
    r"\b(i need to|i will|i[’']ll|let me|i am going to|i[’']m going to|"
    r"first,? (i|we)|next,? (i|we)|start by|i should|i plan to)\b",
    re.IGNORECASE,
)


def _looks_like_plan(text: str) -> bool:
    """Heuristic: text that describes intended steps instead of delivering the result."""
    return _PLANNING_RE.search(text[:200]) is not None


@dataclass
class Step:
    """One model turn and what it decided to do."""

    index: int
    kind: str  # "tool" | "final" | "protocol"
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    raw: str = ""


@dataclass
class AgentRun:
    """The full record of a task run."""

    task: str
    final: str | None
    finished: bool
    steps: list[Step]

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool")


def parse_step(text: str) -> dict[str, Any] | None:
    """Parse a model reply into a dict, tolerating stray text around the JSON object."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def envelope_message(result: ToolResult) -> str:
    """Compact, budget-bounded rendering of a tool result for the model."""
    payload = {
        "tool": result.tool,
        "status": result.status.value,
        "output": result.output[:OUTPUT_CHARS],
        "stderr": result.stderr[:STDERR_CHARS],
        "truncated": result.truncated,
        "next_action_hint": result.next_action_hint,
    }
    return json.dumps(payload, ensure_ascii=False)


class AgentLoop:
    """Runs one task as a bounded sequence of constrained tool calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        client: ChatClient,
        *,
        max_steps: int = MAX_STEPS_DEFAULT,
        max_tokens: int = MAX_TOKENS_DEFAULT,
    ) -> None:
        self.registry = registry
        self.client = client
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def run(self, task: str, on_step: Callable[[Step], None] | None = None) -> AgentRun:
        names = self.registry.names()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(names)},
            {"role": "user", "content": task},
        ]
        run = AgentRun(task=task, final=None, finished=False, steps=[])
        last_fail_sig: str | None = None
        repeat_count = 0
        patch_failures = 0

        for index in range(self.max_steps):
            raw = self.client.chat_text(
                messages,
                json_schema=tool_call_schema(names),
                max_tokens=self.max_tokens,
            )
            data = parse_step(raw)

            if data is None:
                step = Step(index=index, kind="protocol", raw=raw)
                run.steps.append(step)
                self._notify(on_step, step)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": PROTOCOL_REMINDER})
                continue

            final = data.get("final")
            if isinstance(final, str) and final.strip():
                if _looks_like_plan(final):
                    step = Step(index=index, kind="protocol", raw=raw)
                    run.steps.append(step)
                    self._notify(on_step, step)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": PLANNING_REMINDER})
                    continue
                step = Step(index=index, kind="final", raw=raw)
                run.steps.append(step)
                self._notify(on_step, step)
                run.final = final
                run.finished = True
                return run

            tool = data.get("tool")
            args = data.get("args") or {}
            if not isinstance(tool, str) or not isinstance(args, dict):
                step = Step(index=index, kind="protocol", raw=raw)
                run.steps.append(step)
                self._notify(on_step, step)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": PROTOCOL_REMINDER})
                continue

            result = self.registry.call(tool, **args)
            step = Step(index=index, kind="tool", tool=tool, args=args, result=result, raw=raw)
            run.steps.append(step)
            self._notify(on_step, step)
            messages.append({"role": "assistant", "content": raw})
            signature = tool + ":" + json.dumps(args, sort_keys=True)
            if result.status is not ToolStatus.OK and signature == last_fail_sig:
                repeat_count += 1
            else:
                repeat_count = 0
            last_fail_sig = signature if result.status is not ToolStatus.OK else None
            if tool == "apply_patch":
                patch_failures = 0 if result.status is ToolStatus.OK else patch_failures + 1
            message = "Tool result:\n" + envelope_message(result)
            if repeat_count >= 1:
                message = (
                    f"WARNING: identical call already failed {repeat_count} time(s). "
                    "Do not repeat it; change the arguments or the approach.\n" + message
                )
            if patch_failures >= 2:
                message = (
                    f"apply_patch has failed {patch_failures} times against different "
                    "arguments; switch approach: use write_file with the complete "
                    "corrected file content.\n" + message
                )
            messages.append({"role": "user", "content": message})

        return run  # step budget exhausted without a final answer

    @staticmethod
    def _notify(on_step: Callable[[Step], None] | None, step: Step) -> None:
        if on_step is not None:
            on_step(step)


def run_task(
    task: str,
    *,
    registry: ToolRegistry,
    client: ChatClient,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> AgentRun:
    """Convenience wrapper: one task, one bounded run."""
    return AgentLoop(registry, client, max_steps=max_steps).run(task)
