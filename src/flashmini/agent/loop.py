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
from flashmini.runtime.server import ChatClient, ServerError
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


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

UNSEEN_NAME_WARNING = (
    "WARNING: {names} — these written names have never appeared in any file you "
    "read, any search result, or the task. You may be inventing a name. grep a "
    "shorter stem of each name case-insensitively to find the exact spelling used "
    "in this codebase, then correct the file if needed."
)


def _tokens(text: str) -> set[str]:
    """Lowercased identifier-shaped tokens (length >= 4) for name-evidence checks."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


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
        "output": _trim_output(result.output, OUTPUT_CHARS),
        "stderr": _trim_output(result.stderr, STDERR_CHARS),
        "truncated": result.truncated,
        "next_action_hint": result.next_action_hint,
    }
    return json.dumps(payload, ensure_ascii=False)


def _trim_output(text: str, limit: int) -> str:
    """Head+tail trim with an explicit omission marker (v11).

    A flat cut keeps the head of long outputs and loses the failure summary
    at the tail (pytest summaries end the output), and several full-size
    envelopes accumulate into context overflow (round 7: 4447 tokens
    against a 4096-token context).
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[{omitted} chars omitted]\n{text[-half:]}"


def _compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop older tool-result bodies to reclaim context (roadmap A4).

    Keeps the most recent tool result intact; every earlier one becomes a
    one-line placeholder. Called only after the server rejected the request
    for exceeding the context window, and retried exactly once.
    """
    result_idx = [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and str(m.get("content", "")).startswith("Tool result")
    ]
    compact = [dict(m) for m in messages]
    for i in result_idx[:-1]:
        body = str(compact[i]["content"])
        compact[i] = {
            "role": "user",
            "content": f"Tool result: (earlier tool result omitted: {len(body)} chars)",
        }
    return compact


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
        last_ok_sig: str | None = None
        last_ok_output: str | None = None
        ok_repeat_count = 0
        vocab = _tokens(task)
        # Per-run repeat counters: consecutive-only tracking resets whenever any
        # other call intervenes, which is exactly how real loops hide (v9 probe:
        # 4 no-op writes and 10 identical failed reads, guards never fired).
        fail_counts: dict[str, int] = {}
        ok_counts: dict[str, int] = {}
        noop_writes: dict[str, int] = {}

        for index in range(self.max_steps):
            try:
                raw = self.client.chat_text(
                    messages,
                    json_schema=tool_call_schema(names),
                    max_tokens=self.max_tokens,
                )
            except ServerError as exc:
                # Context overflow: compact older tool results and retry once
                # (v11 / roadmap A4). Any other server failure still propagates.
                if "exceeds the available context size" not in str(exc):
                    raise
                messages = _compact_messages(messages)
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
            ok_output = result.output if result.status is ToolStatus.OK else None
            if ok_output is not None and signature == last_ok_sig and ok_output == last_ok_output:
                ok_repeat_count += 1
            else:
                ok_repeat_count = 0
            last_ok_sig = signature if ok_output is not None else None
            last_ok_output = ok_output
            ok_total = 0
            if result.status is not ToolStatus.OK:
                fail_counts[signature] = fail_counts.get(signature, 0) + 1
            else:
                ok_key = f"{signature}|{result.output}"
                ok_total = ok_counts.get(ok_key, 0) + 1
                ok_counts[ok_key] = ok_total
            noop_path: str | None = None
            if (
                tool == "write_file"
                and result.status is ToolStatus.OK
                and result.output.startswith("no changes:")
            ):
                noop_path = str(args.get("path", "the file"))
                noop_writes[noop_path] = noop_writes.get(noop_path, 0) + 1
            if tool == "apply_patch":
                patch_failures = 0 if result.status is ToolStatus.OK else patch_failures + 1
            message = "Tool result:\n" + envelope_message(result)
            if ok_repeat_count >= 1:
                message = (
                    "WARNING: you already ran this exact call and got the same "
                    "result. Repeating it adds nothing new — act on the result "
                    "you have (make the edit or give the answer), or change "
                    "strategy.\n" + message
                )
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
            if result.status is not ToolStatus.OK and fail_counts.get(signature, 0) >= 2:
                message = (
                    f"WARNING: this exact call has now failed "
                    f"{fail_counts[signature]} times in this task (calls in between "
                    "do not reset the count). Do not repeat it — change the "
                    "arguments or the approach.\n" + message
                )
            if ok_total >= 3:
                message = (
                    f"WARNING: you have now run this exact call "
                    f"{ok_total} times with identical results across this "
                    "task. The result will not change — act on it (make the edit "
                    "or give the answer) or change strategy.\n" + message
                )
            if noop_path is not None and noop_writes[noop_path] >= 2:
                message = (
                    f"WARNING: you have now written identical content to {noop_path} "
                    f"{noop_writes[noop_path]} times — the file already has exactly "
                    "this content, so repeating this write can never help. Treat "
                    f"{noop_path} as done; the remaining work is elsewhere: run the "
                    "tests (run_tests) or grep for the names the task asks you to "
                    "change, and edit those files.\n" + message
                )
            unseen_warning = ""
            if result.status is ToolStatus.OK and tool in ("write_file", "apply_patch"):
                # Modifications of existing code must use names with evidence; fresh
                # files (no backup marker) legitimately introduce new names.
                is_modification = tool == "apply_patch" or "backup:" in result.output
                if is_modification:
                    if tool == "write_file":
                        written = str(args.get("content", ""))
                    else:
                        written = "\n".join(
                            line[1:]
                            for line in str(args.get("patch", "")).splitlines()
                            if line.startswith("+") and not line.startswith("+++")
                        )
                    unknown = sorted(_tokens(written) - vocab)[:5]
                    if unknown:
                        unseen_warning = UNSEEN_NAME_WARNING.format(names=", ".join(unknown))
                vocab |= _tokens(result.output)
            if unseen_warning:
                message = unseen_warning + "\n" + message
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
