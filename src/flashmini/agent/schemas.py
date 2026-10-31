"""Per-turn output protocol for the agent: exactly one tool call or one final answer.

The schema is enforced server-side by llama.cpp's `json_schema` request field
(verified live against build b12001), so the model physically cannot emit
anything that is not a JSON object in this shape.
"""

from __future__ import annotations

from typing import Any

TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": "read_file(path, start_line?, end_line?) — read a file with line numbers",
    "write_file": (
        "write_file(path, content, create_parents?) — write or overwrite a file; a backup is kept"
    ),
    "apply_patch": (
        "apply_patch(patch) — apply a *** Begin Patch block atomically (all or nothing)"
    ),
    "grep": "grep(pattern, path?, glob?, case_sensitive?) — search file contents",
    "symbol_search": "symbol_search(query, path?) — find function and class definitions",
    "git_ops": (
        "git_ops(op, args?) — safe git: status, diff, log, show, add, commit, rev-parse, branch"
    ),
    "run_tests": "run_tests(command?) — run pytest; returns a pass/fail summary",
    "lint": "lint(command?) — run ruff",
}


def tool_call_schema(tool_names: list[str]) -> dict[str, Any]:
    """Constrained output shape: a tool call or a final answer, nothing else."""
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": list(tool_names)},
            "args": {"type": "object"},
            "final": {"type": "string"},
        },
    }


def build_system_prompt(tool_names: list[str]) -> str:
    """System prompt: the protocol, the tool list, and the honesty rules."""
    lines = [
        "You are FlashMini, a precise coding agent working inside one repository.",
        "Respond with exactly ONE JSON object and nothing else.",
        'To call a tool: {"tool": "<name>", "args": {...}}',
        'To finish: {"final": "<answer>"}',
        "",
        "Tools:",
    ]
    for name in tool_names:
        lines.append(f"- {TOOL_DESCRIPTIONS.get(name, name)}")
    lines += [
        "",
        "Rules:",
        "- Look before you change: read a file before patching it.",
        "- Never claim something works unless a tool result confirms it.",
        "- After each tool result, continue with the next tool call or the final answer.",
    ]
    return "\n".join(lines)
