"""Per-turn output protocol for the agent: exactly one tool call or one final answer.

The schema is enforced server-side by llama.cpp's `json_schema` request field
(verified live against build b12001), so the model physically cannot emit
anything that is not a JSON object in this shape.
"""

from __future__ import annotations

from typing import Any

from flashmini.tools.patch import PATCH_FORMAT_TEMPLATE

TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": "read_file(path, start_line?, end_line?) — read a file with line numbers",
    "write_file": (
        "write_file(path, content, create_parents?) — write or overwrite with RAW file text "
        "(never the line numbers from read_file); use only names seen in file contents — "
        "never guess identifiers; a backup is kept"
    ),
    "apply_patch": (
        "apply_patch(patch) — apply a *** Begin Patch block atomically (all or nothing); "
        "patch lines are raw file text, never read_file display lines"
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
        'To finish: {"final": "<completed answer — only when the work is done>"}',
        "If the work is not done yet, your reply MUST be a tool call —",
        "never a description of what you plan to do.",
        "",
        "Rules:",
        # prompt-master skill: the most-violated rules go first — they must
        # survive attention decay on a 4B model.
        "- If a tool call fails, do not repeat it unchanged: fix the root cause first.",
        "- Before fixing or renaming an identifier, confirm the correct spelling",
        "  where it is defined or imported; use only names seen in file contents.",
        "  Never invent or normalize a spelling: if every occurrence you find has",
        "  the same odd form, that odd form is likely the typo itself — grep a",
        "  shorter stem of the name (case-insensitive) to locate its correct",
        "  definition and copy the exact form from file contents.",
        "- Look before you change: read a file before patching it.",
        "- Never claim something works unless a tool result confirms it.",
        "- If apply_patch fails twice on the same file, switch approach:",
        "  use write_file with the complete corrected file content.",
        '- Never use the "final" field to describe a step you plan to take:',
        '  call the tool and do it. "final" is the completed answer.',
        "- When writing file content or patch lines, write only the literal file text:",
        "  never copy the line numbers or tab separator from read_file output.",
        "- After each tool result, continue with the next tool call or the final answer.",
        "",
        "Tools:",
    ]
    for name in tool_names:
        lines.append(f"- {TOOL_DESCRIPTIONS.get(name, name)}")
    if "apply_patch" in tool_names:
        lines += [
            "",
            "Patch format for apply_patch (exactly this shape — no '--- a/' or",
            "'+++ b/' unified-diff headers):",
            PATCH_FORMAT_TEMPLATE,
            "Copy context lines verbatim from the file you read.",
        ]
    return "\n".join(lines)
