"""Tests for the CLI parser and step formatting (no server involved)."""

from __future__ import annotations

from flashmini.agent import Step
from flashmini.cli import DEFAULT_FIX_STEPS, DEFAULT_RUN_STEPS, build_parser, format_step
from flashmini.tools import ToolResult


def test_parser_run_defaults() -> None:
    args = build_parser().parse_args(["run", "do a thing"])
    assert args.command == "run"
    assert args.task == "do a thing"
    assert args.max_steps == DEFAULT_RUN_STEPS


def test_parser_fix_defaults() -> None:
    args = build_parser().parse_args(["fix"])
    assert args.command == "fix"
    assert args.max_steps == DEFAULT_FIX_STEPS


def test_parser_run_overrides() -> None:
    args = build_parser().parse_args(["run", "t", "--max-steps", "3", "--workspace", "/tmp"])
    assert args.max_steps == 3
    assert args.workspace == "/tmp"


def test_format_step_tool_ok() -> None:
    result = ToolResult.ok("read_file", output="x")
    step = Step(index=0, kind="tool", tool="read_file", result=result)
    assert format_step(step) == "[0] read_file -> ok"


def test_format_step_denied_shows_status() -> None:
    result = ToolResult.denied("write_file", output="nope")
    step = Step(index=3, kind="tool", tool="write_file", result=result)
    assert format_step(step) == "[3] write_file -> denied"


def test_format_step_final_and_protocol() -> None:
    assert format_step(Step(index=1, kind="final", raw="{}")) == "[1] final"
    assert format_step(Step(index=2, kind="protocol", raw="x")) == "[2] protocol"
