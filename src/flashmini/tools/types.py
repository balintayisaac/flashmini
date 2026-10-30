"""Normalized result envelope returned by every FlashMini tool.

Every invocation -- success, failure, timeout, or refusal -- returns a
``ToolResult`` with the same seven fields. Callers (and, later, the model)
never see raw exceptions or partial state: just a uniform shape.

    {tool, status, output, stderr, duration_ms, truncated, next_action_hint}
"""

from enum import StrEnum

from pydantic import BaseModel


class ToolStatus(StrEnum):
    """Terminal state of a single tool invocation."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"


class ToolResult(BaseModel):
    """Uniform result envelope shared by all FlashMini tools."""

    tool: str
    status: ToolStatus
    output: str = ""
    stderr: str = ""
    duration_ms: int = 0
    truncated: bool = False
    next_action_hint: str | None = None

    @classmethod
    def ok(
        cls,
        tool: str,
        output: str = "",
        *,
        stderr: str = "",
        duration_ms: int = 0,
        truncated: bool = False,
        next_action_hint: str | None = None,
    ) -> "ToolResult":
        return cls(
            tool=tool,
            status=ToolStatus.OK,
            output=output,
            stderr=stderr,
            duration_ms=duration_ms,
            truncated=truncated,
            next_action_hint=next_action_hint,
        )

    @classmethod
    def error(
        cls,
        tool: str,
        output: str = "",
        *,
        stderr: str = "",
        duration_ms: int = 0,
        truncated: bool = False,
        next_action_hint: str | None = None,
    ) -> "ToolResult":
        return cls(
            tool=tool,
            status=ToolStatus.ERROR,
            output=output,
            stderr=stderr,
            duration_ms=duration_ms,
            truncated=truncated,
            next_action_hint=next_action_hint,
        )

    @classmethod
    def timeout(
        cls,
        tool: str,
        output: str = "",
        *,
        duration_ms: int = 0,
        next_action_hint: str | None = None,
    ) -> "ToolResult":
        return cls(
            tool=tool,
            status=ToolStatus.TIMEOUT,
            output=output,
            duration_ms=duration_ms,
            next_action_hint=next_action_hint,
        )

    @classmethod
    def denied(
        cls,
        tool: str,
        output: str = "",
        *,
        duration_ms: int = 0,
        next_action_hint: str | None = None,
    ) -> "ToolResult":
        return cls(
            tool=tool,
            status=ToolStatus.DENIED,
            output=output,
            duration_ms=duration_ms,
            next_action_hint=next_action_hint,
        )
