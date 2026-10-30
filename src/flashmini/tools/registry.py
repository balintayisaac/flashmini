"""Tool registry: name-based dispatch with exception containment.

Callers (and, later, the model) invoke tools through ``call()``. Every
failure mode is converted into a ``ToolResult`` envelope: workspace
violations become ``denied``, bad arguments and crashes become ``error``.
Raw exceptions never escape the registry.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

from flashmini.exec import elapsed_ms
from flashmini.tools.context import ToolContext, WorkspaceViolation
from flashmini.tools.devops import git_ops, lint, run_tests
from flashmini.tools.files import read_file, write_file
from flashmini.tools.patch import apply_patch
from flashmini.tools.search import grep, symbol_search
from flashmini.tools.types import ToolResult

ToolFn = Callable[..., ToolResult]


class ToolRegistry:
    """A name → tool mapping bound to one workspace context."""

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, /, **kwargs: object) -> ToolResult:
        """Invoke a tool by name; always returns a ToolResult, never raises."""
        started = time.monotonic()
        fn = self._tools.get(name)
        if fn is None:
            return ToolResult.error(
                name,
                f"unknown tool: {name!r}",
                duration_ms=elapsed_ms(started),
                next_action_hint=f"available tools: {', '.join(self.names())}",
            )
        try:
            result = fn(**kwargs)
        except WorkspaceViolation as exc:
            return ToolResult.denied(
                name,
                str(exc),
                duration_ms=elapsed_ms(started),
                next_action_hint="operate inside the workspace root",
            )
        except TypeError as exc:
            return ToolResult.error(
                name,
                f"invalid arguments: {exc}",
                duration_ms=elapsed_ms(started),
                next_action_hint="check the tool's parameters and retry",
            )
        except Exception as exc:  # noqa: BLE001 -- containment is the point
            return ToolResult.error(
                name,
                f"tool crashed: {type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms(started),
                next_action_hint="this is a bug in the tool, not the input",
            )
        if not isinstance(result, ToolResult):
            return ToolResult.error(
                name,
                "tool returned a non-ToolResult value",
                duration_ms=elapsed_ms(started),
                next_action_hint="this is a bug in the tool, not the input",
            )
        return result


def build_default_registry(ctx: ToolContext) -> ToolRegistry:
    """Registry with all Phase 1.2 tools bound to ``ctx``."""
    registry = ToolRegistry(ctx)
    registry.register("read_file", partial(read_file, ctx))
    registry.register("write_file", partial(write_file, ctx))
    registry.register("apply_patch", partial(apply_patch, ctx))
    registry.register("grep", partial(grep, ctx))
    registry.register("symbol_search", partial(symbol_search, ctx))
    registry.register("git_ops", partial(git_ops, ctx))
    registry.register("run_tests", partial(run_tests, ctx))
    registry.register("lint", partial(lint, ctx))
    return registry
