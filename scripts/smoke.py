"""Smoke test: dogfood the tool layer against the FlashMini repo itself.

Exercises every registered tool once (plus a few security checks), prints a
PASS/FAIL table, and exits non-zero if anything fails. Run via `make smoke`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flashmini.tools import (
    ToolContext,
    ToolStatus,
    build_default_registry,
)

REPO = Path(__file__).resolve().parent.parent

BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"


def main() -> int:
    rows: list[tuple[str, bool, str]] = []
    ctx = ToolContext(workspace=REPO)
    reg = build_default_registry(ctx)

    # --- read_file ---------------------------------------------------------
    r = reg.call("read_file", path="README.md")
    read_ok = r.status is ToolStatus.OK and "FlashMini" in r.output
    rows.append(("read_file", read_ok, f"{len(r.output)} chars"))

    # --- grep ---------------------------------------------------------------
    r = reg.call("grep", pattern="class ToolContext", path="src", glob="*.py")
    first = r.output.splitlines()[0] if r.output else "no output"
    rows.append(("grep", r.status is ToolStatus.OK and "context.py" in r.output, first))

    # --- symbol_search ------------------------------------------------------
    r = reg.call("symbol_search", query="ToolRegistry")
    rows.append(
        (
            "symbol_search",
            r.status is ToolStatus.OK and "class ToolRegistry" in r.output,
            r.output.splitlines()[0] if r.output else "no output",
        )
    )

    # --- git_ops ------------------------------------------------------------
    r = reg.call("git_ops", op="status", args=["--short"])
    rows.append(("git_ops status", r.status is ToolStatus.OK, "repo status ok"))

    r = reg.call("git_ops", op="push")
    rows.append(("git_ops push (should deny)", r.status is ToolStatus.DENIED, "denied as designed"))

    # --- workspace jail security check --------------------------------------
    r = reg.call("read_file", path="../../etc/passwd")
    rows.append(
        ("read_file escape (should deny)", r.status is ToolStatus.DENIED, "denied as designed")
    )

    # --- lint / run_tests -----------------------------------------------------
    r = reg.call("lint")
    rows.append(("lint", r.status is ToolStatus.OK, "ruff clean"))

    r = reg.call("run_tests")
    rows.append(("run_tests", r.status is ToolStatus.OK, r.next_action_hint or "pytest"))

    # --- write/patch roundtrip in a throwaway sandbox workspace --------------
    with tempfile.TemporaryDirectory() as td:
        sctx = ToolContext(workspace=Path(td))
        sreg = build_default_registry(sctx)

        w = sreg.call(
            "write_file", path="pkg/hello.txt", content="alpha\nbeta\n", create_parents=True
        )
        patch = (
            f"{BEGIN_PATCH}\n"
            "*** Update File: pkg/hello.txt\n"
            "@@\n"
            " alpha\n"
            "-beta\n"
            "+gamma\n"
            f"{END_PATCH}\n"
        )
        p = sreg.call("apply_patch", patch=patch)
        rd = sreg.call("read_file", path="pkg/hello.txt")
        ok = (
            w.status is ToolStatus.OK
            and p.status is ToolStatus.OK
            and rd.status is ToolStatus.OK
            and "gamma" in rd.output
            and "beta" not in rd.output
        )
        rows.append(("write/patch roundtrip", ok, "sandbox workspace"))

        # atomicity: a failing patch must leave the file untouched
        before = rd.output
        bad_patch = (
            f"{BEGIN_PATCH}\n*** Update File: pkg/hello.txt\n@@\n-not-there\n+x\n{END_PATCH}\n"
        )
        bad = sreg.call("apply_patch", patch=bad_patch)
        after = sreg.call("read_file", path="pkg/hello.txt")
        rows.append(
            (
                "patch atomicity",
                bad.status is ToolStatus.ERROR and after.output == before,
                "file untouched after failed patch",
            )
        )

    # --- report ---------------------------------------------------------------
    width = max(len(name) for name, _, _ in rows)
    print(f"FlashMini tool-layer smoke test — workspace: {REPO}")
    print()
    failures = 0
    for name, ok, detail in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {name:<{width}}  {detail}")
        failures += 0 if ok else 1
    print()
    print(f"{len(rows) - failures}/{len(rows)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
