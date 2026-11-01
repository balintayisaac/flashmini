"""AgentBench v0: live, deterministic, scoped tasks for the Flash loop.

Runs a small suite of verifiable tasks through the real agent loop against
the real model (or any OpenAI-compatible endpoint), grades each task with
deterministic checks, and records scores as JSON so the numbers can be
tracked over time and re-run after every improvement.

Usage:
    uv run python scripts/bench/agentbench.py                  # local winner model
    uv run python scripts/bench/agentbench.py --base-url URL   # any OAI-compatible endpoint
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from flashmini.agent import AgentLoop, AgentRun
from flashmini.runtime import ChatClient, LlamaServer
from flashmini.tools import ToolContext, build_default_registry

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "bench" / "results"
MAX_STEPS = 12
LOCAL_MODEL_NOTE = "Qwen3.5-4B MTP UD-Q4_K_XL; --spec-type draft-mtp --spec-draft-n-max 2"
VENDORED_RG = REPO / "vendor" / "bin" / "rg"


# ---------------------------------------------------------------------------
# Helpers


def write(ws: Path, rel: str, text: str) -> None:
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def used(run: AgentRun, tool: str, needle: str | None = None) -> bool:
    """True if the run called `tool`; with `needle`, any arg must contain it."""
    for step in run.steps:
        if step.kind != "tool" or step.tool != tool:
            continue
        if needle is None or needle.lower() in json.dumps(step.args).lower():
            return True
    return False


def tools_used(run: AgentRun) -> list[str]:
    return [step.tool for step in run.steps if step.kind == "tool"]


def steps_summary(run: AgentRun) -> list[dict[str, object]]:
    """Compact per-step record for the results JSON (keeps runs self-diagnosing)."""
    out: list[dict[str, object]] = []
    for step in run.steps:
        entry: dict[str, object] = {"index": step.index, "kind": step.kind}
        if step.kind == "tool":
            entry["tool"] = step.tool
            entry["args"] = step.args
            if step.result is not None:
                entry["status"] = step.result.status.value
                entry["output"] = step.result.output[:600]
                entry["hint"] = step.result.next_action_hint
        else:
            entry["raw"] = step.raw[:300]
        out.append(entry)
    return out


def pytest_ok(ws: Path) -> tuple[bool, str]:
    # PYTHONDONTWRITEBYTECODE: a same-second, same-size .py edit would
    # otherwise be masked by a stale __pycache__ entry (mtime+size check).
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return proc.returncode == 0, f"pytest rc={proc.returncode} ({tail})"


# ---------------------------------------------------------------------------
# Task definitions


@dataclass
class Task:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    grade: Callable[[Path, AgentRun], tuple[bool, str]]


def _qa_setup(ws: Path) -> None:
    write(ws, "NOTES.md", "# Notes\n\nThe deploy code for this project is ORANGE-42.\n")


def _qa_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return "ORANGE-42" in (run.final or ""), f"final={run.final!r}"


def _sym_setup(ws: Path) -> None:
    write(ws, "lib/ops.py", "def compute_cost(items):\n    return sum(items) * 2\n")
    write(ws, "lib/other.py", "def reset_all():\n    return None\n")


def _sym_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok = "ops.py" in (run.final or "") and used(run, "symbol_search")
    return ok, f"used_symbol_search={used(run, 'symbol_search')} final={run.final!r}"


def _grep_setup(ws: Path) -> None:
    write(ws, "a.py", "value = 1\n")
    write(ws, "b.py", "token = 'ZEBRA-19'\n")
    write(ws, "c.md", "# doc\n")
    write(ws, "d.txt", "nothing here\n")


def _grep_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok = "b.py" in (run.final or "") and used(run, "grep", "ZEBRA-19")
    return ok, f"used_grep={used(run, 'grep', 'ZEBRA-19')} final={run.final!r}"


def _fix_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["."]\n')
    write(ws, "app/__init__.py", "")
    write(ws, "app/mathx.py", "def add(a, b):\n    return a - b\n")
    test_src = "from app.mathx import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    write(ws, "tests/test_mathx.py", test_src)


def _fix_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    flags = [t for t in ("run_tests", "apply_patch", "write_file") if used(run, t)]
    return ok, f"{detail}; agent tools={flags}"


def _patch_setup(ws: Path) -> None:
    write(ws, "config.py", "MODE = 'dev'\nLEVEL = 1\nDEBUG = True\n")


def _patch_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    content = (ws / "config.py").read_text()
    ok = (
        "MODE = 'prod'" in content
        and "'dev'" not in content
        and "LEVEL = 1" in content
        and "DEBUG = True" in content
        and used(run, "apply_patch")
    )
    return ok, f"file={content!r}"


def _proto_read_setup(ws: Path) -> None:
    write(ws, "README.md", "# Demo\nHello.\n")


def _proto_read_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "read_file", "README.md"), f"tools={tools_used(run)}"


def _proto_grep_setup(ws: Path) -> None:
    write(ws, "data/notes.txt", "bananas are yellow\n")


def _proto_grep_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "grep", "banana"), f"tools={tools_used(run)}"


def _proto_test_setup(ws: Path) -> None:
    write(ws, "tests/test_ok.py", "def test_ok():\n    assert True\n")


def _proto_test_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    return used(run, "run_tests"), f"tools={tools_used(run)}"


def _bughunt_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["."]\n')
    write(ws, "app/__init__.py", "")
    buggy = (
        "def biggest(values):\n"
        "    result = values[0]\n"
        "    for v in values:\n"
        "        if v < result:\n"
        "            result = v\n"
        "    return result\n"
    )
    write(ws, "app/scan.py", buggy)
    write(
        ws,
        "tests/test_scan.py",
        "from app.scan import biggest\n\n\n"
        "def test_biggest():\n"
        "    assert biggest([3, 9, 4]) == 9\n",
    )


def _bughunt_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    flags = [t for t in ("run_tests", "read_file", "apply_patch") if used(run, t)]
    return ok, f"{detail}; agent tools={flags}"


def _multifile_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["."]\n')
    write(ws, "lib/__init__.py", "")
    write(ws, "lib/pricing.py", "def calc(base, rate):\n    return base * rate\n")
    write(ws, "app/__init__.py", "")
    write(
        ws,
        "app/main.py",
        "from lib.pricing import calc\n\n\ndef total(x):\n    return calc(x, 2)\n",
    )
    write(
        ws,
        "tests/test_pricing.py",
        "from lib.pricing import compute\n\n\ndef test_compute():\n    assert compute(3, 2) == 6\n",
    )
    write(
        ws,
        "tests/test_main.py",
        "from app.main import total\n\n\ndef test_total():\n    assert total(5) == 10\n",
    )


def _multifile_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    pricing = (ws / "lib/pricing.py").read_text()
    main = (ws / "app/main.py").read_text()
    ok, detail = pytest_ok(ws)
    clean = "calc" not in pricing and "calc" not in main and "def compute" in pricing
    return ok and clean, f"{detail}; rename_clean={clean}"


def _adapt_setup(ws: Path) -> None:
    write(ws, "settings.ini", "[app]\nname = demo\ntimeout_s = 10\n")


def _adapt_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    content = (ws / "settings.ini").read_text()
    expected = "[app]\nname = demo\ntimeout_s = 10\n[cache]\nsize_mb = 64"
    ok = (
        "[cache]" in content
        and "size_mb = 64" in content
        and "[app]" in content
        and "timeout_s = 10" in content
        and content.rstrip("\n") == expected
    )
    return ok, f"file={content!r}"


def _writetest_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", '[tool.pytest.ini_options]\npythonpath = ["."]\n')
    write(ws, "app/__init__.py", "")
    write(ws, "app/stats.py", "def mean(values):\n    return sum(values) / len(values)\n")


def _writetest_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    test_file = ws / "tests/test_mean.py"
    if not test_file.exists():
        return False, "tests/test_mean.py was not created"
    text = test_file.read_text()
    defs = text.count("def test_")
    ok, detail = pytest_ok(ws)
    wrote_tests = defs >= 2 and "mean" in text
    need = used(run, "run_tests")
    return ok and wrote_tests and need, f"{detail}; test_defs={defs}; used_run_tests={need}"


def _typo_setup(ws: Path) -> None:
    write(ws, "a.py", "FROBNICATOR = True\n")
    write(ws, "b.py", "from a import FROBNICATOR\n")
    write(ws, "c.py", "value = FROBNICATER\n")


def _typo_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    files = {p.name: p.read_text() for p in ws.rglob("*.py") if ".flashmini" not in p.parts}
    bad = any("FROBNICATER" in text for text in files.values())
    fixed = "FROBNICATOR" in files.get("c.py", "")
    greps = sum(1 for t in tools_used(run) if t == "grep")
    return fixed and not bad and greps >= 1, f"greps={greps}; fixed={fixed}; bad_left={bad}"


# ---------------------------------------------------------------------------
# v9 hard tasks — capability axes: cross-file refactor with signature change,
# cross-file debugging (cause away from symptom), spec implementation,
# config repair, and a hand-built data transform.


_PYPROJECT = '[tool.pytest.ini_options]\npythonpath = ["."]\n'

_REFACTOR_CALC = "def conv(temp_c, offset=32):\n    return temp_c * 1.8 + offset\n"
_REFACTOR_REPORT = (
    "from app.calc import conv\n\ndef summary(temps):\n    return [conv(t) for t in temps]\n"
)
_REFACTOR_CLI = "from app.calc import conv\n\ndef show(t):\n    print(conv(t))\n"
_REFACTOR_TESTS = (
    "from app.calc import celsius_to_fahrenheit\n\n"
    "def test_freezing():\n    assert celsius_to_fahrenheit(0) == 32.0\n\n"
    "def test_boiling():\n    assert celsius_to_fahrenheit(100) == 212.0\n"
)


def _refactor_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", _PYPROJECT)
    write(ws, "app/__init__.py", "")
    write(ws, "tools/__init__.py", "")
    write(ws, "app/calc.py", _REFACTOR_CALC)
    write(ws, "app/report.py", _REFACTOR_REPORT)
    write(ws, "tools/cli.py", _REFACTOR_CLI)
    write(ws, "tests/test_calc.py", _REFACTOR_TESTS)


def _refactor_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    if not ok:
        return False, detail
    if "def celsius_to_fahrenheit" not in (ws / "app" / "calc.py").read_text():
        return False, "celsius_to_fahrenheit not defined in app/calc.py"
    for rel in ("app/calc.py", "app/report.py", "tools/cli.py"):
        if re.search(r"\bconv\b", (ws / rel).read_text()):
            return False, f"old name conv still present in {rel}"
    return True, "renamed with baked-in offset; callers updated; pytest green"


_TAX_BROKEN = (
    "TAX_RATE = 0.8\n\ndef with_tax(amount):\n    return round(amount * (1 + TAX_RATE), 2)\n"
)
_CART_PY = "from lib.tax import with_tax\n\ndef total(prices):\n    return with_tax(sum(prices))\n"
_CART_TESTS = (
    "from app.cart import total\n\n\ndef test_total():\n    assert total([10.0, 2.5]) == 15.0\n"
)


def _debugcross_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", _PYPROJECT)
    write(ws, "app/__init__.py", "")
    write(ws, "lib/__init__.py", "")
    write(ws, "lib/tax.py", _TAX_BROKEN)
    write(ws, "app/cart.py", _CART_PY)
    write(ws, "tests/test_cart.py", _CART_TESTS)


def _debugcross_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    if not ok:
        return False, detail
    if (ws / "app" / "cart.py").read_text() != _CART_PY:
        return False, "app/cart.py was modified (task forbids it)"
    if (ws / "tests" / "test_cart.py").read_text() != _CART_TESTS:
        return False, "tests were modified (task forbids it)"
    if "TAX_RATE = 0.2" not in (ws / "lib" / "tax.py").read_text():
        return False, "cause not fixed: TAX_RATE is still not 0.2 in lib/tax.py"
    return True, "cause fixed in lib/tax.py; symptom file and tests untouched"


_SLUG_SPEC = (
    "# slugify spec\n"
    "\n"
    "slugify(text) converts a string to a slug:\n"
    "- lowercase everything\n"
    "- replace every run of non-alphanumeric characters with a single hyphen\n"
    "- strip leading and trailing hyphens\n"
    '- if the result is empty, return "n-a"\n'
    "\n"
    "Examples:\n"
    '- "Hello, World!" -> "hello-world"\n'
    '- "  A  B  " -> "a-b"\n'
    '- "***" -> "n-a"\n'
)
_SLUG_TESTS = (
    "from app.slug import slugify\n\n"
    "def test_basic():\n    assert slugify('Hello, World!') == 'hello-world'\n\n"
    "def test_spaces():\n    assert slugify('  A  B  ') == 'a-b'\n\n"
    "def test_symbols():\n    assert slugify('***') == 'n-a'\n\n"
    "def test_empty():\n    assert slugify('') == 'n-a'\n"
)


def _spec_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", _PYPROJECT)
    write(ws, "app/__init__.py", "")
    write(ws, "SPEC.md", _SLUG_SPEC)
    write(ws, "tests/test_slugify.py", _SLUG_TESTS)


def _spec_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    if not ok:
        return False, detail
    if not (ws / "app" / "slug.py").exists():
        return False, "app/slug.py missing"
    return True, "spec implemented; pre-written tests green"


_SETTINGS_PY = (
    "import configparser\n\n\n"
    "config = configparser.ConfigParser()\n"
    'config.read("config.ini")\n\n'
    'HOST = config.get("server", "host")\n'
    'PORT = config.getint("server", "port")\n'
    'DEBUG = config.getboolean("server", "debug")\n'
)
_SETTINGS_INI_BROKEN = "[SERVER]\nhost = localhost\nport = eight-thousand\ndebug = maybe\n"
_SETTINGS_TESTS = (
    "from app.settings import DEBUG, HOST, PORT\n\n"
    "def test_values():\n    assert (HOST, PORT, DEBUG) == ('localhost', 8080, False)\n"
)


def _config_setup(ws: Path) -> None:
    write(ws, "pyproject.toml", _PYPROJECT)
    write(ws, "app/__init__.py", "")
    write(ws, "app/settings.py", _SETTINGS_PY)
    write(ws, "config.ini", _SETTINGS_INI_BROKEN)
    write(ws, "tests/test_settings.py", _SETTINGS_TESTS)


def _config_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    ok, detail = pytest_ok(ws)
    if not ok:
        return False, detail
    if (ws / "app" / "settings.py").read_text() != _SETTINGS_PY:
        return False, "app/settings.py was modified (task forbids it)"
    if (ws / "tests" / "test_settings.py").read_text() != _SETTINGS_TESTS:
        return False, "tests were modified (task forbids it)"
    return True, "config.ini repaired; code and tests untouched"


_USERS_CSV = "name,role,active\nZed,ops,yes\nAda,admin,yes\nBob,dev,no\nMia,dev,yes\n"
_PIPELINE_EXPECTED: list[dict[str, object]] = [
    {"name": "Ada", "role": "admin", "active": True},
    {"name": "Mia", "role": "dev", "active": True},
    {"name": "Zed", "role": "ops", "active": True},
]


def _pipeline_setup(ws: Path) -> None:
    write(ws, "data/users.csv", _USERS_CSV)


def _pipeline_grade(ws: Path, run: AgentRun) -> tuple[bool, str]:
    path = ws / "data" / "users.json"
    if not path.exists():
        return False, "data/users.json not created"
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"
    if rows != _PIPELINE_EXPECTED:
        return False, "JSON mismatch (need active-only, sorted by name, boolean active)"
    if (ws / "data" / "users.csv").read_text() != _USERS_CSV:
        return False, "users.csv was modified (task forbids it)"
    return True, "correct JSON produced by hand; CSV untouched"


TASKS: list[Task] = [
    Task(
        "qa-read",
        "Read NOTES.md in this workspace, then tell me the deploy code it mentions.",
        _qa_setup,
        _qa_grade,
    ),
    Task(
        "find-symbol",
        "Find which file defines the function compute_cost. Use symbol_search, "
        "then report just the file path in your final answer.",
        _sym_setup,
        _sym_grade,
    ),
    Task(
        "find-text",
        "Use grep to find which file contains the string ZEBRA-19, "
        "then report that file's name in your final answer.",
        _grep_setup,
        _grep_grade,
    ),
    Task(
        "fix-test",
        "The project tests are failing: run them, find the bug in app/mathx.py, "
        "fix it so the tests pass. Do not change the tests. Then say what you fixed.",
        _fix_setup,
        _fix_grade,
    ),
    Task(
        "patch-update",
        "Use apply_patch to change MODE from 'dev' to 'prod' in config.py. Change nothing else.",
        _patch_setup,
        _patch_grade,
    ),
    Task(
        "proto-read",
        "Read the file README.md in this workspace.",
        _proto_read_setup,
        _proto_read_grade,
    ),
    Task(
        "proto-grep",
        "Search the workspace for the word banana using grep.",
        _proto_grep_setup,
        _proto_grep_grade,
    ),
    Task(
        "proto-test",
        "Run the workspace test suite with the run_tests tool.",
        _proto_test_setup,
        _proto_test_grade,
    ),
    Task(
        "bug-hunt",
        "Tests in this workspace are failing, but nothing tells you where the bug is. "
        "Run the tests, locate the wrong function in app/, fix it, and verify. "
        "Do not change the tests.",
        _bughunt_setup,
        _bughunt_grade,
    ),
    Task(
        "multi-file",
        "The tests import a function named compute, but the code defines and calls it as "
        "calc. Rename calc to compute everywhere needed so all tests pass, and change "
        "nothing else. When searching, grep for the bare identifier (calc) rather than "
        "full import statements.",
        _multifile_setup,
        _multifile_grade,
    ),
    Task(
        "adapt-recovery",
        "Add a new section [cache] to settings.ini containing size_mb = 64. "
        "Keep every existing line unchanged, then read the file back to verify.",
        _adapt_setup,
        _adapt_grade,
    ),
    Task(
        "write-test",
        "Write a new file tests/test_mean.py with two test cases for app.stats.mean: "
        "a normal case and a single-value edge case. Run the tests and make sure "
        "everything passes.",
        _writetest_setup,
        _writetest_grade,
    ),
    Task(
        "typo-hunt",
        "One file contains the misspelled identifier FROBNICATER. Find it with grep, "
        "check where the name is actually defined to learn the correct spelling "
        "(grep a shorter stem of the name, case-insensitive, to catch every variant), "
        "fix only that file, then verify with grep that the misspelling is gone.",
        _typo_setup,
        _typo_grade,
    ),
    Task(
        "refactor-sig",
        "The function conv in app/calc.py takes an offset argument. Rename it to "
        "celsius_to_fahrenheit, remove the offset parameter, and bake the value 32 "
        "into the function body. Callers of conv exist in other files: find them "
        "with grep and update every one so all tests pass. Do not change the tests.",
        _refactor_setup,
        _refactor_grade,
    ),
    Task(
        "debug-cross",
        "The tests are failing. The bug is NOT in app/cart.py — find the real cause "
        "elsewhere and fix it. Do not change app/cart.py or the tests.",
        _debugcross_setup,
        _debugcross_grade,
    ),
    Task(
        "implement-spec",
        "Read SPEC.md, then create the file app/slug.py with a slugify function "
        "exactly as the spec describes. Run the tests and make sure they pass. "
        "Do not change the tests or the spec.",
        _spec_setup,
        _spec_grade,
    ),
    Task(
        "config-recovery",
        "The app fails to read config.ini. Fix the configuration file so the tests "
        "pass. Do not change app/settings.py or the tests.",
        _config_setup,
        _config_grade,
    ),
    Task(
        "pipeline",
        "Convert data/users.csv into data/users.json by hand (there are no tests to "
        "run): keep only rows where active is yes, sort them by name, and represent "
        "active as a JSON boolean. Keep the fields name, role, active. Do not modify "
        "the CSV.",
        _pipeline_setup,
        _pipeline_grade,
    ),
]


# ---------------------------------------------------------------------------
# Runner


@dataclass
class TaskResult:
    name: str
    passed: bool
    detail: str
    tool_calls: int
    tools_used: list[str]
    wall_s: float
    final: str | None
    steps: list[dict[str, object]] = field(default_factory=list)


def run_suite(
    client: ChatClient, only: list[str] | None = None, verbose: bool = False
) -> list[TaskResult]:
    if VENDORED_RG.exists():
        os.environ.setdefault("FLASHMINI_RG", str(VENDORED_RG))
    tasks = [t for t in TASKS if not only or t.name in only]
    results: list[TaskResult] = []
    for task in tasks:
        with tempfile.TemporaryDirectory(
            prefix=f"agentbench-{task.name}-", dir="/tmp/opencode"
        ) as td:
            ws = Path(td)
            task.setup(ws)
            registry = build_default_registry(ToolContext(workspace=ws))
            loop = AgentLoop(registry, client, max_steps=MAX_STEPS)
            run: AgentRun | None = None
            start = time.monotonic()
            try:
                run = loop.run(task.prompt)
                passed, detail = task.grade(ws, run)
            except Exception as exc:  # noqa: BLE001 — a crashed task is a failed task
                passed, detail = False, f"crashed: {type(exc).__name__}: {exc}"
            wall = time.monotonic() - start
            tools = tools_used(run) if run else []
            results.append(
                TaskResult(
                    name=task.name,
                    passed=passed,
                    detail=detail,
                    tool_calls=len(tools),
                    tools_used=tools,
                    wall_s=round(wall, 1),
                    final=run.final if run else None,
                    steps=steps_summary(run) if run else [],
                )
            )
            mark = "PASS" if passed else "FAIL"
            print(f"{mark} {task.name:12s} {wall:6.1f}s tools={tools} :: {detail}")
            if verbose and run:
                for s in run.steps:
                    if s.kind == "tool" and s.result is not None:
                        args_s = json.dumps(s.args)[:160]
                        print(f"    [{s.index}] {s.tool} args={args_s}")
                        out_s = s.result.output[:220]
                        hint_s = s.result.next_action_hint
                        print(f"        -> {s.result.status.value}: {out_s!r} hint={hint_s!r}")
                    else:
                        print(f"    [{s.index}] {s.kind}: {s.raw[:160]!r}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentBench v0 — live scoped tasks")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible endpoint; default = local llama-server"
    )
    parser.add_argument("--model-note", default="", help="label for the model under test")
    parser.add_argument("--only", default="", help="comma-separated task names to run")
    parser.add_argument("--verbose", action="store_true", help="print every agent step")
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.base_url:
        note = args.model_note or "remote endpoint"
        print(f"AgentBench v0 — endpoint: {args.base_url} ({note})")
        results = run_suite(ChatClient(base_url=args.base_url), only, args.verbose)
        endpoint = args.base_url
    else:
        note = args.model_note or LOCAL_MODEL_NOTE
        print(f"AgentBench v0 — local winner model ({note})")
        with LlamaServer(workspace=REPO) as server:
            results = run_suite(ChatClient(base_url=server.base_url), only, args.verbose)
        endpoint = "local llama-server"

    passed = sum(1 for r in results if r.passed)
    summary = {
        "stamp": stamp,
        "endpoint": endpoint,
        "model_note": note,
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 3),
        "results": [r.__dict__ for r in results],
    }
    out = RESULTS_DIR / f"agentbench-{stamp}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"AgentBench v0: {passed}/{len(results)} passed ({summary['pass_rate']:.0%})")
    print(f"results: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
