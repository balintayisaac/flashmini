"""v9 hard tasks: setup → grade False (unsolved) → known-good solution → grade True."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from flashmini.agent import AgentRun

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "agentbench.py"
_spec = importlib.util.spec_from_file_location("agentbench_v9", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
agentbench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = agentbench
_spec.loader.exec_module(agentbench)

RUN = AgentRun(task="t", final=None, finished=True, steps=[])


def _task(name: str):
    for t in agentbench.TASKS:
        if t.name == name:
            return t
    raise AssertionError(f"missing task {name!r} in TASKS")


def test_refactor_sig(tmp_path: Path) -> None:
    t = _task("refactor-sig")
    t.setup(tmp_path)
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"must start unsolved: {detail}"

    (tmp_path / "app" / "calc.py").write_text(
        "def celsius_to_fahrenheit(temp_c):\n    return temp_c * 1.8 + 32\n"
    )
    (tmp_path / "app" / "report.py").write_text(
        "from app.calc import celsius_to_fahrenheit\n"
        "\n"
        "def summary(temps):\n"
        "    return [celsius_to_fahrenheit(t) for t in temps]\n"
    )
    (tmp_path / "tools" / "cli.py").write_text(
        "from app.calc import celsius_to_fahrenheit\n"
        "\n"
        "def show(t):\n"
        "    print(celsius_to_fahrenheit(t))\n"
    )
    ok, detail = t.grade(tmp_path, RUN)
    assert ok, detail


def test_debug_cross(tmp_path: Path) -> None:
    t = _task("debug-cross")
    t.setup(tmp_path)
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"must start unsolved: {detail}"

    (tmp_path / "lib" / "tax.py").write_text(
        "TAX_RATE = 0.2\n\ndef with_tax(amount):\n    return round(amount * (1 + TAX_RATE), 2)\n"
    )
    ok, detail = t.grade(tmp_path, RUN)
    assert ok, detail

    # Negative control: touching the forbidden symptom file must fail the grade.
    (tmp_path / "app" / "cart.py").write_text("broken = True\n")
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"tampering with cart.py must fail: {detail}"


def test_implement_spec(tmp_path: Path) -> None:
    t = _task("implement-spec")
    t.setup(tmp_path)
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"must start unsolved: {detail}"

    (tmp_path / "app" / "slug.py").write_text(
        "import re\n"
        "\n"
        "\n"
        "def slugify(text):\n"
        '    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n'
        '    return s or "n-a"\n'
    )
    ok, detail = t.grade(tmp_path, RUN)
    assert ok, detail


def test_config_recovery(tmp_path: Path) -> None:
    t = _task("config-recovery")
    t.setup(tmp_path)
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"must start unsolved: {detail}"

    (tmp_path / "config.ini").write_text("[server]\nhost = localhost\nport = 8080\ndebug = false\n")
    ok, detail = t.grade(tmp_path, RUN)
    assert ok, detail

    # Negative control: patching the code instead of the config must fail.
    (tmp_path / "app" / "settings.py").write_text("# tampered\n")
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"modifying settings.py must fail: {detail}"


def test_pipeline(tmp_path: Path) -> None:
    t = _task("pipeline")
    t.setup(tmp_path)
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"must start unsolved: {detail}"

    expected = [
        {"name": "Ada", "role": "admin", "active": True},
        {"name": "Mia", "role": "dev", "active": True},
        {"name": "Zed", "role": "ops", "active": True},
    ]
    (tmp_path / "data" / "users.json").write_text(json.dumps(expected, indent=2) + "\n")
    ok, detail = t.grade(tmp_path, RUN)
    assert ok, detail

    # Negative control: mutating the source CSV must fail.
    (tmp_path / "data" / "users.csv").write_text("name,role,active\nAda,admin,yes\n")
    ok, detail = t.grade(tmp_path, RUN)
    assert not ok, f"modifying users.csv must fail: {detail}"
