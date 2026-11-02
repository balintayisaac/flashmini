#!/usr/bin/env python3
"""Depth-first distillation corpus: synthetic implement-spec tasks.

Generates the measured weakness class (implement-spec: the 4B model could
not write canonical one-liners like re.sub(r"[^a-z0-9]+", "-", s).strip("-"))
as concrete workspaces: SPEC.md + pre-written tests, with app/<name>.py left
for the agent (teacher) to create. Grading is pytest — a passing teacher
trajectory is verified training data; the bench results builder absorbs it.

Depth-first plan (user-approved): teacher-solve these via the agent loop on
NIM, keep grader-passing runs, SFT on Kaggle, gate on the 18-task bench.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PYPROJECT = '[tool.pytest.ini_options]\npythonpath = ["."]\n'

# (task_name, description, examples, solution, test_cases)
# spec: s: str -> str; each test case is (input, expected).
TEMPLATES: list[tuple[str, str, str, str, list[tuple[str, str]]]] = [
    (
        "slugify",
        "lowercase the text; replace every run of non-alphanumeric characters "
        "with a single hyphen; strip leading/trailing hyphens; an empty result "
        "becomes 'n-a'.",
        "'Hello, World!' -> 'hello-world'; '  A  B  ' -> 'a-b'; '***' -> 'n-a'",
        "import re\n\n\ndef slugify(s: str) -> str:\n"
        '    out = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")\n'
        '    return out or "n-a"\n',
        [
            ("Hello, World!", "hello-world"),
            ("  A  B  ", "a-b"),
            ("***", "n-a"),
            ("Already-Clean", "already-clean"),
        ],
    ),
    (
        "snake_case",
        "insert an underscore before every uppercase letter that is not the "
        "first character, then lowercase everything.",
        "'CamelCase' -> 'camel_case'; 'simple' -> 'simple'; 'XMLParser' -> 'x_m_l_parser'",
        "import re\n\n\ndef snake_case(s: str) -> str:\n"
        '    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()\n',
        [("CamelCase", "camel_case"), ("simple", "simple"), ("XMLParser", "x_m_l_parser")],
    ),
    (
        "initials",
        "take the first character of every whitespace-separated word and "
        "return them concatenated, uppercase.",
        "'Grace Brewster Hopper' -> 'GBH'; '  multiple   spaces ' -> 'MS'; 'ada' -> 'A'",
        'def initials(s: str) -> str:\n    return "".join(w[0] for w in s.split()).upper()\n',
        [("Grace Brewster Hopper", "GBH"), ("  multiple   spaces ", "MS"), ("ada", "A")],
    ),
    (
        "mask_secret",
        "replace every character except the last four with '*'; inputs of "
        "four or fewer characters are entirely masked.",
        "'1234567890' -> '******7890'; 'abc' -> '***'; '' -> ''",
        "def mask_secret(s: str) -> str:\n"
        '    if len(s) <= 4:\n        return "*" * len(s)\n'
        '    return "*" * (len(s) - 4) + s[-4:]\n',
        [("1234567890", "******7890"), ("abc", "***"), ("", "")],
    ),
    (
        "normalize_ws",
        "collapse every run of whitespace characters (spaces, tabs, newlines) "
        "into a single space and strip the ends.",
        "'a\\n  b\\t c' -> 'a b c'; 'already fine' -> 'already fine'; '   ' -> ''",
        'def normalize_ws(s: str) -> str:\n    return " ".join(s.split())\n',
        [("a\n  b\t c", "a b c"), ("already fine", "already fine"), ("   ", "")],
    ),
    (
        "filename_safe",
        "replace every character that is not a letter, digit, dot, underscore "
        "or hyphen with an underscore; keep everything else unchanged.",
        "'my file:v2?.txt' -> 'my_file_v2_.txt'; 'clean_v2.txt' -> 'clean_v2.txt'; "
        "'a/b\\\\c' -> 'a_b_c'",
        "import re\n\n\ndef filename_safe(s: str) -> str:\n"
        '    return re.sub(r"[^A-Za-z0-9._-]", "_", s)\n',
        [
            ("my file:v2?.txt", "my_file_v2_.txt"),
            ("clean_v2.txt", "clean_v2.txt"),
            ("a/b\\c", "a_b_c"),
        ],
    ),
    (
        "cap_words",
        "capitalize the first letter of every whitespace-separated word and "
        "lowercase the rest of each word; single-space the words.",
        "'hello world' -> 'Hello World'; 'ADA lovelace' -> 'Ada Lovelace'; '' -> ''",
        'def cap_words(s: str) -> str:\n    return " ".join(w.capitalize() for w in s.split())\n',
        [("hello world", "Hello World"), ("ADA lovelace", "Ada Lovelace"), ("", "")],
    ),
    (
        "reverse_words",
        "return the words in reverse order, single-spaced.",
        "'hello world' -> 'world hello'; 'one' -> 'one'; '  trim   me  ' -> 'me trim'",
        'def reverse_words(s: str) -> str:\n    return " ".join(reversed(s.split()))\n',
        [("hello world", "world hello"), ("one", "one"), ("  trim   me  ", "me trim")],
    ),
]


def all_template_names() -> list[str]:
    return [t[0] for t in TEMPLATES]


def make_workspace(ws: Path, name: str) -> None:
    """Materialize one synthetic task workspace (SPEC.md, tests, pyproject)."""
    task = next(t for t in TEMPLATES if t[0] == name)
    _, desc, examples, _solution, cases = task
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "pyproject.toml").write_text(PYPROJECT)
    spec = (
        f"# SPEC: {name}\n\n"
        f"Implement `{name}(s: str) -> str` in `app/{name}.py`:\n\n{desc}\n\n"
        "Examples:\n\n" + "\n".join(f"- {e}" for e in examples.split("; ")) + "\n"
    )
    (ws / "SPEC.md").write_text(spec)
    (ws / "app").mkdir(exist_ok=True)
    lines = [f"from app.{name} import {name}", "", ""]
    for i, (inp, expected) in enumerate(cases):
        lines.append(f"def test_case_{i}():")
        lines.append(f"    assert {name}({inp!r}) == {expected!r}")
        lines.append("")
    (ws / "tests").mkdir(exist_ok=True)
    (ws / "tests" / f"test_{name}.py").write_text("\n".join(lines))


def grade(ws: Path) -> bool:
    """True iff the workspace's pytest run passes (agent's app/<name>.py in place)."""
    # Plain workspaces: no uv-managed deps, so run the interpreter's pytest
    # directly with the inherited environment (a hand-built env broke uv and
    # subprocess lookups; PYTHONDONTWRITEBYTECODE guards stale .pyc reuse).
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    return proc.returncode == 0
