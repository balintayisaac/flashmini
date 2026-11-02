"""Tests for the depth-first distillation corpus generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spec_corpus", Path(__file__).parents[1] / "scripts" / "distill" / "spec_corpus.py"
)
assert _SPEC and _SPEC.loader
spec_corpus = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = spec_corpus
_SPEC.loader.exec_module(spec_corpus)


def test_every_template_is_solvable_and_grade_flips(tmp_path: Path) -> None:
    """Each synthetic task: unsolved -> grade False; reference solution -> True."""
    for name in spec_corpus.all_template_names():
        ws = tmp_path / name
        spec_corpus.make_workspace(ws, name)
        assert not (ws / "app" / f"{name}.py").exists(), f"{name}: solution pre-written"
        assert spec_corpus.grade(ws) is False, f"{name}: graded True without solution"
        template = next(t for t in spec_corpus.TEMPLATES if t[0] == name)
        (ws / "app" / f"{name}.py").write_text(template[3])
        assert spec_corpus.grade(ws) is True, f"{name}: reference solution fails grade"


def test_generation_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    spec_corpus.make_workspace(a, "slugify")
    spec_corpus.make_workspace(b, "slugify")
    for rel in ("SPEC.md", "tests/test_slugify.py", "pyproject.toml"):
        assert (a / rel).read_text() == (b / rel).read_text()
