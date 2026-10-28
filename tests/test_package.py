"""Smoke tests: the package installs, imports, and reports its version."""

import flashmini


def test_package_imports() -> None:
    assert flashmini is not None


def test_version_is_semverish() -> None:
    parts = flashmini.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
