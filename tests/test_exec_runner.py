"""Tests for the shared subprocess runner."""

import sys
from pathlib import Path

from flashmini.exec import run_command


def test_successful_command(tmp_path: Path) -> None:
    result = run_command(["echo", "hello"], cwd=tmp_path, timeout_s=5)
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert result.timed_out is False
    assert result.duration_ms >= 0


def test_nonzero_exit(tmp_path: Path) -> None:
    result = run_command(["bash", "-c", "exit 3"], cwd=tmp_path, timeout_s=5)
    assert result.returncode == 3


def test_timeout_kills_process_group(tmp_path: Path) -> None:
    result = run_command(["bash", "-c", "sleep 5"], cwd=tmp_path, timeout_s=0.5)
    assert result.timed_out is True
    assert result.duration_ms < 3000


def test_output_truncation(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "print('x' * 5000)"],
        cwd=tmp_path,
        timeout_s=10,
        max_output_bytes=1000,
    )
    assert result.stdout_truncated is True
    assert len(result.stdout) == 1000


def test_missing_binary(tmp_path: Path) -> None:
    result = run_command(["definitely-not-a-real-binary-xyz"], cwd=tmp_path, timeout_s=5)
    assert result.returncode is None
    assert "failed to start" in result.stderr


def test_stdin_text(tmp_path: Path) -> None:
    result = run_command(["cat"], cwd=tmp_path, timeout_s=5, stdin_text="piped")
    assert result.stdout == "piped"
