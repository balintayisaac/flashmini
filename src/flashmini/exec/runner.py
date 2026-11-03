"""Shared subprocess runner: timeouts, process-group kill, bounded output.

All tools that shell out (grep, git, tests, lint) go through
``run_command``. Output is captured into temp files so a runaway process
cannot exhaust memory, and a timed-out process group is killed hard.
"""

import contextlib
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    """What a completed (or killed) subprocess produced."""

    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    sandboxed: bool = False


def elapsed_ms(start: float) -> int:
    """Milliseconds since a ``time.monotonic()`` reading, for envelopes."""
    return int((time.monotonic() - start) * 1000)


# Model-triggered subprocesses must not inherit the user's secrets.  The
# default environment is an allowlist; anything starting with these prefixes
# (project-local, non-secret plumbing) passes, except explicitly listed keys.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "VIRTUAL_ENV",
        "PYTHONHASHSEED",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
)
_ENV_ALLOWLIST_PREFIXES = ("UV_", "FLASHMINI_")
_ENV_SECRET_KEYS = ("FLASHMINI_API_KEY", "KAGGLE_API_TOKEN")
_SECRET_PATTERNS = (
    (re.compile(r"nvapi-[A-Za-z0-9_-]{8,}"), "<redacted-nvapi-key>"),
    (re.compile(r"KGAT_[A-Za-z0-9]+"), "<redacted-kaggle-token>"),
)


def _child_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Allowlisted copy of the parent environment (secrets withheld)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if (k in _ENV_ALLOWLIST or k.startswith(_ENV_ALLOWLIST_PREFIXES))
        and k not in _ENV_SECRET_KEYS
    }
    if extra:
        env.update(extra)
    return env


def _scrub_secrets(text: str) -> str:
    """Defense-in-depth: keep known key material out of captured output."""
    for key in _ENV_SECRET_KEYS:
        val = os.environ.get(key)
        if val:
            text = text.replace(val, f"<redacted-{key.lower().replace('_', '-')}>")
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    max_output_bytes: int = 200_000,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> ExecResult:
    """Run ``argv`` in ``cwd`` with a hard timeout and bounded output."""
    start = time.monotonic()
    proc_env = _child_env(env)
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=proc_env,
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
        except OSError as exc:
            return ExecResult(
                argv=argv,
                returncode=None,
                stdout="",
                stderr=f"failed to start {argv[0]!r}: {exc}",
                duration_ms=elapsed_ms(start),
                timed_out=False,
            )
        timed_out = False
        try:
            proc.communicate(
                input=stdin_text.encode() if stdin_text is not None else None,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            proc.communicate()
        out_f.seek(0)
        err_f.seek(0)
        raw_out = out_f.read(max_output_bytes + 1)
        raw_err = err_f.read(max_output_bytes + 1)
    return ExecResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=_scrub_secrets(raw_out[:max_output_bytes].decode("utf-8", errors="replace")),
        stderr=_scrub_secrets(raw_err[:max_output_bytes].decode("utf-8", errors="replace")),
        duration_ms=elapsed_ms(start),
        timed_out=timed_out,
        stdout_truncated=len(raw_out) > max_output_bytes,
        stderr_truncated=len(raw_err) > max_output_bytes,
    )


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill the whole process group so children of shell wrappers die too."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
