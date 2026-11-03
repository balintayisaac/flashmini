"""Manage a local llama-server process and talk to it over HTTP.

Defaults come from the Phase 1.3 model decision (docs/decisions.md):
Qwen3.5-4B MTP with ``--spec-type draft-mtp --spec-draft-n-max 1``,
measured at ~4.2 tok/s generation with a ~4.8 GiB peak footprint at
``-c 4096`` on the reference machine (A1 sweep, 2026-11-01).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

DEFAULT_THREADS = 4
DEFAULT_CTX = 4096
DEFAULT_SPEC_TYPE = "draft-mtp"
# A1 sweep (bench/results/mtp-sweep-20261101-173341.json): depth 1 ties
# depth 2 on tok/s (overlapping distributions) at 200 MB less peak RSS in
# all 6 paired measurements — decisive on the swap-pressured 7.6 GiB box.
DEFAULT_SPEC_DRAFT_N_MAX = 1
DEFAULT_STARTUP_TIMEOUT_S = 180.0
DEFAULT_CHAT_TIMEOUT_S = 600.0
# NIM free-tier flakiness: retry HTTP 429 (rate limits) and 500 (backend
# errors — DS4.1-flash crashed on plain 2048-token requests) twice.
_RETRY_DELAYS = (5.0, 15.0)
HEALTH_POLL_INTERVAL_S = 0.5
PREFERRED_MODEL_FILE = "Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf"


class ServerError(Exception):
    """llama-server is missing, failed to become ready, or returned an error."""


def find_llama_server(workspace: Path) -> Path | None:
    """Locate the llama-server binary: env override, then the workspace vendor dir, then PATH."""
    env = os.environ.get("FLASHMINI_LLAMA_SERVER")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_file():
            return candidate
    local = workspace / "vendor" / "llama" / "llama-server"
    if local.is_file():
        return local
    found = shutil.which("llama-server")
    return Path(found) if found else None


def find_default_model(workspace: Path) -> Path | None:
    """Pick a model: the pinned Phase 1.3 winner first, then any GGUF, alphabetically."""
    models = workspace / "models"
    if not models.is_dir():
        return None
    preferred = models / PREFERRED_MODEL_FILE
    if preferred.is_file():
        return preferred
    ggufs = sorted(models.glob("*.gguf"))
    return ggufs[0] if ggufs else None


def free_port() -> int:
    """Return a currently free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_args(
    server_bin: Path,
    model_path: Path,
    *,
    port: int,
    threads: int = DEFAULT_THREADS,
    ctx_size: int = DEFAULT_CTX,
    spec_type: str | None = DEFAULT_SPEC_TYPE,
    spec_draft_n_max: int = DEFAULT_SPEC_DRAFT_N_MAX,
) -> list[str]:
    """Assemble llama-server argv. Flags are verified against build b12001's --help."""
    args = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
    ]
    if spec_type is not None:
        args += ["--spec-type", spec_type, "--spec-draft-n-max", str(spec_draft_n_max)]
    return args


@dataclass
class LlamaServer:
    """Owns one llama-server process and exposes its base URL once ready."""

    workspace: Path
    model_path: Path | None = None
    server_bin: Path | None = None
    threads: int = DEFAULT_THREADS
    ctx_size: int = DEFAULT_CTX
    spec_type: str | None = DEFAULT_SPEC_TYPE
    spec_draft_n_max: int = DEFAULT_SPEC_DRAFT_N_MAX
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _port: int = field(default=0, init=False, repr=False)
    _log_file: IO[bytes] | None = field(default=None, init=False, repr=False)

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        if not self._port:
            raise ServerError("server is not running")
        return f"http://127.0.0.1:{self._port}"

    @property
    def log_path(self) -> Path:
        return self.workspace / ".flashmini" / "logs" / "llama-server.log"

    def resolve(self) -> tuple[Path, Path]:
        """Return the (binary, model) pair this server will use, or raise ServerError."""
        server_bin = self.server_bin or find_llama_server(self.workspace)
        if server_bin is None:
            raise ServerError(
                "llama-server binary not found "
                "(looked at $FLASHMINI_LLAMA_SERVER, vendor/llama/llama-server, PATH)"
            )
        model = self.model_path or find_default_model(self.workspace)
        if model is None:
            raise ServerError("no GGUF model found under models/")
        return server_bin, model

    def start(self) -> None:
        """Spawn the server and block until /health reports ready."""
        if self._proc is not None and self._proc.poll() is None:
            raise ServerError("server is already running")
        server_bin, model = self.resolve()
        if not self._port:
            self._port = free_port()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("ab")
        args = build_server_args(
            server_bin,
            model,
            port=self._port,
            threads=self.threads,
            ctx_size=self.ctx_size,
            spec_type=self.spec_type,
            spec_draft_n_max=self.spec_draft_n_max,
        )
        self._proc = subprocess.Popen(
            args,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self.wait_ready()
        except Exception:
            self.stop()
            raise

    def wait_ready(self) -> None:
        """Poll /health until ready, the process dies, or the deadline passes."""
        deadline = time.monotonic() + self.startup_timeout_s
        url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise ServerError(
                    f"llama-server exited with code {proc.returncode}; "
                    f"last log lines:\n{self._log_tail()}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(HEALTH_POLL_INTERVAL_S)
        raise ServerError(
            f"llama-server did not become ready within {self.startup_timeout_s:.0f}s; "
            f"last log lines:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        """Terminate the server (process group first for safety) and close logs."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _log_tail(self, lines: int = 12) -> str:
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return "(no log)"
        return "\n".join(text.splitlines()[-lines:])

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


@dataclass
class ChatClient:
    """Minimal client for llama-server's OpenAI-compatible chat endpoint."""

    base_url: str
    timeout_s: float = DEFAULT_CHAT_TIMEOUT_S
    api_key: str | None = None
    model: str | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        grammar: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        cache_prompt: bool = True,
        enable_thinking: bool = False,
        template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST a chat request; returns the raw response object.

        Thinking is disabled by default via ``chat_template_kwargs``: Qwen3.5's
        hybrid reasoning otherwise spends the whole budget in
        ``reasoning_content`` and leaves ``content`` empty. Verified against
        llama-server b12001, together with per-request ``json_schema``
        enforcement and ``cache_prompt`` prefix reuse (cache hits show in
        ``usage.prompt_tokens_details.cached_tokens``).
        """
        if self.model is not None:
            # OpenAI-compat remote endpoints (e.g. NVIDIA NIM): send the model
            # field; drop llama-server-only extensions. response_format is NOT
            # sent: NIM reasoning models (DS4.1-flash) return EMPTY content
            # under json_object mode — the system prompt already mandates
            # single-JSON replies, so structured decoding is unnecessary.
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            return self._post("/v1/chat/completions", payload)
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": cache_prompt,
        }
        kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if template_kwargs:
            kwargs.update(template_kwargs)
        payload["chat_template_kwargs"] = kwargs
        if json_schema is not None:
            payload["json_schema"] = json_schema
        if grammar is not None:
            payload["grammar"] = grammar
        return self._post("/v1/chat/completions", payload)

    def chat_text(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """POST a chat request and return just the assistant text."""
        data = self.chat(messages, **kwargs)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServerError(f"unexpected chat response shape: {exc}") from exc
        return str(content or "")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delays = (*_RETRY_DELAYS, None)
        raw = b""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1"):
            # Remote OpenAI-style bases already include /v1 (e.g. NIM) — never
            # join into /v1/v1/... (round: DS4.1 head-to-head 0/18 by 404).
            path = path[3:]
        for delay in delays:
            request = urllib.request.Request(
                f"{base}{path}",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500) and delay is not None:
                    time.sleep(delay)
                    continue
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise ServerError(f"HTTP {exc.code} from llama-server {path}: {detail}") from exc
            except (urllib.error.URLError, OSError) as exc:
                if delay is not None:
                    # Connection resets are the documented NIM/Kaggle egress
                    # flake mode (audit I11): retry like 429/500 instead of
                    # failing the whole trajectory on the first drop.
                    time.sleep(delay)
                    continue
                raise ServerError(
                    f"could not reach llama-server at {self.base_url}: {exc}"
                ) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ServerError(
                f"malformed response body from llama-server {path}: {raw[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise ServerError("unexpected non-object response from llama-server")
        return data
