"""Tests for the llama-server runtime wrapper (no real model required)."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flashmini.runtime import (
    ChatClient,
    LlamaServer,
    ServerError,
    build_server_args,
    find_default_model,
    find_llama_server,
    free_port,
)
from flashmini.runtime import server as server_mod


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gguf")
    return path


class TestBuildServerArgs:
    def test_defaults_include_mtp_speculation(self, tmp_path: Path) -> None:
        args = build_server_args(tmp_path / "llama-server", tmp_path / "m.gguf", port=9931)
        assert args[0].endswith("llama-server")
        assert args[1:3] == ["-m", str(tmp_path / "m.gguf")]
        joined = " ".join(args)
        assert "--host 127.0.0.1" in joined
        assert "--port 9931" in joined
        assert "-t 4" in joined
        assert "-c 4096" in joined
        assert "--spec-type draft-mtp" in joined
        assert "--spec-draft-n-max 2" in joined

    def test_speculation_disabled(self, tmp_path: Path) -> None:
        args = build_server_args(
            tmp_path / "llama-server", tmp_path / "m.gguf", port=1, spec_type=None
        )
        assert "--spec-type" not in args


class TestDiscovery:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _touch(tmp_path / "custom" / "llama-server")
        monkeypatch.setenv("FLASHMINI_LLAMA_SERVER", str(fake))
        assert find_llama_server(tmp_path) == fake

    def test_workspace_vendor_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLASHMINI_LLAMA_SERVER", raising=False)
        fake = _touch(tmp_path / "vendor" / "llama" / "llama-server")
        assert find_llama_server(tmp_path) == fake

    def test_default_model_prefers_pinned_winner(self, tmp_path: Path) -> None:
        preferred = _touch(tmp_path / "models" / server_mod.PREFERRED_MODEL_FILE)
        _touch(tmp_path / "models" / "other-q2.gguf")
        assert find_default_model(tmp_path) == preferred

    def test_default_model_falls_back_to_sorted_gguf(self, tmp_path: Path) -> None:
        _touch(tmp_path / "models" / "bbb.gguf")
        first = _touch(tmp_path / "models" / "aaa.gguf")
        assert find_default_model(tmp_path) == first

    def test_default_model_missing(self, tmp_path: Path) -> None:
        assert find_default_model(tmp_path) is None


def test_free_port_is_bindable() -> None:
    port = free_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


class _JSONHandler(BaseHTTPRequestHandler):
    status = 200
    response: dict | None = None
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        body = json.dumps(type(self).response or {}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def mock_client() -> ChatClient:
    _JSONHandler.requests = []
    _JSONHandler.status = 200
    _JSONHandler.response = {"choices": [{"message": {"content": "hello"}}]}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _JSONHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield ChatClient(f"http://{host}:{port}", timeout_s=5)
    httpd.shutdown()
    httpd.server_close()


class TestChatClient:
    def test_parses_content_and_sends_schema(self, mock_client: ChatClient) -> None:
        schema = {"type": "object"}
        text = mock_client.chat_text([{"role": "user", "content": "hi"}], json_schema=schema)
        assert text == "hello"
        sent = _JSONHandler.requests[-1]
        assert sent["json_schema"] == schema
        assert sent["cache_prompt"] is True
        assert sent["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent["messages"][0]["content"] == "hi"

    def test_enable_thinking_and_template_kwargs(self, mock_client: ChatClient) -> None:
        mock_client.chat(
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
            template_kwargs={"custom": 1},
        )
        sent = _JSONHandler.requests[-1]
        assert sent["chat_template_kwargs"] == {"enable_thinking": True, "custom": 1}

    def test_http_error_raises_server_error(self, mock_client: ChatClient) -> None:
        _JSONHandler.status = 500
        _JSONHandler.response = {"error": "boom"}
        with pytest.raises(ServerError, match="500"):
            mock_client.chat([{"role": "user", "content": "hi"}])

    def test_unexpected_shape_raises(self, mock_client: ChatClient) -> None:
        _JSONHandler.response = {"choices": []}
        with pytest.raises(ServerError, match="shape"):
            mock_client.chat_text([{"role": "user", "content": "hi"}])


class TestServerLifecycle:
    def test_base_url_requires_start(self, tmp_path: Path) -> None:
        with pytest.raises(ServerError, match="not running"):
            _ = LlamaServer(tmp_path).base_url

    def test_resolve_requires_binary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLASHMINI_LLAMA_SERVER", raising=False)
        monkeypatch.setattr(server_mod.shutil, "which", lambda name: None)
        with pytest.raises(ServerError, match="llama-server binary"):
            LlamaServer(tmp_path).resolve()

    def test_resolve_requires_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _touch(tmp_path / "vendor" / "llama" / "llama-server")
        monkeypatch.delenv("FLASHMINI_LLAMA_SERVER", raising=False)
        with pytest.raises(ServerError, match="model"):
            LlamaServer(tmp_path).resolve()

    def test_stop_without_start_is_safe(self, tmp_path: Path) -> None:
        LlamaServer(tmp_path).stop()
