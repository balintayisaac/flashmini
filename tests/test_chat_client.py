"""v13: ChatClient OpenAI-compat remote mode (NIM) + 429 retry.

The remote branch must send an OpenAI-compatible payload (model,
response_format) with an Authorization header and must NOT send
llama-server-only fields (cache_prompt, chat_template_kwargs, grammar).
Local mode must stay byte-identical to before.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flashmini.runtime import server as server_module
from flashmini.runtime.server import ChatClient


class _CaptureHandler(BaseHTTPRequestHandler):
    """Captures (Authorization header, payload) per POST; scripted statuses."""

    statuses: list[int] = []
    requests: list[dict] = []
    response: dict = {"choices": [{"message": {"content": "ok"}}]}
    raw_body: bytes | None = None  # when set, written verbatim (malformed-body tests)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "payload": payload}
        )
        statuses = type(self).statuses
        status = statuses.pop(0) if statuses else 200
        body = (
            type(self).raw_body
            if type(self).raw_body is not None
            else json.dumps(type(self).response).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def capture(monkeypatch: pytest.MonkeyPatch):
    _CaptureHandler.requests = []
    _CaptureHandler.statuses = []
    _CaptureHandler.raw_body = None
    monkeypatch.setattr(server_module, "_RETRY_DELAYS", (0.01, 0.01))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


_SCHEMA = {"type": "object", "properties": {"tool": {"type": "string"}}}


class TestRemoteCompat:
    def test_remote_payload_is_openai_compat(self, capture: str) -> None:
        client = ChatClient(capture, model="test/model", api_key="sk-test")
        client.chat_text([{"role": "user", "content": "hi"}], json_schema=_SCHEMA, max_tokens=700)
        req = _CaptureHandler.requests[0]
        assert req["auth"] == "Bearer sk-test"
        payload = req["payload"]
        assert payload["model"] == "test/model"
        # NIM reasoning models (DS4.1-flash) return EMPTY content under
        # response_format json_object — never send it (v14; probe-verified).
        assert "response_format" not in payload
        assert "cache_prompt" not in payload
        assert "chat_template_kwargs" not in payload
        assert "grammar" not in payload
        assert payload["max_tokens"] == 700

    def test_local_mode_payload_unchanged(self, capture: str) -> None:
        client = ChatClient(capture)
        client.chat_text([{"role": "user", "content": "hi"}], json_schema=_SCHEMA)
        payload = _CaptureHandler.requests[0]["payload"]
        assert "model" not in payload
        assert payload["cache_prompt"] is True
        assert "chat_template_kwargs" in payload
        assert "response_format" not in payload


class TestRateLimitRetry:
    def test_429s_are_retried_then_succeed(self, capture: str) -> None:
        _CaptureHandler.statuses = [429, 429, 200]
        client = ChatClient(capture, model="test/model", api_key="sk")
        out = client.chat_text([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert len(_CaptureHandler.requests) == 3

    def test_500s_are_retried_then_succeed(self, capture: str) -> None:
        # NIM backend flakiness: DS4.1-flash returned HTTP 500 on 16/18 bench
        # tasks. 500 must be retryable like 429 (probe: plain mt=2048 → 500).
        _CaptureHandler.statuses = [500, 500, 200]
        client = ChatClient(capture, model="test/model", api_key="sk")
        out = client.chat_text([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert len(_CaptureHandler.requests) == 3

    def test_persistent_429_raises(self, capture: str) -> None:
        _CaptureHandler.statuses = [429, 429, 429]
        client = ChatClient(capture, model="test/model", api_key="sk")
        with pytest.raises(server_module.ServerError):
            client.chat_text([{"role": "user", "content": "hi"}])
        assert len(_CaptureHandler.requests) == 3


class TestMalformedBody:
    def test_non_json_body_raises_server_error(self, capture: str) -> None:
        """A proxy error page (HTML with status 200) must surface as
        ServerError — the agent loop only contains ServerError, so a bare
        JSONDecodeError here would crash the CLI (hygiene-audit Critical)."""
        _CaptureHandler.raw_body = b"<html>502 Bad Gateway</html>"
        client = ChatClient(capture, model="test/model", api_key="sk")
        with pytest.raises(server_module.ServerError, match="malformed response body"):
            client.chat_text([{"role": "user", "content": "hi"}])

    def test_truncated_json_body_raises_server_error(self, capture: str) -> None:
        _CaptureHandler.raw_body = b'{"choices": [{"mess'
        client = ChatClient(capture, model="test/model", api_key="sk")
        with pytest.raises(server_module.ServerError, match="malformed response body"):
            client.chat_text([{"role": "user", "content": "hi"}])


class TestBaseUrlJoin:
    def test_v1_base_url_is_not_doubled(self, capture) -> None:
        """NIM-style bases already end in /v1; the wire path must carry it once."""
        client = ChatClient(f"{capture}/v1", model="m", api_key="k")
        client.chat_text([{"role": "user", "content": "hi"}])
        assert _CaptureHandler.requests[0]["path"].count("/v1") == 1

    def test_local_base_url_keeps_v1_path(self, capture) -> None:
        client = ChatClient(capture)
        client.chat_text([{"role": "user", "content": "hi"}])
        assert _CaptureHandler.requests[0]["path"] == "/v1/chat/completions"
