# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the standalone search command without external requests or credentials."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from northrl.rollout_entrypoints.jobbench import search


@pytest.fixture(autouse=True)
def search_environment(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-search-key")
    monkeypatch.delenv("SERPER_API_URL", raising=False)


def _response(monkeypatch, data):
    payload = data if isinstance(data, bytes) else json.dumps(data).encode()
    captured: dict = {}

    def open_request(request, *, timeout):
        captured.update(request=request, timeout=timeout)
        return BytesIO(payload)

    monkeypatch.setattr(
        search, "build_opener", lambda *handlers: SimpleNamespace(open=open_request)
    )
    return captured


def test_search_posts_authenticated_json_and_returns_source_links(monkeypatch, capsys):
    captured = _response(
        monkeypatch,
        {
            "organic": [
                {"title": "Primary source", "link": "https://source.example", "snippet": "Facts"},
                {"title": "Another source", "link": "https://other.example"},
            ],
            "debug": "test-search-key",
        },
    )
    assert search.main(["office 参考资料", "--num", "1"]) == 0
    request = captured["request"]
    assert request.full_url == "https://google.serper.dev/search"
    assert request.method == "POST"
    assert request.get_header("X-api-key") == "test-search-key"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {"q": "office 参考资料", "num": 1}
    assert captured["timeout"] == 30
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == {
        "query": "office 参考资料",
        "results": [
            {"title": "Primary source", "url": "https://source.example", "snippet": "Facts"}
        ],
    }
    assert "test-search-key" not in output.out


def test_empty_results_are_valid(monkeypatch):
    _response(monkeypatch, {"organic": []})
    assert search.search("query") == {"query": "query", "results": []}


@pytest.mark.parametrize(
    "url",
    [
        "http://google.serper.dev/search",
        "https://user:secret@example.com/search",
        "https://example.com/search?api_key=secret",
        "https://example.com/search#fragment",
        "https://example.com:bad/search",
        "https://exa\nmple.com/search",
        "file:///tmp/key",
        "",
    ],
)
def test_invalid_endpoint_does_not_send_credentials(monkeypatch, url):
    monkeypatch.setenv("SERPER_API_URL", url)
    monkeypatch.setattr(search, "build_opener", lambda *args: pytest.fail("network was called"))
    with pytest.raises(search.SearchError, match="HTTPS URL"):
        search.search("query")


@pytest.mark.parametrize("query,num", [("", 1), ("  ", 1), ("q", 0), ("q", 11), ("q", True)])
def test_invalid_query_or_count_fails_without_request(monkeypatch, query, num):
    monkeypatch.setattr(search, "build_opener", lambda *args: pytest.fail("network was called"))
    with pytest.raises(search.SearchError):
        search.search(query, num)


def test_missing_key_has_actionable_error(monkeypatch, capsys):
    monkeypatch.delenv("SERPER_API_KEY")
    assert search.main(["query"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "SERPER_API_KEY is not configured" in output.err


@pytest.mark.parametrize(
    "data",
    [b"invalid secret JSON", b"\xff", [], {"message": "secret error"}, {"organic": [None]}],
)
def test_malformed_results_do_not_echo_response(monkeypatch, capsys, data):
    _response(monkeypatch, data)
    assert search.main(["query"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "secret" not in output.err
    assert "invalid" in output.err


def test_response_size_is_bounded(monkeypatch):
    monkeypatch.setattr(search, "_MAX_RESPONSE_BYTES", 20)
    _response(monkeypatch, b" " * 21)
    with pytest.raises(search.SearchError, match="size limit"):
        search.search("query")


def test_transport_error_does_not_echo_credentials(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise URLError("test-search-key")

    monkeypatch.setattr(search, "build_opener", lambda *args: SimpleNamespace(open=fail))
    assert search.main(["query"]) == 1
    output = capsys.readouterr()
    assert "test-search-key" not in output.err
    assert "check network" in output.err


@contextmanager
def _http_server(status):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.path, self.headers.get("X-API-KEY")))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(status)
            if 300 <= status < 400:
                self.send_header("Location", "/credential-target")
            self.end_headers()
            self.wfile.write(b'{"message": "test-search-key"}')

        def do_GET(self):
            calls.append((self.path, self.headers.get("X-API-KEY")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"organic": []}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/search", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "status,expected", [(302, "redirect refused"), (401, "authentication"), (429, "quota")]
)
def test_http_errors_and_redirects_do_not_leak_key(monkeypatch, capsys, status, expected):
    with _http_server(status) as (url, calls):
        # Keep production HTTPS enforcement; only this local transport test bypasses it.
        monkeypatch.setattr(search, "_validate_api_url", lambda value: value)
        monkeypatch.setenv("SERPER_API_URL", url)
        assert search.main(["query"]) == 1
    assert calls == [("/search", "test-search-key")]
    output = capsys.readouterr()
    assert not output.out
    assert expected in output.err
    assert "test-search-key" not in output.err
