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

"""Policy and real socket tests for the standalone task-configured proxy."""

import argparse
import importlib.util
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("proxy", reason="requires the sandbox dependency proxy.py==2.4.10")
from proxy.core.connection.server import TcpServerConnection
from proxy.http.exception import HttpRequestRejected
from proxy.http.parser import HttpParser

MODULE_PATH = Path(__file__).resolve().parents[3] / "northrl/remote_rollout/network_proxy.py"
SPEC = importlib.util.spec_from_file_location("network_proxy", MODULE_PATH)
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)
BLOCKED_DOMAINS = ("github.com", "githubusercontent.com", "gitlab.com", "bitbucket.org")


def plugin():
    return policy.NetworkPolicyPlugin(
        "test",
        argparse.Namespace(blocked_domains=BLOCKED_DOMAINS, network_task_id="task-one"),
        None,
        None,
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("GiTHuB.Com.", "github.com"),
        ("bücher.example", "xn--bcher-kva.example"),
        ("a.GitHub.com", "a.github.com"),
    ],
)
def test_normalizes_domains(value, expected):
    assert policy.normalize_domain(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1",
        "2130706433",
        "0x7f000001",
        "0177.0.0.1",
        "[::1]",
        "::1",
        "github.com/path",
        " github.com",
        "github.com@allowed.example",
    ],
)
def test_rejects_literal_and_invalid_hosts(value):
    with pytest.raises(ValueError):
        policy.normalize_domain(value)


def test_loads_json_domains():
    assert policy.load_blocked_domains('["GiThub.com.", "github.com"]') == ("github.com",)
    assert policy.load_blocked_domains('["gitlab.com"]') == ("gitlab.com",)
    with pytest.raises(ValueError, match="explicitly configured"):
        policy.load_blocked_domains(None)
    with pytest.raises(ValueError):
        policy.load_blocked_domains("[1]")


def test_each_proxy_uses_its_own_domains_and_logs_the_task(monkeypatch, caplog):
    monkeypatch.setattr(policy, "public_address", lambda *_: "8.8.8.8")

    def configured(domain, task_id):
        return policy.NetworkPolicyPlugin(
            task_id,
            argparse.Namespace(blocked_domains=(domain,), network_task_id=task_id),
            None,
            None,
        )

    def request(domain):
        return HttpParser.request(
            f"CONNECT {domain}:443 HTTP/1.1\r\nHost: {domain}:443\r\n\r\n".encode()
        )

    github_policy = configured("github.com", "task-github")
    pypi_policy = configured("pypi.org", "task-pypi")
    with pytest.raises(HttpRequestRejected):
        github_policy.before_upstream_connection(request("api.github.com"))
    with pytest.raises(HttpRequestRejected):
        pypi_policy.before_upstream_connection(request("pypi.org"))
    github_policy.before_upstream_connection(request("pypi.org"))
    pypi_policy.before_upstream_connection(request("github.com"))
    assert "task_id='task-github' domain=api.github.com" in caplog.text
    assert "task_id='task-pypi' domain=pypi.org" in caplog.text


def dns_answers(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in value else socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (value, 443),
        )
        for value in addresses
    ]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "192.0.2.1",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "::ffff:8.8.8.8",
        "64:ff9b::7f00:1",
    ],
)
def test_dns_rejects_any_nonpublic_answer(monkeypatch, address):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: dns_answers("8.8.8.8", address)
    )
    with pytest.raises(HttpRequestRejected):
        policy.public_address("allowed.example", 443)


def test_resolves_once_and_pins_numeric_address(monkeypatch):
    calls = []

    def lookup(*args, **kwargs):
        calls.append(args)
        return dns_answers("8.8.8.8") if len(calls) == 1 else dns_answers("127.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    p = plugin()
    request = HttpParser.request(
        b"CONNECT allowed.example:443 HTTP/1.1\r\nHost: allowed.example:443\r\n\r\n"
    )
    p.before_upstream_connection(request)
    p.handle_client_request(request)
    assert p.resolve_dns("allowed.example", 443) == ("8.8.8.8", None)
    assert p.resolve_dns("allowed.example", 443) == ("8.8.8.8", None)
    assert calls == [("allowed.example", 443)]


@pytest.mark.parametrize(
    "host",
    [
        "github.com",
        "raw.githubusercontent.com",
        "A.GITHUB.COM.",
        "gitlab.com",
        "bitbucket.org",
    ],
)
def test_blocked_destinations_do_not_reach_dns(monkeypatch, host):
    def unexpected(*args, **kwargs):
        pytest.fail("blocked domain reached DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected)
    with pytest.raises(HttpRequestRejected):
        plugin().before_upstream_connection(
            HttpParser.request(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
        )


def test_suffix_matching_does_not_block_similar_names(monkeypatch):
    monkeypatch.setattr(policy, "public_address", lambda *args: "8.8.8.8")
    p = plugin()
    p.before_upstream_connection(
        HttpParser.request(b"GET http://notgithub.com/ HTTP/1.1\r\nHost: notgithub.com\r\n\r\n")
    )
    assert p.resolve_dns("notgithub.com", 80) == ("8.8.8.8", None)


def test_keepalive_cannot_change_destination(monkeypatch):
    monkeypatch.setattr(policy, "public_address", lambda *args: "8.8.8.8")
    p = plugin()
    p.before_upstream_connection(
        HttpParser.request(b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\n\r\n")
    )
    with pytest.raises(HttpRequestRejected):
        p.handle_client_request(
            HttpParser.request(
                b"GET http://another.example/ HTTP/1.1\r\nHost: another.example\r\n\r\n"
            )
        )


def test_rejects_ambiguous_framing(monkeypatch):
    monkeypatch.setattr(policy, "public_address", lambda *args: "8.8.8.8")
    request = HttpParser.request(
        b"POST http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
    )
    with pytest.raises(HttpRequestRejected):
        plugin().before_upstream_connection(request)


def receive_headers(client):
    data = b""
    while b"\r\n\r\n" not in data:
        part = client.recv(65536)
        if not part:
            break
        data += part
    return data


@pytest.fixture
def running_proxy(monkeypatch):
    """Real sockets; only the upstream address is redirected to a local fixture."""
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream.settimeout(0.1)
    stop = threading.Event()
    received = queue.Queue()
    handlers = []

    def handle(client):
        with client:
            client.settimeout(3)
            data = client.recv(65536)
            received.put(data)
            if data.startswith(b"GET "):
                client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                # Keep alive so a second request is checked by the plugin.
                try:
                    data = client.recv(65536)
                    if data:
                        received.put(data)
                except socket.timeout:
                    pass
            else:
                client.sendall(data)

    def serve():
        while not stop.is_set():
            try:
                client, _ = upstream.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            handler = threading.Thread(target=handle, args=(client,), daemon=True)
            handlers.append(handler)
            handler.start()

    original_connect = TcpServerConnection.connect

    def connect(self, addr=None, source_address=None):
        # This assertion proves the real proxy library uses the DNS callback's
        # vetted numeric IP instead of resolving the original hostname again.
        assert addr in (("8.8.8.8", 80), ("8.8.8.8", 443))
        return original_connect(self, addr=upstream.getsockname(), source_address=source_address)

    monkeypatch.setattr(policy, "public_address", lambda *args: "8.8.8.8")
    monkeypatch.setattr(TcpServerConnection, "connect", connect)
    saved_signals = {
        number: signal.getsignal(number)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    }
    server = policy.make_proxy(0, BLOCKED_DOMAINS)
    assert server.flags.enable_conn_pool is False
    serving = threading.Thread(target=serve, daemon=True)
    try:
        with server:
            serving.start()
            yield ("127.0.0.1", server.flags.port), received
    finally:
        stop.set()
        upstream.close()
        if serving.is_alive():
            serving.join(timeout=2)
        for handler in handlers:
            handler.join(timeout=4)
        for number, handler in saved_signals.items():
            signal.signal(number, handler)


@pytest.mark.parametrize(
    "raw_request",
    [
        b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n",
        b"CONNECT 8.8.8.8:443 HTTP/1.1\r\nHost: 8.8.8.8:443\r\n\r\n",
        b"CONNECT allowed.example:22 HTTP/1.1\r\nHost: allowed.example:22\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: github.com\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nHost: github.com\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nX-Test: safe\nHost: github.com\r\n\r\n",
        b"GET http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nContent-Length: -1\r\n\r\n",
        b"POST http://allowed.example/ HTTP/1.1\r\nHost: allowed.example\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    ],
)
def test_socket_denials(running_proxy, raw_request):
    address, received = running_proxy
    with socket.create_connection(address, timeout=3) as client:
        client.sendall(raw_request)
        assert receive_headers(client).startswith(b"HTTP/1.1 403")
    assert received.empty()


def test_duplicate_host_is_rebuilt_as_one_validated_header(running_proxy):
    # proxy.py collapses duplicates to their last value. Check the actual wire
    # request, ensuring that an earlier blocked Host never reaches the upstream.
    address, received = running_proxy
    with socket.create_connection(address, timeout=3) as client:
        client.sendall(
            b"GET http://allowed.example/check HTTP/1.1\r\n"
            b"Host: github.com\r\nHost: allowed.example\r\n\r\n"
        )
        assert receive_headers(client).startswith(b"HTTP/1.1 200")
        forwarded = received.get(timeout=3)
        assert forwarded.lower().count(b"\r\nhost:") == 1
        assert b"Host: allowed.example" in forwarded
        assert b"github.com" not in forwarded


def test_socket_http_forwarding_and_keepalive_check(running_proxy):
    address, received = running_proxy
    with socket.create_connection(address, timeout=3) as client:
        client.sendall(
            b"GET http://allowed.example/check HTTP/1.1\r\nHost: allowed.example\r\n\r\n"
        )
        response = receive_headers(client)
        if not response.endswith(b"OK"):
            response += client.recv(2)
        assert response.endswith(b"OK")
        forwarded = received.get(timeout=3)
        assert forwarded.startswith(b"GET /check HTTP/1.1")
        assert b"Host: allowed.example" in forwarded
        client.sendall(b"GET http://github.com/ HTTP/1.1\r\nHost: github.com\r\n\r\n")
        assert receive_headers(client).startswith(b"HTTP/1.1 403")
    assert received.empty()


def test_socket_connect_is_a_byte_tunnel(running_proxy):
    address, received = running_proxy
    with socket.create_connection(address, timeout=3) as client:
        client.sendall(b"CONNECT allowed.example:443 HTTP/1.1\r\nHost: allowed.example:443\r\n\r\n")
        assert receive_headers(client).startswith(b"HTTP/1.1 200")
        client.sendall(b"opaque TLS bytes")
        assert client.recv(1024) == b"opaque TLS bytes"
        assert received.get(timeout=3) == b"opaque TLS bytes"


@pytest.mark.skipif(os.geteuid() == 0, reason="the service intentionally refuses UID 0")
def test_standalone_isolated_cli_ready_and_sigterm(tmp_path):
    # The root launcher names the copied file proxy.py; -I must prevent this
    # filename from shadowing the installed third-party proxy package.
    copied_script = tmp_path / "proxy.py"
    copied_script.write_bytes(MODULE_PATH.read_bytes())
    service_home = tmp_path / "proxy-home"
    service_home.mkdir(mode=0o700)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(copied_script),
            "--port",
            "0",
            "--blocked-domains",
            '["github.com"]',
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": os.defpath, "HOME": str(service_home), "LANG": "C"},
    )
    try:
        readable, _, _ = select.select([process.stdout], [], [], 10)
        assert readable, "proxy did not report readiness"
        ready = json.loads(process.stdout.readline())
        assert ready["event"] == "ready"
        assert ready["host"] == "127.0.0.1"
        assert ready["pid"] == process.pid
        assert (service_home / ".proxy").is_dir()
        # The prepare helper closes the readiness pipe while the service lives.
        process.stdout.close()
        process.stdout = None
        with socket.create_connection((ready["host"], ready["port"]), timeout=3) as client:
            client.sendall(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n")
            assert receive_headers(client).startswith(b"HTTP/1.1 403")
        process.terminate()
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
