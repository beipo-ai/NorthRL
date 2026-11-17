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

"""Exercise E2B TCP configuration without creating a remote sandbox."""

import logging
import os
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from northrl.utils import e2b_transport

LINUX_TCP = pytest.mark.skipif(
    not hasattr(socket, "TCP_USER_TIMEOUT") or not Path("/proc/self/fd").is_dir(),
    reason="TCP_USER_TIMEOUT configuration requires Linux",
)


@contextmanager
def _connected_pair():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.socket() as client:
            client.connect(("127.0.0.1", port))
            accepted, _ = listener.accept()
            with accepted:
                yield client, accepted, port


def _timeout(connection):
    return connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)


@LINUX_TCP
def test_configures_connected_peer_and_reports_previous_and_current_values(caplog):
    with _connected_pair() as (client, accepted, port):
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30_000)
        accepted_timeout = _timeout(accepted)
        sandbox = SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")

        with caplog.at_level(logging.INFO, logger="northrl.utils.e2b_transport"):
            result = e2b_transport.set_e2b_tcp_user_timeout(sandbox)

        assert result.status == "configured"
        assert result.host == "127.0.0.1"
        assert result.port == port
        assert result.timeout_ms == e2b_transport.E2B_TCP_USER_TIMEOUT_MS == 3_600_000
        assert result.matched_sockets == result.configured_sockets == 1
        assert result.failed_sockets == 0
        assert result.previous_timeouts_ms == (30_000,)
        assert result.current_timeouts_ms == (3_600_000,)
        assert _timeout(client) == 3_600_000
        assert _timeout(accepted) == accepted_timeout
        event = next(
            record.e2b_tcp_user_timeout
            for record in caplog.records
            if hasattr(record, "e2b_tcp_user_timeout")
        )
        assert event["status"] == "configured"
        assert event["matched_sockets"] == event["configured_sockets"] == 1
        assert event["previous_timeouts_ms"] == (30_000,)
        assert event["current_timeouts_ms"] == (3_600_000,)


@LINUX_TCP
def test_other_peer_is_unchanged_and_repeated_configuration_keeps_socket_open():
    with _connected_pair() as (client, accepted, port):
        with _connected_pair() as (other_client, _, _):
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30_000)
            other_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 45_000)
            sandbox = SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")

            first = e2b_transport.set_e2b_tcp_user_timeout(sandbox)
            second = e2b_transport.set_e2b_tcp_user_timeout(sandbox)

            assert first.previous_timeouts_ms == (30_000,)
            assert second.previous_timeouts_ms == (3_600_000,)
            assert second.current_timeouts_ms == (3_600_000,)
            assert second.configured_sockets == 1
            assert _timeout(other_client) == 45_000
            client.sendall(b"still open")
            assert accepted.recv(10) == b"still open"


@LINUX_TCP
def test_non_socket_descriptors_do_not_leak_or_close():
    with _connected_pair() as (client, _, port):
        with tempfile.TemporaryFile() as ordinary_file:
            before_fds = len(os.listdir("/proc/self/fd"))
            result = e2b_transport.set_e2b_tcp_user_timeout(
                SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")
            )

            assert result.status == "configured"
            assert len(os.listdir("/proc/self/fd")) == before_fds
            ordinary_file.write(b"still open")
            ordinary_file.seek(0)
            assert ordinary_file.read() == b"still open"
            assert _timeout(client) == 3_600_000


@LINUX_TCP
def test_returns_no_matching_sockets_for_unconnected_endpoint():
    with socket.socket() as unused_listener:
        unused_listener.bind(("127.0.0.1", 0))
        port = unused_listener.getsockname()[1]
        result = e2b_transport.set_e2b_tcp_user_timeout(
            SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")
        )

    assert result.status == "no_matching_sockets"
    assert result.matched_sockets == result.configured_sockets == result.failed_sockets == 0
    assert result.previous_timeouts_ms == result.current_timeouts_ms == ()


def test_missing_tcp_user_timeout_is_best_effort(monkeypatch):
    monkeypatch.delattr(e2b_transport.socket, "TCP_USER_TIMEOUT", raising=False)
    result = e2b_transport.set_e2b_tcp_user_timeout(
        SimpleNamespace(envd_api_url="https://judge.example.test")
    )

    assert result.status == "unsupported"
    assert result.matched_sockets == result.configured_sockets == 0


@LINUX_TCP
@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "not-a-url",
        "http://[broken",
        "https://judge.test:invalid",
        "https://judge.test:0",
        "https://judge.test:65536",
    ],
)
def test_invalid_endpoint_is_best_effort(endpoint):
    result = e2b_transport.set_e2b_tcp_user_timeout(SimpleNamespace(envd_api_url=endpoint))

    assert result.status == "invalid_endpoint"
    assert result.matched_sockets == result.configured_sockets == 0


@LINUX_TCP
@pytest.mark.parametrize("scheme, expected_port", [("http", 80), ("https", 443)])
def test_dns_failure_is_best_effort(monkeypatch, scheme, expected_port):
    def fail_lookup(host, port, **_kwargs):
        assert host == "judge.example.test"
        assert port == expected_port
        raise socket.gaierror("resolver unavailable")

    monkeypatch.setattr(e2b_transport.socket, "getaddrinfo", fail_lookup)
    result = e2b_transport.set_e2b_tcp_user_timeout(
        SimpleNamespace(envd_api_url=f"{scheme}://judge.example.test")
    )

    assert result.status == "lookup_failed"
    assert result.host == "judge.example.test"
    assert result.port == expected_port
    assert result.matched_sockets == result.configured_sockets == 0


@LINUX_TCP
def test_logs_do_not_include_endpoint_credentials_path_or_query(caplog):
    with _connected_pair() as (_, _, port):
        sandbox = SimpleNamespace(
            envd_api_url=(
                f"http://private-user:private-password@127.0.0.1:{port}"
                "/private-path?token=private-token"
            )
        )
        with caplog.at_level(logging.INFO, logger="northrl.utils.e2b_transport"):
            result = e2b_transport.set_e2b_tcp_user_timeout(sandbox)

    assert result.status == "configured"
    assert caplog.records
    for record in caplog.records:
        text = record.getMessage() + repr(getattr(record, "e2b_tcp_user_timeout", {}))
        assert all(
            secret not in text
            for secret in ("private-user", "private-password", "private-path", "private-token")
        )


class _SocketOptionFailure:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def setsockopt(self, *_args, **_kwargs):
        raise OSError("TCP option unavailable")


@LINUX_TCP
def test_socket_option_failure_reports_failure_and_keeps_socket_open(monkeypatch):
    with _connected_pair() as (client, accepted, port):
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30_000)
        original_socket = socket.socket

        def failing_socket(*args, **kwargs):
            return _SocketOptionFailure(original_socket(*args, **kwargs))

        monkeypatch.setattr(e2b_transport.socket, "socket", failing_socket)
        result = e2b_transport.set_e2b_tcp_user_timeout(
            SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")
        )

        assert result.status == "configuration_failed"
        assert result.matched_sockets == result.failed_sockets == 1
        assert result.configured_sockets == 0
        assert _timeout(client) == 30_000
        client.sendall(b"open")
        assert accepted.recv(4) == b"open"


@LINUX_TCP
def test_one_failed_socket_does_not_prevent_other_connections_from_being_configured(monkeypatch):
    with _connected_pair() as (first_client, _, port):
        with socket.socket() as second_client:
            second_client.connect(("127.0.0.1", port))
            first_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 30_000)
            second_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 45_000)
            failing_local_address = second_client.getsockname()
            original_socket = socket.socket

            def selective_failure(*args, **kwargs):
                candidate = original_socket(*args, **kwargs)
                if candidate.getsockname() == failing_local_address:
                    return _SocketOptionFailure(candidate)
                return candidate

            monkeypatch.setattr(e2b_transport.socket, "socket", selective_failure)
            result = e2b_transport.set_e2b_tcp_user_timeout(
                SimpleNamespace(envd_api_url=f"http://127.0.0.1:{port}")
            )

            assert result.status == "partial"
            assert result.matched_sockets == 2
            assert result.configured_sockets == result.failed_sockets == 1
            assert result.previous_timeouts_ms == (30_000, 45_000)
            assert result.current_timeouts_ms == (3_600_000,)
            assert _timeout(first_client) == 3_600_000
            assert _timeout(second_client) == 45_000
