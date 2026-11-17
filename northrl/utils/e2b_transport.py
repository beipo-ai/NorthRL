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

"""Best-effort TCP configuration for established E2B command connections."""

import logging
import os
import socket
from dataclasses import asdict, dataclass, replace
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

E2B_TCP_USER_TIMEOUT_MS = 3_600_000


@dataclass(frozen=True)
class E2BTCPTimeoutResult:
    """Safe endpoint metadata and observed socket configuration results."""

    status: str = "no_matching_sockets"
    host: str | None = None
    port: int | None = None
    timeout_ms: int = E2B_TCP_USER_TIMEOUT_MS
    matched_sockets: int = 0
    configured_sockets: int = 0
    failed_sockets: int = 0
    previous_timeouts_ms: tuple[int, ...] = ()
    current_timeouts_ms: tuple[int, ...] = ()


def _configure_matching_sockets(
    result: E2BTCPTimeoutResult, addresses: set[str], file_descriptors: list[str]
) -> E2BTCPTimeoutResult:
    matched = configured = failed = 0
    previous_values: set[int] = set()
    current_values: set[int] = set()
    for file_descriptor in file_descriptors:
        duplicated_fd = None
        candidate = None
        matching = False
        try:
            duplicated_fd = os.dup(int(file_descriptor))
            candidate = socket.socket(fileno=duplicated_fd)
            # Close only the duplicate; the SDK retains its original descriptor.
            duplicated_fd = None
            peer = candidate.getpeername()
            if not (isinstance(peer, tuple) and peer[0] in addresses and peer[1] == result.port):
                continue
            matching = True
            matched += 1
            previous_values.add(candidate.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT))
            candidate.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, result.timeout_ms)
            current = candidate.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)
            current_values.add(current)
            if current == result.timeout_ms:
                configured += 1
            else:
                failed += 1
        except (OSError, ValueError):
            if matching:
                failed += 1
        finally:
            if candidate is not None:
                candidate.close()
            elif duplicated_fd is not None:
                os.close(duplicated_fd)
    status = "no_matching_sockets"
    if configured:
        status = "partial" if failed else "configured"
    elif failed:
        status = "configuration_failed"
    return replace(
        result,
        status=status,
        matched_sockets=matched,
        configured_sockets=configured,
        failed_sockets=failed,
        previous_timeouts_ms=tuple(sorted(previous_values)),
        current_timeouts_ms=tuple(sorted(current_values)),
    )


def set_e2b_tcp_user_timeout(
    sandbox: Any, timeout_ms: int = E2B_TCP_USER_TIMEOUT_MS
) -> E2BTCPTimeoutResult:
    """Raise TCP stall tolerance on existing sockets matching the envd IP/port.

    Call after a short command has established the SDK's command connection.
    Newly opened connections do not inherit this setting. A shared proxy address
    can match other connections in this process as well. Command and application
    deadlines must still be enforced by the caller.
    """
    result = E2BTCPTimeoutResult(timeout_ms=timeout_ms)
    try:
        endpoint = urlsplit(str(getattr(sandbox, "envd_api_url", "")))
        port = endpoint.port
        if port is None:
            port = {"http": 80, "https": 443}.get(endpoint.scheme)
        if endpoint.hostname is None or port is None or not 1 <= port <= 65535:
            result = replace(result, status="invalid_endpoint")
        else:
            result = replace(result, host=endpoint.hostname, port=port)
            if not hasattr(socket, "TCP_USER_TIMEOUT"):
                result = replace(result, status="unsupported")
            else:
                addresses = {
                    str(item[4][0])
                    for item in socket.getaddrinfo(result.host, port, type=socket.SOCK_STREAM)
                }
                file_descriptors = os.listdir("/proc/self/fd")
                result = _configure_matching_sockets(result, addresses, file_descriptors)
    except ValueError:
        result = replace(result, status="invalid_endpoint")
    except OSError:
        result = replace(result, status="lookup_failed")
    fields = asdict(result)
    logger.info(
        "E2B TCP user timeout: %s",
        "; ".join(f"{key}={value!r}" for key, value in fields.items()),
        extra={"e2b_tcp_user_timeout": fields},
    )
    return result
