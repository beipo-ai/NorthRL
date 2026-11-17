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

"""Task-configured sandbox forward proxy, requiring ``proxy.py==2.4.10``.

This standalone file can be copied to a root-owned, publicly readable directory
and launched with ``python -I /path/network_proxy.py --blocked-domains '[...]'`` after dropping
to a dedicated service UID. The launcher must enforce the agent's egress through
this proxy, protect its code/configuration, and keep the proxy UID separate from
the agent UID. Proxy environment variables alone are not an enforcement boundary.
The service needs a writable, private HOME because proxy.py initializes ~/.proxy.

HTTPS uses CONNECT without TLS interception. Allowed domains can host mirrors or
forwarders; an allowed CONNECT endpoint can also accept a different TLS SNI or
HTTP Host (domain fronting). A domain blacklist cannot prevent these routes to
the same content. HTTP and CONNECT destinations are restricted to ports 80/443.

The first stdout line is JSON with event=ready, host, port, and pid. SIGTERM and
SIGINT shut down the listener and workers. Diagnostics are written to stderr.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import re
import signal
import socket
from typing import NoReturn
from urllib.parse import urlsplit

from proxy import Proxy
from proxy.common.version import __version__ as proxy_version
from proxy.http.exception import HttpRequestRejected
from proxy.http.parser import HttpParser
from proxy.http.proxy import HttpProxyBasePlugin

PROXY_VERSION = "2.4.10"
logger = logging.getLogger(__name__)
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*\Z")
_HTTP_TOKEN = re.compile(rb"[!#$%&'*+.^_`|~0-9a-zA-Z-]+\Z")
# Translation/tunnel addresses can embed non-public IPv4 destinations even when
# older Python ipaddress releases classify the enclosing IPv6 address as global.
_TRANSLATION_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")
)


def normalize_domain(value: str | bytes) -> str:
    """Canonical DNS hostname, excluding literals and legacy numeric IPv4 forms."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("expected a DNS hostname")
    host = value.encode("idna").decode("ascii").lower().rstrip(".")
    if not host or len(host) > 253 or any(not _LABEL.fullmatch(p) for p in host.split(".")):
        raise ValueError("invalid DNS hostname or IP literal")
    if _NUMERIC_HOST.fullmatch(host):
        raise ValueError("IP literals are prohibited")
    return host


def load_blocked_domains(value: str | None) -> tuple[str, ...]:
    if value is None:
        raise ValueError("blocked domains must be explicitly configured")
    domains = json.loads(value)
    if not isinstance(domains, list) or any(not isinstance(item, str) for item in domains):
        raise ValueError("blocked domains must be a JSON array of DNS names")
    return tuple(dict.fromkeys(normalize_domain(item) for item in domains))


def reject(message: str) -> NoReturn:
    raise HttpRequestRejected(
        status_code=403,
        reason=b"Forbidden",
        headers={b"Content-Type": b"text/plain; charset=utf-8"},
        body=("Network policy: " + message + "\n").encode("utf-8"),
    )


def public_address(host: str, port: int) -> str:
    """Resolve once, reject mixed public/private answers, and return a pinned IP."""
    try:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except OSError:
        reject("DNS resolution failed")
    addresses = []
    for family, _, _, _, sockaddr in answers:
        if family not in (socket.AF_INET, socket.AF_INET6):
            reject("unsupported address family")
        address = ipaddress.ip_address(sockaddr[0])
        if any(
            (
                not address.is_global,
                address.is_multicast,
                address.is_reserved,
                address.is_unspecified,
                address.is_loopback,
                address.is_link_local,
                getattr(address, "ipv4_mapped", None) is not None,
                any(address in network for network in _TRANSLATION_NETWORKS),
            )
        ):
            reject("destination does not resolve exclusively to public addresses")
        addresses.append(str(address))
    if not addresses:
        reject("DNS returned no addresses")
    return addresses[0]


class NetworkPolicyPlugin(HttpProxyBasePlugin):
    """Validate before connecting and on every subsequent HTTP request."""

    def _destination_for(self, request: HttpParser) -> tuple[str, int]:
        try:
            # Validate parsed fields before proxy.py serializes them. The pinned
            # parser preserves bare LF inside fields, which a lenient upstream
            # could otherwise interpret as an additional header.
            if request.version not in (b"HTTP/1.0", b"HTTP/1.1") or not _HTTP_TOKEN.fullmatch(
                request.method or b""
            ):
                reject("invalid HTTP request line")
            if any(byte < 32 or byte == 127 for byte in request.path or b""):
                reject("control characters in request target")
            for name, value in (request.headers or {}).values():
                if not _HTTP_TOKEN.fullmatch(name) or any(
                    (byte < 32 and byte != 9) or byte == 127 for byte in value
                ):
                    reject("invalid HTTP header")
            host = normalize_domain(request.host or b"")
            port = request.port
            if port not in (80, 443):
                reject("only destination ports 80 and 443 are permitted")
            for domain in self.flags.blocked_domains:
                if host == domain or host.endswith("." + domain):
                    logger.warning(
                        "network_policy_blocked task_id=%r domain=%s",
                        self.flags.network_task_id,
                        host,
                    )
                    reject("destination domain is blocked")
            # _url is the parsed URL in the pinned proxy.py release. Reject URL
            # credentials/schemes that might be interpreted differently upstream.
            url = request._url  # pylint: disable=protected-access
            if url is None or url.username is not None or url.password is not None:
                reject("URL credentials are prohibited")
            if request.is_https_tunnel:
                if url.scheme is not None or url.remainder is not None:
                    reject("CONNECT requires an authority target")
            elif url.scheme != b"http":
                reject("use an absolute HTTP URL or CONNECT for HTTPS")
            if request.has_header(b"host"):
                raw_host = request.header(b"host").decode("utf-8")
                header = urlsplit("//" + raw_host)
                if (
                    header.username is not None
                    or header.password is not None
                    or header.path
                    or header.query
                    or header.fragment
                ):
                    reject("invalid Host header")
                header_host = normalize_domain(header.hostname or "")
                header_port = header.port or (port if request.is_https_tunnel else 80)
                if (header_host, header_port) != (host, port):
                    reject("Host header and request destination must match")
            elif not request.is_https_tunnel:
                reject("Host header is required")
            if request.has_header(b"transfer-encoding"):
                if (
                    request.has_header(b"content-length")
                    or request.header(b"transfer-encoding").lower() != b"chunked"
                ):
                    reject("ambiguous HTTP message framing")
            elif request.has_header(b"content-length"):
                length = request.header(b"content-length")
                if not re.fullmatch(rb"[0-9]+", length) or int(length) != len(request.body or b""):
                    reject("invalid Content-Length")
                request.add_header(b"Content-Length", str(int(length)).encode("ascii"))
            # The library rebuilds HTTP requests from parsed headers, collapsing
            # duplicates. Canonicalizing Host avoids forwarding an alternate form.
            canonical_host = host if port == 80 else f"{host}:{port}"
            request.add_header(b"Host", canonical_host.encode("ascii"))
            return host, port
        except (UnicodeError, ValueError):
            reject("invalid destination hostname or authority")

    def before_upstream_connection(self, request: HttpParser) -> HttpParser:
        self._destination = self._destination_for(request)
        self._tunnel = request.is_https_tunnel
        self._pinned_ip = public_address(*self._destination)
        return request

    def resolve_dns(self, host: str, port: int) -> tuple[str, None]:
        # Never return None: that would silently fall back to hostname resolution.
        if (normalize_domain(host), port) != self._destination:
            reject("destination changed after DNS validation")
        return self._pinned_ip, None

    def handle_client_request(self, request: HttpParser) -> HttpParser:
        destination = self._destination_for(request)
        if destination != self._destination or request.is_https_tunnel != self._tunnel:
            reject("a proxy connection cannot change its destination or tunnel mode")
        return request


def make_proxy(port: int, blocked_domains: tuple[str, ...], *, task_id: str = "") -> Proxy:
    """Construct without starting; caller owns the context manager lifecycle."""
    if proxy_version != PROXY_VERSION:
        raise RuntimeError(f"requires proxy.py=={PROXY_VERSION}, found {proxy_version}")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    server = Proxy(
        input_args=["--threaded", "--log-level", "WARNING"],
        hostname=ipaddress.ip_address("127.0.0.1"),
        port=port,
        num_acceptors=1,
        num_workers=1,
        plugins=[NetworkPolicyPlugin],
    )
    server.flags.blocked_domains = tuple(normalize_domain(item) for item in blocked_domains)
    server.flags.network_task_id = task_id
    # In 2.4.10, pooled connections do not honor resolve_dns's address override.
    server.flags.enable_conn_pool = False
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--blocked-domains", required=True, help="JSON array of DNS names")
    parser.add_argument("--task-id", default="", help="Task identifier for denied-request logs")
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        parser.error("launch as a dedicated non-root proxy service UID")
    try:
        server = make_proxy(
            args.port, load_blocked_domains(args.blocked_domains), task_id=args.task_id
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    with server:
        print(
            json.dumps(
                {
                    "event": "ready",
                    "host": "127.0.0.1",
                    "port": server.flags.port,
                    "pid": os.getpid(),
                }
            ),
            flush=True,
        )
        while True:
            signal.pause()


if __name__ == "__main__":
    main()
