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

"""Install a task's configured agent egress restriction after trusted preparation.

Only the agent UID is firewalled. The denylist proxy has its own UID;
local_only installs no proxy and permits only loopback services. Neither mode
restricts the root evaluator or local Git objects. Rules last until the
disposable sandbox is destroyed.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import selectors
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from northrl.remote_rollout.contracts import RolloutEnvironmentError
from northrl.remote_rollout.network_policy_config import resolve_network_policy

_ROOT = Path("/opt/northrl-network-policy")
_PROXY_USER = "northrl-netproxy"
_TABLE = "northrl_egress"
_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


@dataclass
class _PolicyContext:
    deny_domains: tuple[str, ...]
    task_id: str | None
    mode: str = "denylist"
    prepared: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode == "local_only" or bool(self.deny_domains)


_POLICY_CONTEXT: ContextVar[_PolicyContext | None] = ContextVar("network_policy", default=None)


@contextmanager
def network_policy_context(
    policy: dict[str, Any] | None, *, task_id: str | None = None
) -> Iterator[None]:
    """Bind host-controlled policy to one prepare, agent, or evaluation hook."""
    selected = {"deny_domains": []} if policy is None else resolve_network_policy(policy, task_id)
    token = _POLICY_CONTEXT.set(
        _PolicyContext(
            tuple(selected["deny_domains"]), task_id, mode=selected.get("mode", "denylist")
        )
    )
    try:
        yield
    finally:
        _POLICY_CONTEXT.reset(token)


def assert_network_policy_prepared() -> None:
    """A configured policy must never silently disappear in an unsupported adapter."""
    context = _POLICY_CONTEXT.get()
    if context is not None and context.enabled and not context.prepared:
        raise RolloutEnvironmentError(
            "This entrypoint did not prepare the configured network policy; "
            "integrate prepare_network_policy and the agent launch helpers first"
        )


def network_policy_enabled() -> bool:
    """Whether this hook must launch its agent under a restricted non-root UID."""
    context = _POLICY_CONTEXT.get()
    return context is not None and context.enabled


def _trusted(path: Path) -> None:
    if path.is_symlink():
        raise RolloutEnvironmentError(f"network policy path must not be a symlink: {path}")
    metadata = path.stat()
    if metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError(
            f"network policy path must be root-owned and protected: {path}"
        )


def _binary(name: str) -> str:
    binary = shutil.which(name, path=_SYSTEM_PATH)
    if binary is None:
        raise RolloutEnvironmentError(f"network policy requires {name} in the sandbox template")
    resolved = Path(binary).resolve(strict=True)
    for path in (resolved, *resolved.parents, *Path(binary).parents):
        _trusted(path.resolve(strict=True))
    # Preserve argv[0] for tools implemented by multicall binaries.
    return binary


def _run(command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            input=input_text,
            timeout=15,
            env={"PATH": _SYSTEM_PATH, "LANG": "C"},
            cwd="/",
        )
    except subprocess.CalledProcessError as exc:
        raise RolloutEnvironmentError(
            f"network setup command failed: {command!r}: {(exc.stderr or '')[-1500:]}"
        ) from exc


def _unprivileged_command(uid: int, gid: int) -> list[str]:
    return [
        _binary("setpriv"),
        "--reuid",
        str(uid),
        "--regid",
        str(gid),
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--",
    ]


def _firewall_rules(uid: int) -> str:
    """One atomic transaction; E2B supports ip/ip6, but not inet or xt_owner."""
    lines = []
    for family in ("ip", "ip6"):
        table = f"{family} {_TABLE}"
        lines.extend(
            [
                f"create table {table}",
                f"add chain {table} output {{ type filter hook output priority -150; policy accept; }}",
                # Protect E2B's root command daemon, including its loopback address.
                f"add rule {table} output meta skuid {uid} tcp dport 49983 reject",
                # Keep the model gateway, filtering proxy and local test servers.
                f'add rule {table} output meta skuid {uid} oifname != "lo" reject',
            ]
        )
    return "\n".join(lines) + "\n"


def _firewall_snapshot(binary: str) -> dict[str, Any]:
    snapshot = {}
    for family in ("ip", "ip6"):
        result = _run([binary, "--json", "--numeric", "list", "table", family, _TABLE])
        entries = json.loads(result.stdout)["nftables"]
        # Version metadata is irrelevant; table/chain/rule content and handles
        # must remain exactly as installed. These rules contain no counters.
        snapshot[family] = [entry for entry in entries if "metainfo" not in entry]
        if len(snapshot[family]) != 4:  # One table, one chain, two rules.
            raise RolloutEnvironmentError(f"unexpected agent {family} firewall contents")
    return snapshot


def _process_start(pid: int) -> str:
    # comm may contain spaces or parentheses; starttime is field 22.
    return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]


def _check_proxy(port: int, deny_domains: tuple[str, ...] | list[str]) -> None:
    domain = deny_domains[0]
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.sendall(
            f"CONNECT {domain}:443 HTTP/1.1\r\nHost: {domain}:443\r\n\r\n".encode("ascii")
        )
        response = connection.recv(1024)
    if not response.startswith(b"HTTP/1.1 403"):
        raise RolloutEnvironmentError("network blacklist proxy failed its blocked-domain check")


def _prepare_local_only(agent_user: str, task_id: str | None) -> None:
    """Keep trusted preparation online; block the agent without a forwarding proxy."""
    if os.geteuid() != 0:
        raise RolloutEnvironmentError("network policy preparation requires root")
    agent = pwd.getpwnam(agent_user)
    if agent.pw_uid == 0 or agent.pw_gid == 0:
        raise RolloutEnvironmentError("network policy requires a non-root agent account")
    firewall = _binary("nft")
    _binary("setpriv")
    _trusted(_ROOT.parent)
    _ROOT.mkdir(mode=0o755)
    _ROOT.chmod(0o755)
    _trusted(_ROOT)
    _run([firewall, "--file", "-"], input_text=_firewall_rules(agent.pw_uid))
    state = {
        "mode": "local_only",
        "agent_user": agent_user,
        "uid": agent.pw_uid,
        "gid": agent.pw_gid,
        "firewall": firewall,
        "firewall_snapshot": _firewall_snapshot(firewall),
        "deny_domains": [],
        "task_id": task_id,
    }
    fd = os.open(_ROOT / "state.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(state, file)


def _prepare(agent_user: str, deny_domains: tuple[str, ...], task_id: str | None) -> None:
    if os.geteuid() != 0:
        raise RolloutEnvironmentError("network policy preparation requires root")
    agent = pwd.getpwnam(agent_user)
    if agent.pw_uid == 0 or agent.pw_gid == 0:
        raise RolloutEnvironmentError("network policy requires a non-root agent account")
    # Native skuid avoids xt_owner, which is absent in the E2B guest kernel.
    firewall = _binary("nft")
    _binary("setpriv")
    _trusted(_ROOT.parent)
    _ROOT.mkdir(mode=0o755)  # One prepare per disposable sandbox; never reuse stale state.
    _ROOT.chmod(0o755)
    _trusted(_ROOT)
    source = Path(__file__).with_name("network_proxy.py")
    _trusted(source)
    proxy_script = _ROOT / "proxy.py"
    shutil.copyfile(source, proxy_script)
    proxy_script.chmod(0o444)
    try:
        proxy_user = pwd.getpwnam(_PROXY_USER)
    except KeyError:
        _run(
            [
                _binary("useradd"),
                "--system",
                "--no-create-home",
                "--home-dir",
                "/nonexistent",
                "--shell",
                "/usr/sbin/nologin",
                _PROXY_USER,
            ]
        )
        proxy_user = pwd.getpwnam(_PROXY_USER)
    if proxy_user.pw_uid in (0, agent.pw_uid) or proxy_user.pw_gid == 0:
        raise RolloutEnvironmentError("network proxy requires a separate unprivileged account")
    # proxy.py initializes ~/.proxy even without TLS interception. Only the
    # service UID may write here; policy code and state remain root-owned.
    proxy_home = _ROOT / "proxy-home"
    proxy_home.mkdir(mode=0o700)
    os.chown(proxy_home, proxy_user.pw_uid, proxy_user.pw_gid)
    log_path = _ROOT / "proxy.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    process: subprocess.Popen[str] | None = None
    try:
        with os.fdopen(log_fd, "w") as log:
            # The service intentionally outlives prepare and the agent process.
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                [
                    *_unprivileged_command(proxy_user.pw_uid, proxy_user.pw_gid),
                    sys.executable,
                    "-I",
                    str(proxy_script),
                    "--port",
                    "0",
                    "--blocked-domains",
                    json.dumps(deny_domains),
                    "--task-id",
                    task_id or "",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                start_new_session=True,
                cwd="/",
                env={"PATH": _SYSTEM_PATH, "HOME": str(proxy_home), "LANG": "C"},
            )
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=20):
                raise RolloutEnvironmentError("network proxy did not become ready in 20 seconds")
            line = process.stdout.readline()
            if not line:
                raise RolloutEnvironmentError(
                    "network proxy exited before readiness: "
                    + log_path.read_text(errors="replace")[-1500:]
                )
            ready = json.loads(line)
        if (
            ready.get("event") != "ready"
            or ready.get("host") != "127.0.0.1"
            or ready.get("pid") != process.pid
            or not isinstance(ready.get("port"), int)
            or not 1024 <= ready["port"] <= 65535
        ):
            raise RolloutEnvironmentError("network proxy returned invalid readiness data")
        process.stdout.close()
        _check_proxy(ready["port"], deny_domains)
        _run([firewall, "--file", "-"], input_text=_firewall_rules(agent.pw_uid))
        firewall_snapshot = _firewall_snapshot(firewall)
        state = {
            "agent_user": agent_user,
            "uid": agent.pw_uid,
            "gid": agent.pw_gid,
            "port": ready["port"],
            "pid": process.pid,
            "process_start": _process_start(process.pid),
            "firewall": firewall,
            "firewall_snapshot": firewall_snapshot,
            "deny_domains": list(deny_domains),
            "task_id": task_id,
        }
        fd = os.open(_ROOT / "state.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(state, file)
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        raise


async def prepare_network_policy(agent_user: str) -> None:
    """Install the trusted policy; only an absent policy or empty denylist is off."""
    context = _POLICY_CONTEXT.get()
    if context is None or not context.enabled:
        if _ROOT.exists():
            raise RolloutEnvironmentError(
                "cannot disable a network policy already installed in this sandbox"
            )
        return
    try:
        if context.mode == "local_only":
            await asyncio.to_thread(_prepare_local_only, agent_user, context.task_id)
        else:
            await asyncio.to_thread(_prepare, agent_user, context.deny_domains, context.task_id)
        context.prepared = True
    except Exception as exc:
        raise RolloutEnvironmentError(f"network policy preparation failed: {exc}") from exc


def _load_policy(agent_user: str) -> dict[str, Any] | None:
    try:
        context = _POLICY_CONTEXT.get()
        if context is None or not context.enabled:
            if _ROOT.exists():
                raise ValueError("an installed policy conflicts with this task's disabled policy")
            return None
        _trusted(_ROOT)
        path = _ROOT / "state.json"
        _trusted(path)
        state = json.loads(path.read_text())
        if (
            state.get("mode", "denylist") != context.mode
            or state["deny_domains"] != list(context.deny_domains)
            or state["task_id"] != context.task_id
        ):
            raise ValueError("installed network policy does not match this task")
        agent = pwd.getpwnam(agent_user)
        if (state["agent_user"], state["uid"], state["gid"]) != (
            agent_user,
            agent.pw_uid,
            agent.pw_gid,
        ) or agent.pw_uid == 0:
            raise ValueError("agent identity changed after network preparation")
        if context.mode == "local_only":
            if any(key in state for key in ("pid", "port", "process_start")):
                raise ValueError("local-only policy must not have a forwarding proxy")
        else:
            if _process_start(state["pid"]) != state["process_start"]:
                raise ValueError("network proxy process changed after preparation")
            _check_proxy(state["port"], state["deny_domains"])
        if _firewall_snapshot(state["firewall"]) != state["firewall_snapshot"]:
            raise ValueError("network firewall changed after preparation")
        return state
    except Exception as exc:
        raise RolloutEnvironmentError(f"network policy is not ready: {exc}") from exc


def agent_network_environment(environment: dict[str, str], agent_user: str) -> dict[str, str]:
    state = _load_policy(agent_user)
    if state is None:
        return environment.copy()
    result = {
        key: value
        for key, value in environment.items()
        if key.lower()
        not in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        }
    }
    if state.get("mode", "denylist") != "local_only":
        proxy_url = f"http://127.0.0.1:{state['port']}"
        for key in ("http_proxy", "https_proxy"):
            result[key] = result[key.upper()] = proxy_url
    result["no_proxy"] = result["NO_PROXY"] = "localhost,127.0.0.1,::1"
    return result


def agent_network_command(command: list[str], agent_user: str) -> list[str]:
    state = _load_policy(agent_user)
    if state is None:
        return list(command)
    prefix = ["runuser", "--user", agent_user, "--preserve-environment", "--"]
    if command[: len(prefix)] != prefix or len(command) == len(prefix):
        raise RolloutEnvironmentError("unexpected agent command; refusing unrestricted launch")
    return [*_unprivileged_command(state["uid"], state["gid"]), *command[len(prefix) :]]
