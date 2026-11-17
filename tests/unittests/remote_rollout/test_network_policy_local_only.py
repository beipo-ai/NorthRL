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

"""Local-only agents keep their model gateway but cannot acquire an egress proxy."""

import asyncio
import json
import stat
from types import SimpleNamespace

import pytest

from northrl.remote_rollout import network_policy as policy
from northrl.remote_rollout.contracts import RolloutEnvironmentError


def _unexpected_proxy_operation(*args, **kwargs):
    pytest.fail("local_only must not start, inspect, or connect to a forwarding proxy")


@pytest.fixture
def no_proxy_operations(monkeypatch):
    monkeypatch.setattr(policy, "_prepare", _unexpected_proxy_operation)
    monkeypatch.setattr(policy, "_check_proxy", _unexpected_proxy_operation)
    monkeypatch.setattr(policy, "_process_start", _unexpected_proxy_operation)
    monkeypatch.setattr(policy.subprocess, "Popen", _unexpected_proxy_operation)
    monkeypatch.setattr(policy.socket, "create_connection", _unexpected_proxy_operation)
    monkeypatch.setattr(policy.shutil, "copyfile", _unexpected_proxy_operation)


@pytest.fixture
def local_installation(tmp_path, monkeypatch, no_proxy_operations):
    """Mock privileged effects while exercising the actual preparation path."""
    calls = []
    binaries = []
    snapshot = {"ip": ["current ipv4 rules"], "ip6": ["current ipv6 rules"]}
    monkeypatch.setattr(policy, "_ROOT", tmp_path / "policy")
    monkeypatch.setattr(policy.os, "geteuid", lambda: 0)
    monkeypatch.setattr(policy, "_trusted", lambda _: None)

    def account(name):
        assert name == "agent", "local_only must not need a separate proxy account"
        return SimpleNamespace(pw_uid=1001, pw_gid=1002)

    def binary(name):
        assert name in {"nft", "setpriv"}, "local_only must not need proxy dependencies"
        binaries.append(name)
        return "/usr/bin/" + name

    def run(command, *, input_text=None):
        calls.append((command, input_text))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(policy.pwd, "getpwnam", account)
    monkeypatch.setattr(policy, "_binary", binary)
    monkeypatch.setattr(policy, "_run", run)
    monkeypatch.setattr(policy, "_firewall_snapshot", lambda _: snapshot)
    return SimpleNamespace(calls=calls, binaries=binaries, snapshot=snapshot)


@pytest.mark.parametrize(
    "configured", [{"mode": "local_only"}, {"mode": "local_only", "deny_domains": []}]
)
def test_empty_domains_still_install_local_only(configured, local_installation):
    with policy.network_policy_context(configured, task_id="task-local"):
        assert policy.network_policy_enabled()
        with pytest.raises(RolloutEnvironmentError, match="did not prepare"):
            policy.assert_network_policy_prepared()
        asyncio.run(policy.prepare_network_policy("agent"))
        policy.assert_network_policy_prepared()
        assert (
            policy.agent_network_environment({}, "agent")["NO_PROXY"] == "localhost,127.0.0.1,::1"
        )

    state_path = policy._ROOT / "state.json"
    state = json.loads(state_path.read_text())
    assert state == {
        "mode": "local_only",
        "agent_user": "agent",
        "uid": 1001,
        "gid": 1002,
        "firewall": "/usr/bin/nft",
        "firewall_snapshot": local_installation.snapshot,
        "deny_domains": [],
        "task_id": "task-local",
    }
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert set(local_installation.binaries) == {"nft", "setpriv"}
    assert len(local_installation.calls) == 1
    command, rules = local_installation.calls[0]
    assert command == ["/usr/bin/nft", "--file", "-"]
    for family in ("ip", "ip6"):
        assert (
            f'add rule {family} {policy._TABLE} output meta skuid 1001 oifname != "lo" reject'
            in rules
        )
        assert (
            f"add rule {family} {policy._TABLE} output meta skuid 1001 tcp dport 49983 reject"
            in rules
        )
    assert not (policy._ROOT / "proxy.py").exists()
    assert not (policy._ROOT / "proxy-home").exists()
    assert not (policy._ROOT / "proxy.log").exists()
    assert not policy.network_policy_enabled()


@pytest.mark.parametrize("failure", ["missing_nft", "firewall_rejected", "incomplete_snapshot"])
def test_failed_installation_never_marks_policy_prepared(local_installation, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise OSError(failure)

    if failure == "missing_nft":
        monkeypatch.setattr(policy, "_binary", fail)
    elif failure == "firewall_rejected":
        monkeypatch.setattr(policy, "_run", fail)
    else:
        monkeypatch.setattr(policy, "_firewall_snapshot", fail)
    with policy.network_policy_context({"mode": "local_only"}, task_id="task-local"):
        with pytest.raises(RolloutEnvironmentError, match=failure):
            asyncio.run(policy.prepare_network_policy("agent"))
        with pytest.raises(RolloutEnvironmentError, match="did not prepare"):
            policy.assert_network_policy_prepared()
        with pytest.raises(RolloutEnvironmentError, match="not ready"):
            policy.agent_network_environment({}, "agent")
    assert not (policy._ROOT / "state.json").exists()


@pytest.mark.parametrize("uid, gid", [(0, 1002), (1001, 0)])
def test_local_only_refuses_root_agent_identity(local_installation, monkeypatch, uid, gid):
    monkeypatch.setattr(policy.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=uid, pw_gid=gid))
    with policy.network_policy_context({"mode": "local_only"}):
        with pytest.raises(RolloutEnvironmentError, match="non-root agent"):
            asyncio.run(policy.prepare_network_policy("agent"))
    assert not local_installation.calls
    assert not policy._ROOT.exists()


@pytest.fixture
def installed_local_only(tmp_path, monkeypatch, no_proxy_operations):
    state = {
        "mode": "local_only",
        "agent_user": "agent",
        "uid": 1001,
        "gid": 1002,
        "firewall": "/usr/bin/nft",
        "firewall_snapshot": {"ip": ["rules"], "ip6": ["rules"]},
        "deny_domains": [],
        "task_id": "task-local",
    }
    monkeypatch.setattr(policy, "_ROOT", tmp_path)
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(policy, "_trusted", lambda _: None)
    monkeypatch.setattr(policy.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=1001, pw_gid=1002))
    monkeypatch.setattr(policy, "_firewall_snapshot", lambda _: state["firewall_snapshot"])
    monkeypatch.setattr(policy, "_binary", lambda name: "/usr/bin/" + name)
    with policy.network_policy_context({"mode": "local_only"}, task_id="task-local"):
        yield state


def test_local_environment_removes_inherited_proxies_and_keeps_model_gateway(installed_local_only):
    gateway = {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
        "OPENAI_BASE_URL": "http://localhost:8000/v1",
        "ANTHROPIC_API_KEY": "test-gateway-key",
        "PATH": "/usr/bin",
    }
    inherited = {
        "HTTP_PROXY": "http://outside-proxy:1",
        "http_proxy": "http://outside-proxy:2",
        "Https_Proxy": "http://outside-proxy:3",
        "HTTPS_PROXY": "http://outside-proxy:4",
        "ALL_PROXY": "socks5://outside-proxy:5",
        "all_proxy": "socks5://outside-proxy:6",
        "NO_PROXY": "*",
        "No_Proxy": "outside.example",
        "no_proxy": "*",
    }
    original = {**gateway, **inherited}
    result = policy.agent_network_environment(original, "agent")
    assert result == {
        **gateway,
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
    }
    assert original == {**gateway, **inherited}


def test_local_launch_drops_privileges_without_changing_agent_arguments(installed_local_only):
    arguments = ["/usr/bin/claude", "--model", "local-model", "literal $(not-a-command)"]
    wrapped = policy.agent_network_command(
        ["runuser", "--user", "agent", "--preserve-environment", "--", *arguments], "agent"
    )
    separator = wrapped.index("--")
    assert wrapped[separator + 1 :] == arguments
    assert wrapped[:5] == ["/usr/bin/setpriv", "--reuid", "1001", "--regid", "1002"]
    assert set(wrapped[5:separator]) == {
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
    }


@pytest.mark.parametrize(
    "failure",
    [
        "missing_state",
        "invalid_state",
        "changed_uid",
        "changed_gid",
        "missing_ipv6",
        "changed_rules",
    ],
)
def test_local_readiness_failure_blocks_both_environment_and_launch(
    installed_local_only, monkeypatch, failure
):
    if failure == "missing_state":
        (policy._ROOT / "state.json").unlink()
    elif failure == "invalid_state":
        (policy._ROOT / "state.json").write_text("not json")
    elif failure in {"changed_uid", "changed_gid"}:
        uid, gid = (2001, 1002) if failure == "changed_uid" else (1001, 2002)
        monkeypatch.setattr(
            policy.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=uid, pw_gid=gid)
        )
    elif failure == "missing_ipv6":

        def missing_snapshot(_):
            raise OSError("IPv6 table missing")

        monkeypatch.setattr(policy, "_firewall_snapshot", missing_snapshot)
    else:
        monkeypatch.setattr(policy, "_firewall_snapshot", lambda _: {"ip": [], "ip6": []})
    command = ["runuser", "--user", "agent", "--preserve-environment", "--", "claude"]
    with pytest.raises(RolloutEnvironmentError, match="not ready"):
        policy.agent_network_environment({}, "agent")
    with pytest.raises(RolloutEnvironmentError, match="not ready"):
        policy.agent_network_command(command, "agent")


@pytest.mark.parametrize(
    "proxy_field, value", [("pid", 12345), ("port", 18080), ("process_start", "123")]
)
def test_local_state_rejects_any_forwarding_proxy(installed_local_only, proxy_field, value):
    state = {**installed_local_only, proxy_field: value}
    (policy._ROOT / "state.json").write_text(json.dumps(state))
    with pytest.raises(RolloutEnvironmentError, match="forwarding proxy"):
        policy.agent_network_environment({}, "agent")


@pytest.mark.parametrize("mode", ["denylist", None, "unknown"])
def test_local_task_rejects_state_from_another_mode(installed_local_only, mode):
    state = dict(installed_local_only)
    if mode is None:
        del state["mode"]
    else:
        state["mode"] = mode
    (policy._ROOT / "state.json").write_text(json.dumps(state))
    with pytest.raises(RolloutEnvironmentError, match="does not match this task"):
        policy.agent_network_environment({}, "agent")


@pytest.mark.parametrize(
    "configured, task_id",
    [
        ({"mode": "denylist", "deny_domains": ["github.com"]}, "task-local"),
        ({"mode": "local_only"}, "other-task"),
    ],
)
def test_local_state_cannot_be_reused_by_another_policy(installed_local_only, configured, task_id):
    with policy.network_policy_context(configured, task_id=task_id):
        with pytest.raises(RolloutEnvironmentError, match="does not match this task"):
            policy.agent_network_environment({}, "agent")


@pytest.mark.parametrize(
    "configured", [None, {"deny_domains": []}, {"mode": "denylist", "deny_domains": []}]
)
def test_installed_local_policy_cannot_be_silently_disabled(installed_local_only, configured):
    with policy.network_policy_context(configured):
        with pytest.raises(RolloutEnvironmentError, match="cannot disable"):
            asyncio.run(policy.prepare_network_policy("agent"))
        with pytest.raises(RolloutEnvironmentError, match="disabled policy"):
            policy.agent_network_environment({}, "agent")


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_legacy_and_explicit_denylists_keep_the_proxy_preparation_path(monkeypatch, explicit_mode):
    calls = []
    monkeypatch.setattr(policy, "_prepare", lambda *args: calls.append(args))
    monkeypatch.setattr(policy, "_prepare_local_only", _unexpected_proxy_operation)
    configured = {"deny_domains": ["*.PYPI.org"]}
    if explicit_mode:
        configured["mode"] = "denylist"
    with policy.network_policy_context(configured, task_id="task-denylist"):
        assert policy.network_policy_enabled()
        asyncio.run(policy.prepare_network_policy("agent"))
        policy.assert_network_policy_prepared()
    assert calls == [("agent", ("pypi.org",), "task-denylist")]
