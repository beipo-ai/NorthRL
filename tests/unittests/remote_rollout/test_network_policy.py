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

"""Checks for per-task policy isolation and fail-closed UID enforcement."""

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from northrl.remote_rollout import network_policy as policy
from northrl.remote_rollout.contracts import RolloutEnvironmentError


@pytest.fixture
def installed(tmp_path, monkeypatch):
    state = {
        "agent_user": "agent",
        "uid": 1001,
        "gid": 1002,
        "pid": 12345,
        "process_start": "123",
        "port": 18080,
        "firewall": "/usr/sbin/nft",
        "firewall_snapshot": {"ip": ["rules"], "ip6": ["rules"]},
        "deny_domains": ["github.com"],
        "task_id": "task-one",
    }
    monkeypatch.setattr(policy, "_ROOT", tmp_path)
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(policy, "_trusted", lambda _: None)
    monkeypatch.setattr(policy.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=1001, pw_gid=1002))
    monkeypatch.setattr(policy, "_process_start", lambda _: "123")
    monkeypatch.setattr(policy, "_check_proxy", lambda *_: None)
    monkeypatch.setattr(policy, "_firewall_snapshot", lambda _: state["firewall_snapshot"])
    monkeypatch.setattr(policy, "_binary", lambda name: "/usr/bin/" + name)
    with policy.network_policy_context({"deny_domains": ["github.com"]}, task_id="task-one"):
        yield state


def test_environment_overrides_bypass_variables_without_mutating_input(installed):
    original = {
        "ALL_PROXY": "socks5://other:1",
        "https_proxy": "http://other:2",
        "NO_PROXY": "*",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    }
    result = policy.agent_network_environment(original, "agent")
    assert "ALL_PROXY" not in result
    assert result["https_proxy"] == result["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert result["http_proxy"] == result["HTTP_PROXY"] == "http://127.0.0.1:18080"
    assert result["NO_PROXY"] == result["no_proxy"] == "localhost,127.0.0.1,::1"
    assert result["ANTHROPIC_BASE_URL"] == original["ANTHROPIC_BASE_URL"]
    assert original["NO_PROXY"] == "*"


def test_launcher_drops_ids_groups_and_capabilities(installed):
    command = policy.agent_network_command(
        ["runuser", "--user", "agent", "--preserve-environment", "--", "/usr/bin/claude", "fix it"],
        "agent",
    )
    assert command[:5] == ["/usr/bin/setpriv", "--reuid", "1001", "--regid", "1002"]
    for flag in (
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
    ):
        assert flag in command
    assert command[-3:] == ["--", "/usr/bin/claude", "fix it"]
    with pytest.raises(RolloutEnvironmentError, match="unexpected agent command"):
        policy.agent_network_command(["sudo", "/usr/bin/claude"], "agent")


@pytest.mark.parametrize(
    "failure", ["missing_state", "changed_uid", "dead_proxy", "ipv6_rule_missing", "changed_rules"]
)
def test_policy_readiness_failure_prevents_launch(installed, monkeypatch, failure):
    if failure == "missing_state":
        (policy._ROOT / "state.json").unlink()
    elif failure == "changed_uid":
        monkeypatch.setattr(policy.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=0, pw_gid=0))
    elif failure == "dead_proxy":
        monkeypatch.setattr(policy, "_process_start", lambda _: "999")
    elif failure == "ipv6_rule_missing":

        def snapshot(binary):
            raise subprocess.CalledProcessError(1, [binary, "list", "table", "ip6", policy._TABLE])

        monkeypatch.setattr(policy, "_firewall_snapshot", snapshot)
    else:
        monkeypatch.setattr(policy, "_firewall_snapshot", lambda _: {"ip": [], "ip6": []})
    with pytest.raises(RolloutEnvironmentError, match="not ready"):
        policy.agent_network_environment({}, "agent")


def test_preparation_failure_is_environment_error(monkeypatch):
    def fail(*_):
        raise PermissionError("netfilter unavailable")

    monkeypatch.setattr(policy, "_prepare", fail)
    with policy.network_policy_context({"deny_domains": ["github.com"]}):
        with pytest.raises(RolloutEnvironmentError, match="netfilter unavailable"):
            asyncio.run(policy.prepare_network_policy("agent"))


def test_trusted_binary_symlink_keeps_program_name(tmp_path, monkeypatch):
    multicall = tmp_path / "busybox"
    multicall.touch()
    binary = tmp_path / "useradd"
    binary.symlink_to(multicall)
    monkeypatch.setattr(policy.shutil, "which", lambda *_, **__: str(binary))
    monkeypatch.setattr(policy, "_trusted", lambda _: None)
    assert policy._binary("useradd") == str(binary)


def test_binary_rejects_untrusted_parent_of_symlink(tmp_path, monkeypatch):
    target = tmp_path / "trusted-executable"
    target.touch()
    untrusted = tmp_path / "writable-path-directory"
    untrusted.mkdir()
    binary = untrusted / "nft"
    binary.symlink_to(target)
    monkeypatch.setattr(policy.shutil, "which", lambda *_, **__: str(binary))

    def trusted(path):
        if path == untrusted:
            raise RolloutEnvironmentError("untrusted PATH directory")

    monkeypatch.setattr(policy, "_trusted", trusted)
    with pytest.raises(RolloutEnvironmentError, match="untrusted PATH directory"):
        policy._binary("nft")


def test_firewall_uses_native_uid_match_for_both_e2b_address_families():
    rules = policy._firewall_rules(1001)
    for family in ("ip", "ip6"):
        assert f"create table {family} {policy._TABLE}" in rules
        assert f"add chain {family} {policy._TABLE} output" in rules
        assert (
            f"add rule {family} {policy._TABLE} output meta skuid 1001 tcp dport 49983 reject"
            in rules
        )
        assert (
            f'add rule {family} {policy._TABLE} output meta skuid 1001 oifname != "lo" reject'
            in rules
        )
    assert "flush" not in rules
    assert "inet" not in rules  # The actual E2B guest kernel lacks this family.
    assert rules.count("hook output priority -150; policy accept;") == 2


def test_snapshot_checks_both_families_and_ignores_only_version_metadata(monkeypatch):
    commands = []
    entries = [
        {"table": {"handle": 1}},
        {"chain": {"hook": "output", "prio": -150}},
        {"rule": {"expr": ["control"]}},
        {"rule": {"expr": ["egress"]}},
    ]

    def run(command):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"nftables": [{"metainfo": {}}, *entries]}))

    monkeypatch.setattr(policy, "_run", run)
    assert policy._firewall_snapshot("nft") == {"ip": entries, "ip6": entries}
    assert commands == [
        ["nft", "--json", "--numeric", "list", "table", family, policy._TABLE]
        for family in ("ip", "ip6")
    ]


def test_missing_rule_prevents_snapshot_publication(monkeypatch):
    monkeypatch.setattr(policy, "_run", lambda _: SimpleNamespace(stdout='{"nftables": []}'))
    with pytest.raises(RolloutEnvironmentError, match="unexpected agent ip firewall"):
        policy._firewall_snapshot("nft")


def test_symlink_state_is_rejected(tmp_path):
    target = tmp_path / "real-state"
    target.touch()
    link = tmp_path / "state.json"
    link.symlink_to(target)
    with pytest.raises(RolloutEnvironmentError, match="symlink"):
        policy._trusted(link)


@pytest.mark.parametrize("configured", [None, {"deny_domains": []}])
def test_disabled_policy_does_not_touch_privileges_or_network(tmp_path, monkeypatch, configured):
    monkeypatch.setattr(policy, "_ROOT", tmp_path / "absent-state")

    def install(*_):
        pytest.fail("disabled policy must not install anything")

    monkeypatch.setattr(policy, "_prepare", install)
    command = ["runuser", "--user", "agent", "--preserve-environment", "--", "claude"]
    environment = {"HTTPS_PROXY": "http://existing-proxy", "NO_PROXY": "*"}
    with policy.network_policy_context(configured, task_id="task-off"):
        asyncio.run(policy.prepare_network_policy("agent"))
        policy.assert_network_policy_prepared()
        assert policy.agent_network_command(command, "agent") == command
        assert policy.agent_network_environment(environment, "agent") == environment
    assert not policy._ROOT.exists()


def test_absent_policy_preserves_legacy_empty_task_id():
    with policy.network_policy_context(None, task_id=""):
        policy.assert_network_policy_prepared()


def test_prepare_uses_selected_domains_and_guard_does_not_leak_between_tasks(monkeypatch):
    calls = []
    monkeypatch.setattr(policy, "_prepare", lambda *args: calls.append(args))
    with policy.network_policy_context({"deny_domains": ["*.PYPI.org"]}, task_id="task-one"):
        with pytest.raises(RolloutEnvironmentError, match="did not prepare"):
            policy.assert_network_policy_prepared()
        asyncio.run(policy.prepare_network_policy("agent"))
        policy.assert_network_policy_prepared()
    assert calls == [("agent", ("pypi.org",), "task-one")]
    with policy.network_policy_context({"deny_domains": ["github.com"]}, task_id="task-two"):
        with pytest.raises(RolloutEnvironmentError, match="did not prepare"):
            policy.assert_network_policy_prepared()
    assert policy._POLICY_CONTEXT.get() is None


@pytest.mark.parametrize(
    "domains, task_id", [(["pypi.org"], "task-one"), (["github.com"], "task-two")]
)
def test_installed_policy_cannot_be_reused_by_another_task(installed, domains, task_id):
    with policy.network_policy_context({"deny_domains": domains}, task_id=task_id):
        with pytest.raises(RolloutEnvironmentError, match="does not match this task"):
            policy.agent_network_environment({}, "agent")


def test_proxy_healthcheck_uses_current_task_domain(monkeypatch):
    class Connection:
        sent = b""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def sendall(self, data):
            self.sent += data

        def recv(self, _):
            return b"HTTP/1.1 403 Forbidden\r\n\r\n"

    connection = Connection()
    monkeypatch.setattr(policy.socket, "create_connection", lambda *_, **__: connection)
    policy._check_proxy(18080, ["pypi.org"])
    assert b"CONNECT pypi.org:443" in connection.sent
    assert b"github.com" not in connection.sent
