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

"""Host policy validation and task-specific replacement semantics."""

import copy

import pytest

from northrl.remote_rollout.network_policy_config import (
    normalize_network_policy,
    resolve_network_policy,
)


def test_normalizes_domains_without_mutating_configuration():
    policy = {
        "deny_domains": ["*.GitHub.COM.", "github.com", "BÜCHER.de"],
        "task_overrides": {"task-1": {"deny_domains": ["*.GitLab.COM"]}},
    }
    original = copy.deepcopy(policy)
    assert normalize_network_policy(policy) == {
        "deny_domains": ["github.com", "xn--bcher-kva.de"],
        "task_overrides": {"task-1": {"deny_domains": ["gitlab.com"]}},
    }
    assert policy == original


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {"deny_domain": []},
        {"deny_domains": None},
        {"deny_domains": "github.com"},
        {"deny_domains": [123]},
        {"task_overrides": None},
        {"task_overrides": []},
        {"task_overrides": {"": {}}},
        {"task_overrides": {" task-1": {}}},
        {"task_overrides": {1: {}}},
        {"task_overrides": {"task-1": None}},
        {"task_overrides": {"task-1": {}}},
        {"task_overrides": {"task-1": {"allow_domains": []}}},
        {"task_overrides": {"task-1": {"deny_domains": None}}},
    ],
)
def test_rejects_invalid_policy_shape(policy):
    with pytest.raises(ValueError, match="network_policy"):
        normalize_network_policy(policy)


@pytest.mark.parametrize(
    "domain",
    [
        "",
        "github.com ",
        "https://github.com",
        "github.com/path",
        "github.com:443",
        "user@github.com",
        "127.0.0.1",
        "127.1",
        "2130706433",
        "0x7f000001",
        "::1",
        "[::1]",
        "127。0。0。1",
        "*",
        "*github.com",
        "api.*.github.com",
        "github..com",
        "github.com..",
        "-github.com",
        "github_.com",
        "a" * 64 + ".com",
    ],
)
def test_rejects_urls_ips_and_malformed_domains(domain):
    with pytest.raises(ValueError, match="network_policy"):
        normalize_network_policy({"deny_domains": [domain]})


@pytest.mark.parametrize(
    "task_id, expected",
    [(None, ["github.com"]), ("other", ["github.com"]), ("task-1", ["gitlab.com"]), ("task-2", [])],
)
def test_task_override_replaces_default_list(task_id, expected):
    policy = {
        "deny_domains": ["github.com"],
        "task_overrides": {
            "task-1": {"deny_domains": ["gitlab.com"]},
            "task-2": {"deny_domains": []},
        },
    }
    resolved = resolve_network_policy(policy, task_id)
    assert resolved == {"deny_domains": expected}
    resolved["deny_domains"].append("example.org")
    assert "example.org" not in repr(policy)


@pytest.mark.parametrize("task_id", ["", " task-1", 1, True, []])
def test_invalid_task_identity_is_rejected(task_id):
    with pytest.raises(ValueError, match="metadata.task_id"):
        resolve_network_policy({}, task_id)


def test_missing_fields_default_to_empty():
    assert normalize_network_policy({}) == {"deny_domains": [], "task_overrides": {}}
    assert resolve_network_policy(
        {"task_overrides": {"task-1": {"deny_domains": []}}}, "task-1"
    ) == {"deny_domains": []}


@pytest.mark.parametrize("deny_domains", [None, []])
def test_local_only_normalizes_and_resolves_without_mutation(deny_domains):
    policy = {"mode": "local_only"}
    if deny_domains is not None:
        policy["deny_domains"] = deny_domains
    original = copy.deepcopy(policy)
    normalized = normalize_network_policy(policy)
    assert normalized == {"mode": "local_only", "deny_domains": [], "task_overrides": {}}
    assert resolve_network_policy(normalized, "task-1") == {
        "mode": "local_only",
        "deny_domains": [],
    }
    normalized["deny_domains"].append("example.org")
    assert policy == original


def test_explicit_denylist_mode_is_preserved():
    policy = {"mode": "denylist", "deny_domains": ["*.GitHub.COM"]}
    assert normalize_network_policy(policy) == {
        "mode": "denylist",
        "deny_domains": ["github.com"],
        "task_overrides": {},
    }
    assert resolve_network_policy(policy, None) == {
        "mode": "denylist",
        "deny_domains": ["github.com"],
    }


@pytest.mark.parametrize(
    "default, override, expected",
    [
        (
            {"deny_domains": ["github.com"]},
            {"mode": "local_only"},
            {"mode": "local_only", "deny_domains": []},
        ),
        ({"mode": "local_only"}, {"deny_domains": []}, {"deny_domains": []}),
        (
            {"mode": "local_only"},
            {"deny_domains": ["*.GitLab.COM"]},
            {"deny_domains": ["gitlab.com"]},
        ),
        (
            {"mode": "local_only"},
            {"mode": "denylist", "deny_domains": []},
            {"mode": "denylist", "deny_domains": []},
        ),
        (
            {"mode": "local_only"},
            {"mode": "local_only", "deny_domains": []},
            {"mode": "local_only", "deny_domains": []},
        ),
    ],
)
def test_task_override_replaces_entire_mode_policy(default, override, expected):
    policy = {**default, "task_overrides": {"task-1": override}}
    original = copy.deepcopy(policy)
    normalized = normalize_network_policy(policy)
    assert normalized["task_overrides"]["task-1"] == expected
    assert resolve_network_policy(policy, "other") == resolve_network_policy(default, None)
    resolved = resolve_network_policy(policy, "task-1")
    assert resolved == expected
    resolved["deny_domains"].append("example.org")
    assert policy == original


@pytest.mark.parametrize("mode", [None, "", "LOCAL_ONLY", "local-only", "allowlist", [], 1, True])
def test_rejects_invalid_mode_even_in_unselected_override(mode):
    with pytest.raises(ValueError, match="network_policy"):
        normalize_network_policy({"mode": mode})
    with pytest.raises(ValueError, match="network_policy"):
        resolve_network_policy(
            {"task_overrides": {"task-1": {"mode": mode, "deny_domains": []}}}, "other"
        )


@pytest.mark.parametrize(
    "rule",
    [
        {"mode": "local_only", "deny_domains": ["github.com"]},
        {"mode": "local_only", "deny_domains": None},
        {"mode": "local_only", "deny_domains": "github.com"},
        {"mode": "local_only", "allow_domains": []},
        {"mode": "local_only", "local_ony": True},
    ],
)
def test_local_only_rejects_conflicting_or_unknown_options(rule):
    with pytest.raises(ValueError, match="network_policy"):
        normalize_network_policy(rule)
    with pytest.raises(ValueError, match="network_policy"):
        resolve_network_policy({"task_overrides": {"task-1": rule}}, "other")


@pytest.mark.parametrize("override", [{}, {"mode": "denylist"}])
def test_override_cannot_implicitly_disable_local_only(override):
    with pytest.raises(ValueError, match="network_policy"):
        normalize_network_policy({"mode": "local_only", "task_overrides": {"task-1": override}})
