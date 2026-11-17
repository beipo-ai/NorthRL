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

"""Validate trusted network configuration and select one task's domain policy.

The default ``denylist`` mode blocks domains and their subdomains; a leading
``*.`` is accepted as equivalent syntax. ``local_only`` permits no external
network access. An explicit task override replaces the whole policy, including
an empty denylist that disables restrictions for that task.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*\Z")


def _normalize_domain(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("network_policy.deny_domains entries must be DNS hostnames")
    hostname = value.removeprefix("*.")
    try:
        hostname = hostname.encode("idna").decode("ascii").lower().removesuffix(".")
    except UnicodeError as exc:
        raise ValueError("network_policy.deny_domains contains an invalid DNS hostname") from exc
    if (
        not hostname
        or len(hostname) > 253
        or any(not _LABEL.fullmatch(label) for label in hostname.split("."))
        or _NUMERIC_HOST.fullmatch(hostname)
    ):
        raise ValueError("network_policy.deny_domains requires DNS hostnames, not URLs or IPs")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return hostname
    raise ValueError("network_policy.deny_domains cannot contain IP addresses")


def _normalize_domains(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("network_policy.deny_domains must be a list of DNS hostnames")
    return list(dict.fromkeys(_normalize_domain(domain) for domain in value))


def _normalize_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    mode = value.get("mode", "denylist")
    if not isinstance(mode, str) or mode not in ("denylist", "local_only"):
        raise ValueError("network_policy.mode must be denylist or local_only")
    domains = _normalize_domains(value.get("deny_domains", []))
    if mode == "local_only" and domains:
        raise ValueError("network_policy.local_only cannot specify nonempty deny_domains")
    rule: dict[str, Any] = {"deny_domains": domains}
    # Preserve legacy output shape unless the mode was explicitly supplied.
    if "mode" in value:
        rule["mode"] = mode
    return rule


def normalize_network_policy(value: Any) -> dict[str, Any]:
    """Return a fresh canonical policy, rejecting malformed or unknown options."""
    if not isinstance(value, Mapping):
        raise ValueError("network_policy must be a mapping; use deny_domains: [] to disable")
    if set(value) - {"mode", "deny_domains", "task_overrides"}:
        raise ValueError("network_policy contains unsupported fields")
    rule = _normalize_rule(value)
    overrides = value.get("task_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("network_policy.task_overrides must be a mapping")
    normalized_overrides = {}
    for task_id, override in overrides.items():
        if not isinstance(task_id, str) or not task_id or task_id != task_id.strip():
            raise ValueError("network_policy.task_overrides keys must be nonblank task IDs")
        if not isinstance(override, Mapping) or set(override) - {"mode", "deny_domains"}:
            raise ValueError(
                "network_policy.task_overrides values support only mode and deny_domains"
            )
        if "deny_domains" not in override and override.get("mode") != "local_only":
            raise ValueError(
                "network_policy.task_overrides values must explicitly specify "
                "deny_domains or mode: local_only"
            )
        normalized_overrides[task_id] = _normalize_rule(override)
    return {**rule, "task_overrides": normalized_overrides}


def resolve_network_policy(value: Any, task_id: str | None) -> dict[str, Any]:
    """Resolve a trusted host policy without transmitting other tasks' rules."""
    policy = normalize_network_policy(value)
    if task_id is not None and (
        not isinstance(task_id, str) or not task_id or task_id != task_id.strip()
    ):
        raise ValueError("network_policy requires metadata.task_id to be a nonblank string")
    selected = policy["task_overrides"].get(task_id, policy)
    resolved: dict[str, Any] = {"deny_domains": list(selected["deny_domains"])}
    if "mode" in selected:
        resolved["mode"] = selected["mode"]
    return resolved
