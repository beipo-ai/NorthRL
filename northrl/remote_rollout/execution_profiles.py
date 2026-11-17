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

"""Resolve host-controlled E2B execution profiles before worker fan-out."""

from typing import cast

from omegaconf import DictConfig, OmegaConf


def get_e2b_execution_configs(config: DictConfig) -> dict[str, DictConfig]:
    """Return effective root configs; the empty key denotes legacy execution.

    A profile changes only ``rollout.execution``. Model, sampling and training
    configuration stay common to every task. Resolve these configs once during
    initialization, then select them by registry key for individual rollouts.
    Explicit profile volume mounts replace the common mapping; empty or null
    mounts disable inherited volumes. Explicit network policies also replace
    the entire common policy, so task overrides cannot leak between profiles.
    Other options retain OmegaConf merging.
    """

    execution = config.rollout.execution
    if "profiles" not in execution:
        return {"": config}

    profiles = execution.profiles
    if not isinstance(profiles, (dict, DictConfig)) or not profiles:
        raise ValueError("rollout.execution.profiles must be a non-empty mapping")
    if "entrypoint" in execution or "input_fields" in execution:
        raise ValueError(
            "rollout.execution.profiles cannot be combined with global entrypoint or input_fields"
        )

    common = OmegaConf.masked_copy(execution, [key for key in execution if key != "profiles"])
    effective_configs: dict[str, DictConfig] = {}
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("rollout.execution.profiles keys must be non-empty strings")
        if not isinstance(profile, (dict, DictConfig)):
            raise ValueError(f"rollout.execution.profiles.{name} must be a mapping")
        if "entrypoint" not in profile or "input_fields" not in profile:
            raise ValueError(
                f"rollout.execution.profiles.{name} requires entrypoint and input_fields"
            )
        effective_config = cast(DictConfig, OmegaConf.merge(config))
        effective_config.rollout.execution = OmegaConf.merge(common, profile)
        if "volume_mounts" in profile:
            effective_config.rollout.execution.volume_mounts = profile["volume_mounts"]
        if "network_policy" in profile:
            effective_config.rollout.execution.network_policy = profile["network_policy"]
        effective_configs[name] = effective_config
    return effective_configs
