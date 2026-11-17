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

"""Tests for the pydantic config schema (see CONFIG_REFACTOR.md).

Covers:
  * positive: the reference recipe `recipe/math/weaver.yaml` validates clean
  * negative: each cross-block validator rejects a targeted mutation
  * strictness: extra_field typo rejected by extra="forbid"
"""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl.utils.config_schema import AlgorithmConfig, MISConfig, NorthRLConfig, OptimConfig
from northrl.utils.config_utils import get_derived_sampling_flags
from northrl.utils.data_dumper import DataDumper

# Anchor: project root, two levels up from this file (tests/unittests/utils/)
REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_RECIPE = REPO_ROOT / "recipe" / "math" / "weaver.yaml"


def _load_recipe(path: Path) -> dict[str, Any]:
    """Load a recipe yaml with `${oc.env:...}` interpolations stubbed.

    Recipes use OmegaConf interpolations that yaml.safe_load can't
    resolve. We textually substitute placeholder values so the schema can validate
    the shape independent of any env state.
    """
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"\$\{oc\.env:NORTHRL_DATA_PATH\}", "/data", raw)
    raw = re.sub(r"\$\{oc\.env:EXPERIMENT_PATH\}", "/exp", raw)
    raw = re.sub(r"\$\{oc\.env:FEISHU_WEBHOOK_URL,?\}", "", raw)
    return yaml.safe_load(raw)


def _load_reference_recipe() -> dict[str, Any]:
    return _load_recipe(REFERENCE_RECIPE)


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Exercise the same canonical configuration consumed by the local driver."""
    return copy.deepcopy(_load_reference_recipe())


def _set_algorithm_option(algorithm, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        algorithm = algorithm.setdefault(key, {})
    algorithm[keys[-1]] = value


# ============================================================
# Positive: reference recipe validates
# ============================================================


class TestReferenceRecipe:
    def test_reference_recipe_validates(self, base_config: dict[str, Any]) -> None:
        cfg = NorthRLConfig.model_validate(_load_reference_recipe())
        assert cfg.model_dump() == NorthRLConfig.model_validate(base_config).model_dump()

        # Sanity-check a few field paths to make sure the model is actually populated,
        # not just silently shaped right.
        assert cfg.experiment.name == "weaver-single_turn_math_qwen3_8b"
        assert cfg.models.actor.name == "Qwen/Qwen3-8B"
        assert cfg.models.actor.lora is not None
        assert cfg.models.actor.lora.enabled is True
        assert cfg.models.actor.lora.rank == 32
        assert cfg.algorithm.type == "grpo"
        assert cfg.algorithm.kl.enabled is False
        assert cfg.algorithm.router_replay.enabled is False
        assert cfg.training.weight_sync.mode == "sync"
        assert cfg.backend.type == "weaver"
        assert cfg.training.total_steps == 100
        assert cfg.training.epochs is None
        assert cfg.training.resume.max_retries == 1
        assert cfg.runtime.progress.log_every_n_steps == 10

    def test_resolved_config_round_trip(self, base_config: dict[str, Any]) -> None:
        before = copy.deepcopy(base_config)
        canonical = NorthRLConfig.model_validate(base_config).model_dump(by_alias=True)
        assert base_config == before
        assert NorthRLConfig.model_validate(canonical).model_dump(by_alias=True) == canonical
        through_omegaconf = OmegaConf.to_container(OmegaConf.create(canonical), resolve=True)
        assert (
            NorthRLConfig.model_validate(through_omegaconf).model_dump(by_alias=True) == canonical
        )

    def test_validate_data_uses_datasets_dict(self, base_config: dict[str, Any]) -> None:
        cfg = NorthRLConfig.model_validate(base_config)
        # decision: validate.data only accepts datasets dict (no top-level files shorthand)
        assert list(cfg.validate_.data.datasets.keys()) == ["math"]

    def test_max_tokens_derivation_space(self, base_config: dict[str, Any]) -> None:
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.data.max_prompt_length is not None
        derived = cfg.data.max_sequence_length - cfg.data.max_prompt_length
        # rollout.sampling.max_tokens unset → derived to (12288 - 4096 = 8192) at runtime
        assert cfg.rollout.sampling.max_tokens is None
        assert cfg.rollout.sampling.reasoning_effort is None
        assert cfg.validate_.sampling.reasoning_effort is None
        assert derived == 8192

    def test_resume_max_retries_must_be_non_negative(self, base_config: dict[str, Any]) -> None:
        base_config["training"]["resume"]["max_retries"] = -1

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            NorthRLConfig.model_validate(base_config)

    def test_legacy_cross_entropy_algorithm_is_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["type"] = "cross_entropy"

        with pytest.raises(ValidationError):
            NorthRLConfig.model_validate(base_config)


class TestBackendPerformanceTier:
    @pytest.mark.parametrize(
        "backend_overrides,expected_tier",
        [
            ({}, None),
            ({"performance_tier": None}, None),
            ({"performance_tier": "normal"}, "normal"),
            ({"performance_tier": "fast"}, "fast"),
            ({"performance_tier": "flash"}, "flash"),
        ],
        ids=["omitted", "null", "normal", "fast", "flash"],
    )
    def test_supported_tiers_round_trip(self, base_config, backend_overrides, expected_tier):
        base_config["backend"].pop("performance_tier", None)
        base_config["backend"].update(backend_overrides)

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.backend.performance_tier == expected_tier
        assert cfg.model_dump()["backend"]["performance_tier"] == expected_tier

    def test_misspelled_tier_is_rejected(self, base_config):
        base_config["backend"]["performance_tier"] = "flahs"

        with pytest.raises(ValidationError, match=r"backend\.performance_tier"):
            NorthRLConfig.model_validate(base_config)

    def test_actor_performance_tier_is_rejected(self, base_config):
        base_config["models"]["actor"]["performance_tier"] = "flash"

        with pytest.raises(ValidationError) as exc_info:
            NorthRLConfig.model_validate(base_config)

        assert any(
            error["loc"] == ("models", "actor", "performance_tier")
            and error["type"] == "extra_forbidden"
            for error in exc_info.value.errors()
        )


class TestBackendSchedulingPolicy:
    @pytest.mark.parametrize(
        "overrides,expected",
        [
            ({}, None),
            ({"scheduling_policy": None}, None),
            (
                {"scheduling_policy": {"mode": "fixed", "clusters": ["cluster-1"]}},
                {"mode": "fixed", "clusters": ["cluster-1"]},
            ),
        ],
        ids=["omitted", "null", "explicit"],
    )
    def test_scheduling_policy_round_trip(self, base_config, overrides, expected):
        base_config["backend"].update(overrides)

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.backend.scheduling_policy == expected
        assert cfg.model_dump()["backend"]["scheduling_policy"] == expected

    def test_non_mapping_scheduling_policy_is_rejected(self, base_config):
        base_config["backend"]["scheduling_policy"] = ["cluster-1"]

        with pytest.raises(ValidationError, match=r"backend\.scheduling_policy"):
            NorthRLConfig.model_validate(base_config)


class TestWeightSyncPauseMode:
    @pytest.fixture
    def full_ft_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        base_config["training"]["weight_sync"]["mode"] = "fully-async"
        base_config["models"]["actor"]["lora"] = None
        return base_config

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    @pytest.mark.parametrize("algorithm", ["grpo", "opd"])
    @pytest.mark.parametrize(
        "lora_overrides",
        [{}, {"lora": None}, {"lora": {"enabled": False}}],
        ids=["omitted-lora", "null-lora", "disabled-lora"],
    )
    def test_supported_modes_round_trip(
        self, full_ft_config, pause_mode, algorithm, lora_overrides
    ) -> None:
        full_ft_config["training"]["weight_sync"]["pause_mode"] = pause_mode
        full_ft_config["algorithm"]["type"] = algorithm
        if algorithm == "opd":
            full_ft_config["models"]["teachers"] = [{"name": "Qwen/Qwen3-32B"}]
        actor = full_ft_config["models"]["actor"]
        actor.pop("lora")
        actor.update(lora_overrides)

        cfg = NorthRLConfig.model_validate(full_ft_config)

        assert cfg.training.weight_sync.pause_mode == pause_mode
        assert cfg.model_dump()["training"]["weight_sync"]["pause_mode"] == pause_mode

    @pytest.mark.parametrize("explicit_null", [False, True], ids=["omitted", "null"])
    @pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async", "no-sync"])
    @pytest.mark.parametrize("backend", ["weaver", "mock"])
    def test_disabled_pause_preserves_existing_configs(
        self, base_config, explicit_null, mode, backend
    ) -> None:
        weight_sync = base_config["training"]["weight_sync"]
        weight_sync.update(mode=mode, freeze_inflight=False)
        weight_sync.pop("pause_mode", None)
        if explicit_null:
            weight_sync["pause_mode"] = None
        base_config["backend"]["type"] = backend

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.models.actor.lora.enabled is True
        assert cfg.training.weight_sync.pause_mode is None
        assert cfg.model_dump()["training"]["weight_sync"]["pause_mode"] is None

    @pytest.mark.parametrize("pause_mode", ["", "in-place", "RETRACT", "abort"])
    def test_invalid_mode_is_rejected(self, full_ft_config, pause_mode) -> None:
        full_ft_config["training"]["weight_sync"]["pause_mode"] = pause_mode

        with pytest.raises(ValidationError, match=r"training\.weight_sync\.pause_mode"):
            NorthRLConfig.model_validate(full_ft_config)

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    @pytest.mark.parametrize("mode", ["sync", "batch-async", "no-sync"])
    def test_pause_requires_fully_async(self, full_ft_config, pause_mode, mode) -> None:
        full_ft_config["training"]["weight_sync"].update(mode=mode, pause_mode=pause_mode)

        with pytest.raises(ValidationError, match="pause_mode requires fully-async RL"):
            NorthRLConfig.model_validate(full_ft_config)

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    def test_pause_requires_weaver(self, full_ft_config, pause_mode) -> None:
        full_ft_config["training"]["weight_sync"]["pause_mode"] = pause_mode
        full_ft_config["backend"]["type"] = "mock"

        with pytest.raises(ValidationError, match="pause_mode requires backend.type='weaver'"):
            NorthRLConfig.model_validate(full_ft_config)

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    def test_pause_rejects_lora(self, full_ft_config, pause_mode) -> None:
        full_ft_config["training"]["weight_sync"]["pause_mode"] = pause_mode
        full_ft_config["models"]["actor"]["lora"] = {"enabled": True, "rank": 32}

        with pytest.raises(ValidationError, match="pause_mode requires full_ft"):
            NorthRLConfig.model_validate(full_ft_config)

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    def test_pause_requires_freeze_inflight(self, full_ft_config, pause_mode) -> None:
        full_ft_config["training"]["weight_sync"].update(
            pause_mode=pause_mode, freeze_inflight=False
        )

        with pytest.raises(ValidationError, match="pause_mode requires freeze_inflight=true"):
            NorthRLConfig.model_validate(full_ft_config)


class TestGroupOrderConfig:
    @pytest.mark.parametrize(
        "configured_order,expected_groups",
        [(None, ["z", "a"]), ("fifo", ["z", "a"]), ("lifo", ["a", "z"])],
        ids=["default-fifo", "explicit-fifo", "lifo"],
    )
    def test_public_group_order_controls_pool_selection(
        self, base_config: dict[str, Any], configured_order, expected_groups
    ) -> None:
        from northrl.controller import NorthRLController
        from northrl.northrl_types import Trajectory
        from northrl.trajectory_pool import GroupedTrajectoryStore

        train = base_config["data"]["train"]
        train["batch_size"] = 1
        train["rollout_repeat_n"] = 2
        if configured_order is not None:
            train["group_order"] = configured_order
        base_config["training"]["weight_sync"]["mode"] = "fully-async"
        cfg = NorthRLConfig.model_validate(base_config)
        controller = object.__new__(NorthRLController)
        controller._config = OmegaConf.create(cfg.model_dump(by_alias=True))
        pool_config = controller._derive_trajectory_pool_config()
        store = GroupedTrajectoryStore(pool_config, "actor")

        # Completion order deliberately differs from alphabetical group order.
        for group in ("z", "a"):
            for run in range(2):
                store.put_trajectory(
                    Trajectory(
                        tokens=[1, 2],
                        loss_mask=[0, 1],
                        reward=float(run),
                        is_val=False,
                        extra_fields={
                            "group_id": group,
                            "run_id": run,
                            "rollout_id": f"{group}-{run}",
                            "expected_rollouts": 2,
                        },
                    )
                )

        for group in expected_groups:
            batch = store.take_groups(num_groups=pool_config.groups_per_batch)
            assert batch is not None
            assert [(traj["group_id"], traj["run_id"]) for traj in batch] == [
                (group, 0),
                (group, 1),
            ]
        assert store.is_empty()

    @pytest.mark.parametrize("group_order", ["latest", None, True])
    def test_invalid_group_order_is_rejected(self, base_config, group_order) -> None:
        base_config["data"]["train"]["group_order"] = group_order
        with pytest.raises(ValidationError, match=r"data\.train\.group_order"):
            NorthRLConfig.model_validate(base_config)


class TestOptimizerBatchConfig:
    def test_separate_optimizer_and_forward_backward_batch_sizes(self) -> None:
        cfg = OptimConfig.model_validate({"lr": 1.0e-6, "batch_size": 64, "mini_batch_size": 16})

        assert cfg.batch_size == 64
        assert cfg.mini_batch_size == 16

    def test_forward_backward_batch_size_defaults_to_unsplit(self) -> None:
        cfg = OptimConfig.model_validate({"lr": 1.0e-6, "batch_size": 64})

        assert cfg.mini_batch_size is None

    @pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
    @pytest.mark.parametrize("explicit_null", [False, True])
    def test_supported_modes_default_to_one_update_per_pool_batch(
        self, base_config: dict[str, Any], mode: str, explicit_null: bool
    ) -> None:
        base_config["training"]["weight_sync"]["mode"] = mode
        optim = base_config["algorithm"]["optim"]
        optim.pop("batch_size", None)
        if explicit_null:
            optim["batch_size"] = None
        optim["mini_batch_size"] = 16

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.optim.batch_size is None
        assert cfg.algorithm.optim.mini_batch_size == 16

    @pytest.mark.parametrize("explicit_null", [False, True])
    def test_no_sync_rl_requires_an_explicit_optimizer_batch_size(
        self, base_config: dict[str, Any], explicit_null: bool
    ) -> None:
        base_config["training"]["weight_sync"]["mode"] = "no-sync"
        optim = base_config["algorithm"]["optim"]
        optim.pop("batch_size", None)
        if explicit_null:
            optim["batch_size"] = None

        with pytest.raises(ValidationError, match=r"algorithm\.optim\.batch_size.*sync"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async", "no-sync"])
    def test_explicit_optimizer_batch_size_is_supported_in_each_mode(
        self, base_config: dict[str, Any], mode: str
    ) -> None:
        base_config["training"]["weight_sync"]["mode"] = mode
        base_config["algorithm"]["optim"]["batch_size"] = 64

        assert NorthRLConfig.model_validate(base_config).algorithm.optim.batch_size == 64

    @pytest.mark.parametrize("field", ["batch_size", "mini_batch_size"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_batch_sizes_must_be_positive_when_set(self, field: str, value: int) -> None:
        config = {"lr": 1.0e-6, "batch_size": 64, "mini_batch_size": 16, field: value}

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OptimConfig.model_validate(config)


class TestReasoningEffort:
    @pytest.mark.parametrize("effort", [None, "high", "none"])
    def test_e2b_sampling(self, base_config, effort):
        TestE2BEntrypointConfig._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["sampling"]["reasoning_effort"] = effort
        base_config["validate"]["sampling"]["reasoning_effort"] = "low"

        config = NorthRLConfig.model_validate(base_config)

        assert config.rollout.sampling.reasoning_effort == effort
        assert config.validate_.sampling.reasoning_effort == "low"

    @pytest.mark.parametrize("block", ["rollout", "validate"])
    def test_non_e2b_rejected(self, base_config, block):
        base_config[block]["sampling"]["reasoning_effort"] = "high"
        with pytest.raises(ValidationError, match="reasoning_effort.*e2b"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("block", ["rollout", "validate"])
    @pytest.mark.parametrize("effort", ["", 123])
    def test_invalid_effort(self, base_config, block, effort):
        TestE2BEntrypointConfig._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config[block]["sampling"]["reasoning_effort"] = effort
        with pytest.raises(ValidationError, match="reasoning_effort"):
            NorthRLConfig.model_validate(base_config)


class TestRolloutSelection:
    def test_flat_rollout_round_trips(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"].update(
            type="custom",
            custom_module_path="workers/custom.py",
            params={"max_turns": 5},
        )
        cfg = NorthRLConfig.model_validate(base_config)
        dumped = cfg.model_dump(by_alias=True)

        assert cfg.rollout.custom_class_name == "CustomRolloutWorker"
        assert dumped["rollout"]["type"] == "custom"
        assert dumped["rollout"]["custom_module_path"] == "workers/custom.py"
        assert dumped["rollout"]["params"] == {"max_turns": 5}
        assert cfg.rollout.params == {"max_turns": 5}
        assert cfg.rollout.execution == {}
        assert "agent" not in dumped["rollout"]
        assert "resource" not in dumped["rollout"]
        assert NorthRLConfig.model_validate(dumped).rollout == cfg.rollout

    @pytest.mark.parametrize("keep_new_type", [False, True])
    def test_legacy_agent_block_requires_migration(
        self, base_config: dict[str, Any], keep_new_type: bool
    ) -> None:
        if not keep_new_type:
            base_config["rollout"].pop("type")
        base_config["rollout"]["agent"] = {"type": "single_turn", "params": {}}

        with pytest.raises(ValidationError, match=r"rollout\.agent.*rollout\.type"):
            NorthRLConfig.model_validate(base_config)

    def test_removed_agent_type_requires_migration(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["type"] = "agent"

        with pytest.raises(ValidationError, match=r"rollout\.type='agent'.*single_turn"):
            NorthRLConfig.model_validate(base_config)

    def test_unused_resource_block_requires_removal(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["resource"] = {"workers_per_node": 0, "agents_per_worker": 32}

        with pytest.raises(ValidationError, match=r"rollout\.resource"):
            NorthRLConfig.model_validate(base_config)


class TestE2BEntrypointConfig:
    @staticmethod
    def _set_e2b_rollout(base_config: dict[str, Any], entrypoint: dict[str, str]) -> None:
        base_config["rollout"].update(
            {
                "type": "e2b",
                "execution": {
                    "input_fields": ["instruction", "interface"],
                    "entrypoint": entrypoint,
                },
            }
        )

    @pytest.mark.parametrize("with_execution", [False, True])
    def test_e2b_legacy_params_requires_migration(
        self, base_config: dict[str, Any], with_execution: bool
    ) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["params"] = copy.deepcopy(base_config["rollout"]["execution"])
        if not with_execution:
            del base_config["rollout"]["execution"]

        with pytest.raises(ValidationError, match=r"rollout\.params.*rollout\.execution"):
            NorthRLConfig.model_validate(base_config)

    def test_e2b_allows_default_empty_params(self, base_config: dict[str, Any]) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["params"] = {}

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.params == {}
        assert cfg.rollout.execution["input_fields"] == ["instruction", "interface"]

    def test_e2b_requires_entrypoint(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"].update({"type": "e2b", "execution": {}})

        with pytest.raises(ValidationError, match="entrypoint is required"):
            NorthRLConfig.model_validate(base_config)

    def test_e2b_requires_explicit_input_fields(self, base_config: dict[str, Any]) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        del base_config["rollout"]["execution"]["input_fields"]

        with pytest.raises(ValidationError, match="input_fields is required"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "input_fields",
        [
            None,
            "instruction",
            {"instruction": True},
            ("instruction",),
            [None],
            [1],
            [True],
            [""],
            ["  "],
            ["instruction", "instruction"],
            [["instruction"]],
        ],
    )
    def test_e2b_rejects_invalid_input_fields(self, base_config, input_fields) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["execution"]["input_fields"] = input_fields

        with pytest.raises(ValidationError, match="input_fields must be a list"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("input_fields", [[], ["question", "context"], [" question "]])
    def test_e2b_preserves_explicit_input_field_names(self, base_config, input_fields) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["execution"]["input_fields"] = input_fields

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.execution["input_fields"] == input_fields

    @pytest.mark.parametrize(
        "entrypoint",
        [
            {"target": " /opt/tasks/openhands_rollout.py "},
            {"target": " /scripts/openhands_rollout.py "},
        ],
    )
    def test_entrypoint_script_paths_validate(
        self, base_config: dict[str, Any], entrypoint: dict[str, str]
    ) -> None:
        self._set_e2b_rollout(base_config, entrypoint)

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.type is not None
        assert "tokenizer" not in cfg.rollout.execution
        assert cfg.rollout.execution["env_error_retries"] == 3
        assert "timeout_retries" not in cfg.rollout.execution
        assert cfg.rollout.execution["entrypoint"]["target"] == entrypoint["target"].strip()

    def test_entrypoint_runtime_paths_validate_and_strip(
        self,
        base_config: dict[str, Any],
    ) -> None:
        self._set_e2b_rollout(
            base_config,
            {
                "target": "/tasks/swerebench_rollout.py",
                "task_files_root": " /data/evaluation/tasks ",
                "workspace": " /workspace/my-task ",
            },
        )

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.type == "e2b"
        assert cfg.rollout.execution["entrypoint"] == {
            "target": "/tasks/swerebench_rollout.py",
            "task_files_root": "/data/evaluation/tasks",
            "workspace": "/workspace/my-task",
        }

    @pytest.mark.parametrize("field", ["task_files_root", "workspace"])
    def test_entrypoint_rejects_blank_runtime_path(
        self, base_config: dict[str, Any], field: str
    ) -> None:
        entrypoint = {"target": "/tasks/agent.py", field: " "}
        self._set_e2b_rollout(base_config, entrypoint)
        with pytest.raises(ValidationError, match=field):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("target", "agent.py"),
            ("target", "~/agent.py"),
            ("task_files_root", "evaluation"),
            ("task_files_root", "~/evaluation"),
            ("workspace", "relative/path"),
        ],
    )
    def test_entrypoint_requires_absolute_paths(self, base_config, field, value) -> None:
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py", field: value})
        with pytest.raises(ValidationError, match=rf"entrypoint\.{field} must be an absolute"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("env_error_retries", [True, -1, 1.5, "1", None])
    def test_e2b_rejects_invalid_env_error_retries(
        self,
        base_config: dict[str, Any],
        env_error_retries: Any,
    ) -> None:
        self._set_e2b_rollout(
            base_config,
            {"target": "/tasks/openhands_rollout.py"},
        )
        base_config["rollout"]["execution"]["env_error_retries"] = env_error_retries

        with pytest.raises(
            ValidationError, match="env_error_retries must be a non-negative integer"
        ):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("retries", [0, 3])
    def test_e2b_environment_retry_budget_survives_validation(self, base_config, retries):
        self._set_e2b_rollout(base_config, {"target": "/tasks/swe.py"})
        base_config["rollout"]["execution"]["env_error_retries"] = retries
        assert (
            NorthRLConfig.model_validate(base_config).rollout.execution["env_error_retries"]
            == retries
        )

    def test_e2b_rejects_ignored_legacy_timeout_retry_setting(self, base_config):
        self._set_e2b_rollout(base_config, {"target": "/tasks/swe.py"})
        base_config["rollout"]["execution"]["timeout_retries"] = 1
        with pytest.raises(ValidationError, match="use env_error_retries"):
            NorthRLConfig.model_validate(base_config)

    def test_e2b_does_not_require_sandbox_tokenizer(self, base_config):
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        cfg = NorthRLConfig.model_validate(base_config)
        assert "tokenizer" not in cfg.rollout.execution
        assert "chat_template" not in cfg.rollout.execution
        assert cfg.rollout.execution["gateway_publication_timeout_seconds"] == 300

    @pytest.mark.parametrize("timeout", [0, -1, True, "300", float("inf"), float("nan")])
    def test_e2b_rejects_invalid_gateway_publication_timeout(self, base_config, timeout):
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["execution"]["gateway_publication_timeout_seconds"] = timeout
        with pytest.raises(ValidationError, match="gateway_publication_timeout_seconds"):
            NorthRLConfig.model_validate(base_config)

    def test_e2b_accepts_gateway_publication_timeout(self, base_config):
        self._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        base_config["rollout"]["execution"]["gateway_publication_timeout_seconds"] = 120.5
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.rollout.execution["gateway_publication_timeout_seconds"] == 120.5

    @pytest.mark.parametrize(
        "entrypoint",
        [
            {"target": "/tasks/agent"},
            {"target": "   "},
            {},
            {"target": "/scripts/entrypoint"},
        ],
    )
    def test_invalid_entrypoint_is_rejected(
        self, base_config: dict[str, Any], entrypoint: dict[str, str]
    ) -> None:
        self._set_e2b_rollout(base_config, entrypoint)

        with pytest.raises(ValidationError):
            NorthRLConfig.model_validate(base_config)

    def test_entrypoint_extra_field_is_rejected(self, base_config: dict[str, Any]) -> None:
        self._set_e2b_rollout(
            base_config,
            {
                "target": "/tasks/openhands_rollout.py",
                "args": ["--unsafe"],
            },
        )

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            NorthRLConfig.model_validate(base_config)

    def test_top_level_remote_rollout_is_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["remote_rollout"] = {
            "entrypoint": {
                "target": "/tasks/openhands_rollout.py",
            }
        }

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            NorthRLConfig.model_validate(base_config)


class TestResponseParserConfig:
    def test_builtin_response_parser_defaults_validate(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {"type": "builtin"}

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.parsers.model_dump() == {
            "type": "builtin",
            "reasoning": "think_tag",
            "tool": "qwen25",
        }

    def test_builtin_response_parser_can_disable_stages(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {
            "type": "builtin",
            "reasoning": None,
            "tool": None,
        }

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.parsers.model_dump() == {
            "type": "builtin",
            "reasoning": None,
            "tool": None,
        }

    def test_custom_response_parser_config_validates(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {
            "type": "custom",
            "module_path": " custom_response_parser.py ",
        }

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.parsers.model_dump() == {
            "type": "custom",
            "module_path": "custom_response_parser.py",
        }

    def test_builtin_response_parser_rejects_module_path(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {
            "type": "builtin",
            "module_path": "custom_response_parser.py",
        }

        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            NorthRLConfig.model_validate(base_config)

    def test_custom_response_parser_rejects_builtin_fields(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["rollout"]["parsers"] = {
            "type": "custom",
            "module_path": "custom_response_parser.py",
            "tool": "qwen25",
        }

        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            NorthRLConfig.model_validate(base_config)

    def test_custom_response_parser_requires_module_path(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {"type": "custom"}

        with pytest.raises(ValidationError, match=r"Field required"):
            NorthRLConfig.model_validate(base_config)

    def test_response_parser_rejects_unknown_type(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["parsers"] = {"type": "unknown"}

        with pytest.raises(ValidationError, match=r"Input tag 'unknown'"):
            NorthRLConfig.model_validate(base_config)

    def test_legacy_builtin_parser_config_keeps_previous_defaults(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["rollout"]["parsers"] = {
            "reasoning": None,
            "tool": None,
        }

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.parsers.model_dump() == {
            "type": "builtin",
            "reasoning": "think_tag",
            "tool": "qwen25",
        }


# ============================================================
# Negative: cross-block validators
# ============================================================


class TestSFTPipeline:
    @staticmethod
    def _make_sft_config(base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["type"] = "sft"
        base_config["algorithm"]["loss"]["mode"] = "cross_entropy"
        base_config.pop("rollout")
        base_config["data"].pop("max_prompt_length")
        base_config["data"]["train"]["files"] = ["train.jsonl"]
        base_config["data"]["train"].pop("rollout_repeat_n")
        base_config["validate"] = {"enabled": False}
        base_config["training"]["weight_sync"]["mode"] = "no-sync"

    @pytest.mark.parametrize("explicit_null", [False, True])
    def test_sft_accepts_an_unspecified_optimizer_batch_size(
        self, base_config: dict[str, Any], explicit_null: bool
    ) -> None:
        self._make_sft_config(base_config)
        optim = base_config["algorithm"]["optim"]
        optim.pop("batch_size", None)
        if explicit_null:
            optim["batch_size"] = None

        assert NorthRLConfig.model_validate(base_config).algorithm.optim.batch_size is None

    def test_sft_data_options_validate(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["render_chat"] = "custom_sft.py:render_chat"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.data.train.render_chat == "custom_sft.py:render_chat"
        assert cfg.data.max_prompt_length is None
        assert cfg.data.train.rollout_repeat_n is None
        assert cfg.rollout.num_workers == 1
        assert cfg.rollout.type is None

    @pytest.mark.parametrize("omit_batch_size", [False, True])
    def test_sft_keeps_full_token_batch_with_null_or_omitted_optimizer_batch_size(
        self, base_config: dict[str, Any], omit_batch_size: bool
    ) -> None:
        self._make_sft_config(base_config)
        optim = base_config["algorithm"]["optim"]
        optim["batch_size"] = None
        if omit_batch_size:
            del optim["batch_size"]

        assert NorthRLConfig.model_validate(base_config).algorithm.optim.batch_size is None

    def test_sft_requires_two_tokens_for_model_bound_training_length(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["max_sequence_length"] = 1

        with pytest.raises(
            ValidationError,
            match=r"algorithm\.type='sft'.*max_sequence_length >= 2",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_sft_managed_mixture_requires_explicit_versions_and_total(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"].pop("files")
        base_config["data"]["train"].update(
            {
                "seed": 7,
                "total_samples": 10,
                "sources": [
                    {"type": "local", "files": ["local.jsonl"], "ratio": 0.8},
                    {
                        "type": "managed",
                        "dataset": "hq-math",
                        "version": "2026-08",
                        "ratio": 0.2,
                    },
                ],
            }
        )

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.data.train.files is None
        assert cfg.data.train.total_samples == 10
        assert cfg.data.train.seed == 7
        assert cfg.data.train.sources is not None
        assert cfg.data.train.sources[1].version == "2026-08"  # type: ignore[union-attr]
        assert cfg.data.train.managed_variant_ready_timeout_seconds == 300.0
        assert cfg.data.train.managed_variant_poll_initial_seconds == 1.0
        assert cfg.data.train.managed_variant_poll_max_seconds == 10.0

    def test_managed_variant_poll_initial_must_not_exceed_maximum(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["managed_variant_poll_initial_seconds"] = 2.0
        base_config["data"]["train"]["managed_variant_poll_max_seconds"] = 1.0

        with pytest.raises(ValidationError, match="poll_initial_seconds must not exceed"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        ("field_name", "value", "message"),
        [
            ("dataset", " hq-math", "boundary whitespace"),
            ("version", "2026-08 ", "boundary whitespace"),
            ("dataset", "hq/math", "single safe path segment"),
            ("version", r"2026\\08", "single safe path segment"),
            ("dataset", ".", "single safe path segment"),
            ("version", "..", "single safe path segment"),
            ("dataset", "hq\nmath", "control characters"),
            ("dataset", "d" * 161, "at most 160"),
            ("version", "v" * 129, "at most 128"),
        ],
    )
    def test_sft_managed_identity_is_one_strict_path_segment(
        self,
        base_config: dict[str, Any],
        field_name: str,
        value: str,
        message: str,
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"].pop("files")
        source = {
            "type": "managed",
            "dataset": "hq math",
            "version": "2026-08",
            "ratio": 1.0,
        }
        source[field_name] = value
        base_config["data"]["train"].update({"total_samples": 1, "sources": [source]})

        with pytest.raises(ValidationError, match=message):
            NorthRLConfig.model_validate(base_config)

    def test_sft_mixture_rejects_missing_total(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"].pop("files")
        base_config["data"]["train"]["sources"] = [
            {
                "type": "managed",
                "dataset": "hq-math",
                "version": "2026-08",
                "ratio": 1.0,
            }
        ]

        with pytest.raises(ValidationError, match="total_samples is required"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_mixture_rejects_disabled_global_shuffle(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"].pop("files")
        base_config["data"]["train"].update(
            {
                "total_samples": 4,
                "shuffle": False,
                "sources": [
                    {
                        "type": "managed",
                        "dataset": "hq-math",
                        "version": "2026-08",
                        "ratio": 1.0,
                    }
                ],
            }
        )

        with pytest.raises(ValidationError, match="global deterministic shuffle"):
            NorthRLConfig.model_validate(base_config)

    def test_train_files_and_sources_are_mutually_exclusive(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["total_samples"] = 1
        base_config["data"]["train"]["sources"] = [
            {"type": "local", "files": ["other.jsonl"], "ratio": 1.0}
        ]

        with pytest.raises(ValidationError, match="exactly one of files or sources"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_accepts_epoch_training(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["training"].pop("total_steps")
        base_config["training"]["epochs"] = 3

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.training.total_steps is None
        assert cfg.training.epochs == 3

    def test_sft_accepts_cosine_lr_schedule(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["training"].pop("total_steps")
        base_config["training"]["epochs"] = 1
        base_config["algorithm"]["optim"].update(
            {
                "lr": 2.0e-5,
                "schedule": {
                    "style": "cosine",
                    "min_lr": 2.0e-6,
                    "warmup_fraction": 0.03,
                },
            }
        )

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.optim.lr == 2.0e-5
        assert cfg.algorithm.optim.schedule is not None
        assert cfg.algorithm.optim.schedule.min_lr == 2.0e-6
        assert cfg.algorithm.optim.schedule.warmup_fraction == 0.03

    def test_lr_schedule_rejects_min_lr_above_peak(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["algorithm"]["optim"]["schedule"] = {
            "style": "cosine",
            "min_lr": 2.0,
            "warmup_fraction": 0.03,
        }

        with pytest.raises(ValidationError, match="min_lr cannot exceed"):
            NorthRLConfig.model_validate(base_config)

    def test_lr_schedule_is_sft_only(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["optim"]["schedule"] = {
            "style": "cosine",
            "min_lr": 1.0e-6,
            "warmup_fraction": 0.03,
        }

        with pytest.raises(ValidationError, match="currently supported only"):
            NorthRLConfig.model_validate(base_config)

    def test_training_length_rejects_both_modes(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["training"]["epochs"] = 3

        with pytest.raises(ValidationError, match="exactly one"):
            NorthRLConfig.model_validate(base_config)

    def test_training_length_rejects_missing_mode(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["training"].pop("total_steps")

        with pytest.raises(ValidationError, match="exactly one"):
            NorthRLConfig.model_validate(base_config)

    def test_epochs_are_sft_only(self, base_config: dict[str, Any]) -> None:
        base_config["training"].pop("total_steps")
        base_config["training"]["epochs"] = 3

        with pytest.raises(
            ValidationError,
            match=r"training\.epochs is only supported when algorithm\.type='sft'",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_sft_derives_cross_entropy_loss_when_omitted(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        del base_config["algorithm"]["loss"]["mode"]
        del base_config["training"]["weight_sync"]
        base_config["data"]["train"]["rollout_repeat_n"] = 8

        with pytest.warns(UserWarning, match="rollout_repeat_n is not configurable"):
            cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.loss.mode == "cross_entropy"
        assert cfg.data.train.rollout_repeat_n is None
        assert cfg.training.weight_sync.mode == "no-sync"

    def test_sft_rejects_non_cross_entropy_loss(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["algorithm"]["loss"]["mode"] = "importance_sampling"

        with pytest.raises(
            ValidationError,
            match="requires algorithm.loss.mode='cross_entropy'",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_sft_rejects_custom_loss(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["algorithm"]["entropy_coeff"] = 0
        base_config["algorithm"]["loss"] = {
            "mode": "custom",
            "params": {"name": "direct_logprob"},
        }

        with pytest.raises(
            ValidationError,
            match="algorithm.loss.mode",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_sft_rollout_repeat_warns_even_when_pipeline_is_invalid(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["rollout_repeat_n"] = 4
        base_config["backend"]["type"] = "mock"

        with pytest.warns(UserWarning, match="rollout_repeat_n is not configurable"):
            with pytest.raises(
                ValidationError,
                match="requires backend.type='weaver'",
            ):
                NorthRLConfig.model_validate(base_config)

    def test_sft_render_chat_requires_sft_algorithm(self, base_config: dict[str, Any]) -> None:
        base_config["data"]["train"]["render_chat"] = "custom_sft.py:render_chat"

        with pytest.raises(ValidationError, match="only valid"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_rejects_obsolete_local_packing_option(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["packing"] = True

        with pytest.raises(ValidationError, match="packing"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_render_chat_rejects_blank_reference(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["render_chat"] = "  "

        with pytest.raises(ValidationError, match="must not be blank"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("drop_last", False, "drop_last=true"),
            ("filter_prompts", True, "does not support data.train.filter_prompts"),
        ],
    )
    def test_sft_requires_streaming_data_invariants(
        self,
        base_config: dict[str, Any],
        field: str,
        value: bool,
        message: str,
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"][field] = value

        with pytest.raises(ValidationError, match=message):
            NorthRLConfig.model_validate(base_config)

    def test_sft_local_files_accept_file_order_and_window_samples(
        self, base_config: dict[str, Any]
    ) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["shuffle"] = False
        base_config["data"]["train"]["window_samples"] = 64

        config = NorthRLConfig.model_validate(base_config)

        assert config.data.train.shuffle is False
        assert config.data.train.window_samples == 64

    def test_window_samples_is_sft_only(self, base_config: dict[str, Any]) -> None:
        base_config["data"]["train"]["window_samples"] = 64

        with pytest.raises(ValidationError, match="window_samples is only valid"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_supports_only_jsonl(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["data"]["train"]["files"] = ["train.parquet"]

        with pytest.raises(ValidationError, match="only JSONL"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "rollout",
        [
            None,
            {},
            {"num_workers": 1},
            {
                "num_workers": 8,
                "type": "simple",
                "sampling": {"temperature": 0.25},
                "invalid_field": True,
            },
        ],
    )
    def test_sft_warns_and_ignores_explicit_rollout(
        self, base_config: dict[str, Any], rollout: dict[str, Any] | None
    ) -> None:
        self._make_sft_config(base_config)
        base_config["rollout"] = rollout

        with pytest.warns(UserWarning, match="rollout is ignored for SFT"):
            cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.rollout.num_workers == 1
        assert cfg.rollout.type is None
        assert cfg.rollout.sampling.temperature == 1.0

    def test_sft_canonical_dump_round_trips(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)

        first = NorthRLConfig.model_validate(base_config)
        dumped = first.model_dump(by_alias=True)
        with pytest.warns(UserWarning, match="rollout is ignored for SFT"):
            second = NorthRLConfig.model_validate(dumped)

        assert dumped["rollout"]["type"] is None
        assert second.algorithm.type == "sft"
        assert second.rollout.type is None

    def test_sft_rollout_type_has_migration_error(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"]["type"] = "sft"

        with pytest.raises(
            ValidationError,
            match=r"rollout\.type='sft' is no longer supported.*algorithm\.type='sft'",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_non_sft_requires_rollout_type(self, base_config: dict[str, Any]) -> None:
        base_config["rollout"].pop("type")

        with pytest.raises(ValidationError, match="rollout.type is required unless"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_requires_no_sync(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["training"]["weight_sync"]["mode"] = "sync"

        with pytest.raises(
            ValidationError,
            match=r"algorithm\.type='sft'.*weight_sync\.mode='no-sync'",
        ):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
    @pytest.mark.parametrize("mode", ["no-sync", "fully-async"])
    def test_sft_rejects_sampling_pause(self, base_config, pause_mode, mode) -> None:
        self._make_sft_config(base_config)
        base_config["models"]["actor"]["lora"] = None
        NorthRLConfig.model_validate(base_config)
        base_config["training"]["weight_sync"].update(mode=mode, pause_mode=pause_mode)

        with pytest.raises(ValidationError, match="pause_mode requires fully-async RL"):
            NorthRLConfig.model_validate(base_config)

    def test_sft_requires_validation_disabled(self, base_config: dict[str, Any]) -> None:
        self._make_sft_config(base_config)
        base_config["validate"] = {
            "enabled": True,
            "data": {
                "datasets": {
                    "validation": {
                        "files": ["validation.parquet"],
                        "batch_size": 1,
                    }
                }
            },
        }

        with pytest.raises(
            ValidationError,
            match=r"algorithm\.type='sft'.*validate\.enabled=false",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_validation_enabled_requires_data(self, base_config: dict[str, Any]) -> None:
        base_config["validate"] = {"enabled": True}

        with pytest.raises(
            ValidationError,
            match=r"validate\.data is required when validate\.enabled=true",
        ):
            NorthRLConfig.model_validate(base_config)


class TestCrossBlockValidators:
    def test_managed_sources_are_sft_only(self, base_config: dict[str, Any]) -> None:
        base_config["data"]["train"].pop("files")
        base_config["data"]["train"]["total_samples"] = 1
        base_config["data"]["train"]["sources"] = [
            {
                "type": "managed",
                "dataset": "hq-math",
                "version": "2026-08",
                "ratio": 1.0,
            }
        ]

        with pytest.raises(ValidationError, match="supported only when algorithm.type='sft'"):
            NorthRLConfig.model_validate(base_config)

    def test_non_sft_requires_max_prompt_length(self, base_config: dict[str, Any]) -> None:
        base_config["data"].pop("max_prompt_length")

        with pytest.raises(
            ValidationError,
            match=r"data\.max_prompt_length is required unless algorithm\.type='sft'",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_kl_enabled_without_reference_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["kl"] = {"mode": "in_loss"}
        with pytest.raises(ValidationError, match=r"models\.reference 必须存在"):
            NorthRLConfig.model_validate(base_config)

    def test_reference_without_kl_rejected(self, base_config: dict[str, Any]) -> None:
        # Reference configured but kl not enabled — "配了不用" pattern
        base_config["models"]["reference"] = {"name": None, "checkpoint": None}
        with pytest.raises(ValidationError, match=r"配了不用"):
            NorthRLConfig.model_validate(base_config)

    def test_opd_without_teachers_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["type"] = "opd"
        with pytest.raises(ValidationError, match=r"models\.teachers 必须非空"):
            NorthRLConfig.model_validate(base_config)

    def test_teachers_without_opd_rejected(self, base_config: dict[str, Any]) -> None:
        # Teachers configured but algorithm.type is not opd — "配了不用" pattern
        base_config["models"]["teachers"] = [{"name": "Qwen/Qwen3-32B"}]
        with pytest.raises(ValidationError, match=r"配了不用"):
            NorthRLConfig.model_validate(base_config)

    def test_runtime_launch_mode_field_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["runtime"]["launch_mode"] = "local"
        with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
            NorthRLConfig.model_validate(base_config)

    def test_runtime_launcher_block_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["runtime"]["launcher"] = {"setup_script": "env.sh"}
        with pytest.raises(
            ValidationError,
            match=r"runtime\.launcher is no longer supported; source environment scripts",
        ):
            NorthRLConfig.model_validate(base_config)

    def test_lora_disabled_with_rank_rejected(self, base_config: dict[str, Any]) -> None:
        # enabled=false but rank still set → "配置藏雷"
        base_config["models"]["actor"]["lora"]["enabled"] = False
        # rank stays 32
        with pytest.raises(ValidationError, match=r"rank 必须不写或 null"):
            NorthRLConfig.model_validate(base_config)

    def test_lora_enabled_without_rank_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["models"]["actor"]["lora"]["rank"] = None
        with pytest.raises(ValidationError, match=r"rank 必填"):
            NorthRLConfig.model_validate(base_config)

    def test_max_tokens_above_upper_bound_rejected(self, base_config: dict[str, Any]) -> None:
        # max_seq=12288, max_prompt=4096, so cap is 8192
        base_config["rollout"]["sampling"]["max_tokens"] = 20000
        with pytest.raises(ValidationError, match=r"不能超过.*8192"):
            NorthRLConfig.model_validate(base_config)

    def test_max_tokens_zero_max_gen_rejected(self, base_config: dict[str, Any]) -> None:
        # max_prompt_length pushed to == max_sequence_length → derived gen space = 0
        base_config["data"]["max_prompt_length"] = base_config["data"]["max_sequence_length"]
        with pytest.raises(ValidationError, match=r"必须 ≥ 1"):
            NorthRLConfig.model_validate(base_config)

    def test_custom_loss_with_nonzero_entropy_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["loss"] = {"mode": "custom", "params": {"name": "direct_logprob"}}
        # entropy_coeff stays 1.0e-4 (nonzero)
        with pytest.raises(ValidationError, match=r"entropy_coeff 必须为 0"):
            NorthRLConfig.model_validate(base_config)

    def test_old_custom_loss_block_is_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["custom_loss"] = {"name": "unused", "params": {"scale": 2}}
        with pytest.raises(ValidationError) as error:
            NorthRLConfig.model_validate(base_config)
        assert any(
            item["loc"] == ("algorithm", "custom_loss") and item["type"] == "extra_forbidden"
            for item in error.value.errors()
        )

    def test_router_replay_enabled_minimal_block(self, base_config: dict[str, Any]) -> None:
        # Routing shape / fail-fast live in weaver-trainer now, so an enabled
        # block only needs enabled/mode; NorthRL carries no num_layers/topk.
        # R3 uses trainer-recomputed logprobs with rollout routing replayed in
        # both the denominator and training forwards.
        base_config["algorithm"]["sampling_logprob_source"] = "recomputed"
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R3"}
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.algorithm.router_replay.enabled is True
        assert cfg.algorithm.router_replay.mode == "R3"

        flags = get_derived_sampling_flags(OmegaConf.create(base_config))
        assert flags["return_moe_topk_indices"] is True
        assert flags["return_old_logprob"] is False

    def test_router_replay_r2_does_not_request_rollout_routing(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R2"}

        flags = get_derived_sampling_flags(OmegaConf.create(base_config))

        assert flags["return_moe_topk_indices"] is False

    @pytest.mark.parametrize("removed_field", ["num_layers", "topk", "fail_fast"])
    def test_router_replay_rejects_removed_shape_fields(
        self, base_config: dict[str, Any], removed_field: str
    ) -> None:
        # These moved to weaver-trainer; the strict schema now rejects them so a
        # stale recipe fails loudly instead of silently doing nothing.
        base_config["algorithm"]["router_replay"] = {
            "enabled": True,
            "mode": "R3",
            removed_field: 8,
        }
        with pytest.raises(ValidationError, match=removed_field):
            NorthRLConfig.model_validate(base_config)

    def test_router_replay_r2_requires_recomputed_logprobs(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = "inference"
        base_config["algorithm"]["router_replay"] = {
            "enabled": True,
            "mode": "R2",
        }

        with pytest.raises(ValidationError, match=r"sampling_logprob_source='recomputed'"):
            NorthRLConfig.model_validate(base_config)

    def test_router_replay_r3_requires_recomputed_logprobs(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = "inference"
        base_config["algorithm"]["router_replay"] = {
            "enabled": True,
            "mode": "R3",
        }

        with pytest.raises(ValidationError, match=r"sampling_logprob_source='recomputed'"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("algorithm_type", ["opd", "sft"])
    def test_router_replay_requires_grpo(
        self, base_config: dict[str, Any], algorithm_type: str
    ) -> None:
        base_config["algorithm"]["type"] = algorithm_type
        if algorithm_type == "sft":
            base_config["algorithm"]["loss"]["mode"] = "cross_entropy"
            base_config.pop("rollout")
            base_config["data"]["train"]["files"] = ["train.jsonl"]
            base_config["data"]["train"].pop("rollout_repeat_n")
            base_config["validate"]["enabled"] = False
            base_config["training"]["weight_sync"]["mode"] = "no-sync"
        elif algorithm_type == "opd":
            base_config["models"]["teachers"] = [{"name": "teacher"}]
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R2"}

        with pytest.raises(ValidationError, match=r"requires algorithm\.type='grpo'"):
            NorthRLConfig.model_validate(base_config)

    def test_router_replay_requires_weaver_backend(self, base_config: dict[str, Any]) -> None:
        base_config["backend"]["type"] = "mock"
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R2"}

        with pytest.raises(ValidationError, match=r"requires backend\.type='weaver'"):
            NorthRLConfig.model_validate(base_config)

    def test_router_replay_r3_requires_ref_propagating_rollout(
        self, base_config: dict[str, Any]
    ) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = "recomputed"
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R3"}
        base_config["rollout"]["type"] = "mock"

        with pytest.raises(ValidationError, match=r"propagates moe_topk_indices_ref"):
            NorthRLConfig.model_validate(base_config)

    def test_router_replay_r3_rejects_custom_rollout(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = "recomputed"
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": "R3"}
        base_config["rollout"]["custom_module_path"] = "custom.py"

        with pytest.raises(ValidationError, match=r"propagates moe_topk_indices_ref"):
            NorthRLConfig.model_validate(base_config)

    def test_filter_prompts_without_tokenizer_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["data"]["train"]["filter_prompts"] = True
        # actor.tokenizer stays null
        with pytest.raises(ValidationError, match=r"models\.actor\.tokenizer 必填"):
            NorthRLConfig.model_validate(base_config)


# ============================================================
# Strictness: extra=forbid catches typos
# ============================================================


class TestStrictness:
    def test_top_level_typo_rejected(self, base_config: dict[str, Any]) -> None:
        # Pretend user mistyped `experiment` as `expriment`
        base_config["expriment"] = base_config.pop("experiment")
        with pytest.raises(ValidationError):
            NorthRLConfig.model_validate(base_config)

    def test_nested_typo_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["experiment"]["naem"] = base_config["experiment"].pop("name")
        with pytest.raises(ValidationError):
            NorthRLConfig.model_validate(base_config)

    def test_empty_datasets_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["validate"]["data"]["datasets"] = {}
        with pytest.raises(ValidationError):
            NorthRLConfig.model_validate(base_config)


# ============================================================
# Sanity: feishu.enabled requires url
# ============================================================


class TestFeishu:
    def test_feishu_enabled_without_url_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["runtime"]["logger"]["feishu"] = {"enabled": True, "url": ""}
        with pytest.raises(ValidationError, match=r"feishu\.enabled=true 时 url 必填"):
            NorthRLConfig.model_validate(base_config)

    def test_feishu_disabled_with_empty_url_allowed(self, base_config: dict[str, Any]) -> None:
        # feishu off + url empty/null is the default state — must validate clean
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.runtime.logger.feishu.enabled is False


class TestPromptMeanConfig:
    @staticmethod
    def _enable(base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["loss"]["reduction"] = {"mode": "prompt_mean"}
        base_config["algorithm"]["entropy_coeff"] = 0.0
        base_config["algorithm"]["optim"].update(batch_size=None, mini_batch_size=16)

    @pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
    @pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
    @pytest.mark.parametrize("explicit_null", [False, True])
    def test_accepts_complete_group_optimizer_batches(
        self, base_config: dict[str, Any], loss_fn: str, mode: str, explicit_null: bool
    ) -> None:
        self._enable(base_config)
        base_config["algorithm"]["loss"]["mode"] = loss_fn
        base_config["training"]["weight_sync"]["mode"] = mode
        if not explicit_null:
            base_config["algorithm"]["optim"].pop("batch_size")

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.loss.reduction.mode == "prompt_mean"
        assert cfg.algorithm.optim.batch_size is None
        assert cfg.algorithm.optim.mini_batch_size == 16

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("type", "opd", "requires algorithm.type='grpo'"),
            ("type", "sft", "requires algorithm.type='grpo'"),
            (
                "loss.mode",
                "truncated_importance_sampling",
                "native importance_sampling or ppo_clip",
            ),
            ("entropy_coeff", 1.0e-4, "requires algorithm.entropy_coeff=0"),
            ("kl", {"mode": "in_loss"}, "requires algorithm.kl.mode=null"),
            ("loss.mode", "custom", "does not support algorithm.loss.mode='custom'"),
            (
                "optim",
                {"lr": 1.0e-6, "batch_size": 64},
                "requires algorithm.optim.batch_size=null",
            ),
        ],
    )
    def test_rejects_unsupported_combinations(
        self, base_config: dict[str, Any], field: str, value: Any, message: str
    ) -> None:
        self._enable(base_config)
        _set_algorithm_option(base_config["algorithm"], field, value)

        with pytest.raises(ValidationError, match=re.escape(message)):
            NorthRLConfig.model_validate(base_config)

    def test_supports_mis_with_rejection(self, base_config: dict[str, Any]) -> None:
        self._enable(base_config)
        base_config["algorithm"]["mis"] = {"enabled": True, "use_rs": True}

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.mis.use_rs is True

    def test_no_sync_remains_unsupported(self, base_config: dict[str, Any]) -> None:
        self._enable(base_config)
        base_config["training"]["weight_sync"]["mode"] = "no-sync"

        with pytest.raises(ValidationError, match=r"algorithm\.optim\.batch_size.*sync"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("mode", ["token_mean", "datum_mean"])
    def test_existing_reduction_configs_are_not_restricted(
        self, base_config: dict[str, Any], mode: str
    ) -> None:
        base_config["algorithm"]["loss"]["reduction"]["mode"] = mode
        base_config["algorithm"]["optim"]["batch_size"] = 64

        cfg = NorthRLConfig.model_validate(base_config)

        assert cfg.algorithm.loss.reduction.mode == mode
        assert cfg.algorithm.entropy_coeff != 0

    def test_unknown_reduction_is_rejected(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["loss"]["reduction"]["mode"] = "custom_reduction"
        with pytest.raises(ValidationError, match=r"loss.reduction.mode"):
            NorthRLConfig.model_validate(base_config)


class TestSamplingMaskConfig:
    @pytest.mark.parametrize(
        "keep_sampling_mask", [None, False, True], ids=["default", "off", "on"]
    )
    @pytest.mark.parametrize(
        "sampling",
        [{}, {"top_p": 0.95}, {"top_k": 20}, {"min_p": 0.1}],
        ids=["unfiltered", "top-p", "top-k", "min-p"],
    )
    def test_sampling_mask_requires_explicit_opt_in(
        self,
        base_config: dict[str, Any],
        keep_sampling_mask: bool | None,
        sampling: dict[str, float],
    ) -> None:
        base_config["rollout"]["sampling"].update(sampling)
        if keep_sampling_mask is not None:
            base_config["rollout"]["keep_sampling_mask"] = keep_sampling_mask

        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))
        expected = keep_sampling_mask is True

        assert validated.rollout.keep_sampling_mask is expected
        assert get_derived_sampling_flags(downstream) == {
            "return_sampling_mask": expected,
            "return_old_logprob": False,
            "return_moe_topk_indices": False,
        }

    def test_sampling_mask_preserves_inference_logprobs(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = "inference"
        base_config["rollout"]["keep_sampling_mask"] = True

        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))

        assert get_derived_sampling_flags(downstream) == {
            "return_sampling_mask": True,
            "return_old_logprob": True,
            "return_moe_topk_indices": False,
        }


class TestMISConfig:
    def test_defaults_preserve_existing_recipes(self, base_config: dict[str, Any]) -> None:
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.algorithm.mis == MISConfig()
        assert cfg.algorithm.mis.enabled is False
        assert cfg.algorithm.mis.get_mismatch_metrics is False
        assert cfg.algorithm.mis.tis_mode == "truncate"
        assert cfg.algorithm.mis.tis_upper_bound == 2.0
        assert cfg.algorithm.mis.use_rs is False

    @pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
    def test_native_grpo_recomputed_loss_supported(
        self, base_config: dict[str, Any], loss_fn: str
    ) -> None:
        base_config["algorithm"]["mis"] = {"enabled": True}
        base_config["algorithm"]["loss"]["mode"] = loss_fn
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.algorithm.mis.enabled is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [("top_p", 0.9), ("top_p", 1.1), ("top_k", 1), ("min_p", 0.1), ("min_p", -0.1)],
    )
    @pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
    def test_enabled_rejects_sampling_filters(
        self, base_config: dict[str, Any], feature: str, field: str, value: float
    ) -> None:
        base_config["algorithm"]["mis"] = {feature: True}
        base_config["rollout"]["sampling"][field] = value
        with pytest.raises(ValidationError, match="recompute and training sampling_mask"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
    def test_enabled_rejects_sampling_mask_retention(
        self, base_config: dict[str, Any], feature: str
    ) -> None:
        base_config["algorithm"]["mis"] = {feature: True}
        base_config["rollout"]["keep_sampling_mask"] = True
        with pytest.raises(ValidationError, match="keep_sampling_mask"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("top_k", [-1, 0])
    @pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
    def test_enabled_allows_verified_sampling_convention(
        self, base_config: dict[str, Any], feature: str, top_k: int
    ) -> None:
        base_config["algorithm"]["mis"] = {feature: True}
        base_config["rollout"]["sampling"].update(
            top_p=1.0, top_k=top_k, min_p=0.0, temperature=1.0, repetition_penalty=1.0
        )
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.rollout.sampling.temperature == 1.0
        assert cfg.rollout.sampling.repetition_penalty == 1.0

    @pytest.mark.parametrize("field", ["temperature", "repetition_penalty"])
    @pytest.mark.parametrize("value", [0.0, 0.7, 1.2])
    @pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
    def test_enabled_rejects_unverified_sampling_transforms(
        self, base_config: dict[str, Any], feature: str, field: str, value: float
    ) -> None:
        base_config["algorithm"]["mis"] = {feature: True}
        base_config["rollout"]["sampling"][field] = value
        with pytest.raises(ValidationError, match="sampler old_logprobs semantics"):
            NorthRLConfig.model_validate(base_config)

    def test_disabled_keeps_existing_sampling_filters(self, base_config: dict[str, Any]) -> None:
        base_config["algorithm"]["mis"] = {"enabled": False}
        base_config["rollout"]["sampling"].update(top_p=0.9, top_k=32, min_p=0.1)
        assert NorthRLConfig.model_validate(base_config).algorithm.mis.enabled is False

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("type", "opd", "requires algorithm.type='grpo'"),
            ("type", "sft", "requires algorithm.type='grpo'"),
            ("sampling_logprob_source", "inference", "sampling_logprob_source='recomputed'"),
            (
                "loss.mode",
                "truncated_importance_sampling",
                "native importance_sampling or ppo_clip",
            ),
            ("loss.mode", "custom", "incompatible with algorithm.loss.mode"),
        ],
    )
    def test_enabled_rejects_incompatible_training_contract(
        self, base_config: dict[str, Any], field: str, value: Any, error: str
    ) -> None:
        algorithm = base_config["algorithm"]
        algorithm["mis"] = {"enabled": True}
        _set_algorithm_option(algorithm, field, value)
        with pytest.raises(ValidationError, match=re.escape(error)):
            AlgorithmConfig.model_validate(algorithm)

    @pytest.mark.parametrize("source", ["inference", "recomputed"])
    @pytest.mark.parametrize(
        ("loss_fn", "custom_loss"),
        [
            ("importance_sampling", False),
            ("ppo_clip", False),
            ("truncated_importance_sampling", False),
            ("custom_loss_name", True),
        ],
    )
    def test_metrics_only_keeps_denominator_source_and_loss_choices(
        self, base_config: dict[str, Any], source: str, loss_fn: str, custom_loss: bool
    ) -> None:
        base_config["algorithm"].update(
            mis={"enabled": False, "get_mismatch_metrics": True},
            sampling_logprob_source=source,
            loss=(
                {"mode": "custom", "params": {"name": loss_fn, "base_loss_fn": loss_fn}}
                if custom_loss
                else {"mode": loss_fn}
            ),
            entropy_coeff=0.0,
        )
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.algorithm.mis.enabled is False
        assert cfg.algorithm.mis.get_mismatch_metrics is True
        assert cfg.algorithm.sampling_logprob_source == source
        assert cfg.algorithm.loss.mode == ("custom" if custom_loss else loss_fn)
        if custom_loss:
            assert cfg.algorithm.loss.params["name"] == loss_fn

    @pytest.mark.parametrize("algorithm_type", ["opd", "sft"])
    def test_metrics_only_requires_grpo(
        self, base_config: dict[str, Any], algorithm_type: str
    ) -> None:
        algorithm = base_config["algorithm"]
        algorithm.update(type=algorithm_type, mis={"get_mismatch_metrics": True})
        with pytest.raises(
            ValidationError, match="get_mismatch_metrics=true requires algorithm.type='grpo'"
        ):
            AlgorithmConfig.model_validate(algorithm)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sampling_logprob_source", "inference"),
            ("loss.mode", "truncated_importance_sampling"),
            ("loss.mode", "custom"),
        ],
    )
    def test_metrics_flag_does_not_relax_enabled_correction_contract(
        self, base_config: dict[str, Any], field: str, value: Any
    ) -> None:
        algorithm = base_config["algorithm"]
        algorithm.update(mis={"enabled": True, "get_mismatch_metrics": True})
        _set_algorithm_option(algorithm, field, value)
        with pytest.raises(ValidationError, match="algorithm.mis.enabled=true"):
            AlgorithmConfig.model_validate(algorithm)

    @pytest.mark.parametrize("algorithm_type", ["grpo", "opd", "sft"])
    def test_disabled_does_not_restrict_existing_algorithms(
        self, base_config: dict[str, Any], algorithm_type: str
    ) -> None:
        algorithm = base_config["algorithm"]
        algorithm.update(
            type=algorithm_type,
            loss={"mode": "custom", "params": {"name": "existing_custom_loss"}},
            sampling_logprob_source="inference",
            mis={"enabled": False},
        )
        assert AlgorithmConfig.model_validate(algorithm).mis.enabled is False

    @pytest.mark.parametrize(
        "field",
        [
            "tis_lower_bound",
            "tis_upper_bound",
            "rs_lower_bound",
            "rs_upper_bound",
            "rs_veto_threshold",
        ],
    )
    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
    def test_thresholds_must_be_positive_and_finite(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError, match=field):
            MISConfig.model_validate({field: value})

    def test_tis_upper_bound_cannot_be_null(self) -> None:
        with pytest.raises(ValidationError, match="tis_upper_bound"):
            MISConfig.model_validate({"tis_upper_bound": None})

    @pytest.mark.parametrize("mode", ["clip", "mask"])
    @pytest.mark.parametrize("bounds", [{"tis_upper_bound": 1.0}, {"tis_lower_bound": 3.0}])
    def test_two_sided_tis_rejects_empty_effective_interval(self, mode: str, bounds: dict) -> None:
        with pytest.raises(ValidationError, match="effective tis_lower_bound < tis_upper_bound"):
            MISConfig.model_validate({"enabled": True, "tis_mode": mode, **bounds})

    @pytest.mark.parametrize("mode", ["none", "truncate"])
    def test_inactive_lower_tis_bound_does_not_restrict_one_sided_modes(self, mode: str) -> None:
        cfg = MISConfig(enabled=True, tis_mode=mode, tis_lower_bound=3.0, tis_upper_bound=2.0)
        assert cfg.tis_lower_bound == 3.0

    @pytest.mark.parametrize(
        "bounds",
        [
            {"tis_upper_bound": 1.0},
            {"tis_lower_bound": 3.0},
            {"rs_lower_bound": 2.0},
            {"rs_upper_bound": 0.5},
            {"rs_lower_bound": 0.5, "rs_upper_bound": 0.5},
        ],
    )
    def test_rs_checks_inherited_and_explicit_bounds(self, bounds: dict) -> None:
        with pytest.raises(ValidationError, match="effective rs_lower_bound < rs_upper_bound"):
            MISConfig.model_validate({"enabled": True, "use_rs": True, **bounds})

    def test_pure_rs_uses_explicit_bounds_without_activating_tis(self) -> None:
        cfg = MISConfig(
            enabled=True,
            tis_mode="none",
            tis_lower_bound=3.0,
            tis_upper_bound=2.0,
            tis_level="geometric",
            tis_batch_normalize=True,
            use_rs=True,
            rs_lower_bound=0.2,
            rs_upper_bound=5.0,
        )
        assert cfg.use_rs is True

    @pytest.mark.parametrize("mode", ["truncate", "clip", "mask"])
    def test_geometric_weights_reject_batch_normalization(self, mode: str) -> None:
        with pytest.raises(ValidationError, match="incompatible with tis_level='geometric'"):
            MISConfig(enabled=True, tis_mode=mode, tis_level="geometric", tis_batch_normalize=True)

    @pytest.mark.parametrize("level", ["token", "sequence"])
    def test_other_weight_levels_allow_batch_normalization(self, level: str) -> None:
        assert MISConfig(
            enabled=True, tis_level=level, tis_batch_normalize=True
        ).tis_batch_normalize

    @pytest.mark.parametrize("get_mismatch_metrics", [False, True])
    def test_disabled_stages_do_not_apply_cross_field_constraints(
        self, get_mismatch_metrics: bool
    ) -> None:
        cfg = MISConfig(
            enabled=False,
            get_mismatch_metrics=get_mismatch_metrics,
            tis_mode="clip",
            tis_level="geometric",
            tis_batch_normalize=True,
            tis_lower_bound=5.0,
            use_rs=True,
        )
        assert cfg.enabled is False

    @pytest.mark.parametrize(
        ("source", "mis_block", "expected"),
        [
            ("recomputed", None, False),
            ("recomputed", {"enabled": False}, False),
            ("recomputed", {"enabled": True}, True),
            ("recomputed", {"get_mismatch_metrics": True}, True),
            ("recomputed", {"enabled": False, "get_mismatch_metrics": False}, False),
            ("recomputed", {"enabled": True, "get_mismatch_metrics": True}, True),
            ("inference", {"enabled": False, "get_mismatch_metrics": True}, True),
            ("inference", None, True),
            ("inference", {"enabled": False}, True),
        ],
    )
    def test_sampling_requests_rollout_old_logprob_for_correction_or_metrics(
        self, base_config: dict[str, Any], source: str, mis_block: dict | None, expected: bool
    ) -> None:
        base_config["algorithm"]["sampling_logprob_source"] = source
        if mis_block is not None:
            base_config["algorithm"]["mis"] = mis_block
        flags = get_derived_sampling_flags(OmegaConf.create(base_config))
        assert flags["return_old_logprob"] is expected
        assert flags["return_sampling_mask"] is False

    @pytest.mark.parametrize(("mode", "request_routes"), [("R2", False), ("R3", True)])
    @pytest.mark.parametrize("feature", ["enabled", "get_mismatch_metrics"])
    def test_mis_sampling_flags_preserve_router_replay_requirements(
        self, base_config: dict[str, Any], mode: str, request_routes: bool, feature: str
    ) -> None:
        base_config["algorithm"]["mis"] = {feature: True}
        base_config["algorithm"]["router_replay"] = {"enabled": True, "mode": mode}
        flags = get_derived_sampling_flags(OmegaConf.create(base_config))
        assert flags["return_old_logprob"] is True
        assert flags["return_moe_topk_indices"] is request_routes


class TestScoreCentering:
    @staticmethod
    def configure(cfg):
        cfg["algorithm"].update(
            loss={"mode": "score_centering"}, sampling_logprob_source="inference", entropy_coeff=0.0
        )
        cfg["rollout"].update({"type": "simple"})
        cfg["rollout"]["sampling"].update(
            temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, min_tokens=0
        )
        return cfg

    @pytest.mark.parametrize("keep_sampling_mask", [False, True])
    def test_valid_configuration_and_derived_flags(self, base_config, keep_sampling_mask):
        base_config["rollout"]["keep_sampling_mask"] = keep_sampling_mask
        cfg = NorthRLConfig.model_validate(self.configure(base_config))
        flags = get_derived_sampling_flags(cfg)
        assert flags["topk_output_logprobs"] == 128
        assert flags["return_old_logprob"] is False
        assert flags["return_sampling_mask"] is keep_sampling_mask

    @pytest.mark.parametrize(
        "key,value", [("temperature", 0.7), ("top_p", 0.9), ("top_k", 10), ("min_tokens", 1)]
    )
    def test_reject_modified_distribution(self, base_config, key, value):
        cfg = self.configure(base_config)
        cfg["rollout"]["sampling"][key] = value
        with pytest.raises(ValueError, match="unfiltered"):
            NorthRLConfig.model_validate(cfg)


class TestDebugDataDumpConfig:
    def test_defaults_preserve_existing_dump_behavior(self, base_config):
        config = NorthRLConfig.model_validate(base_config)
        assert config.runtime.debug.dump_every_n_steps == 1
        assert config.runtime.debug.remote_api_dump_options == {}

    def test_dump_options_survive_validation_and_reach_dumper(self, base_config, tmp_path):
        base_config["runtime"]["debug"] = {
            "enable_data_dump": True,
            "dump_dir": str(tmp_path),
            "dump_every_n_steps": 2,
            "remote_api_dump_options": {
                "datums": True,
                "prepared_trajectories": False,
                "training_metrics": True,
            },
        }
        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))
        dumper = DataDumper(downstream)
        assert dumper.should_dump("datums", 0)
        assert not dumper.should_dump("datums", 1)
        assert dumper.should_dump("datums", 2)
        assert dumper.should_dump("training_metrics", 2)
        assert not dumper.should_dump("prepared_trajectories", 2)

    @pytest.mark.parametrize("interval", [0, -1, 1.5, True, "2"])
    def test_rejects_invalid_dump_interval(self, base_config, interval):
        base_config["runtime"]["debug"] = {"dump_every_n_steps": interval}
        with pytest.raises(ValidationError, match="dump_every_n_steps"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("options", [{"datum": True}, {"datums": "invalid"}])
    def test_rejects_invalid_remote_dump_options(self, base_config, options):
        base_config["runtime"]["debug"] = {"remote_api_dump_options": options}
        with pytest.raises(ValidationError, match="remote_api_dump_options"):
            NorthRLConfig.model_validate(base_config)

    def test_swe_grpo_recipe_validates_with_datums_dump(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NORTHRL_PATH", str(REPO_ROOT))
        monkeypatch.setenv("NORTHRL_SWEBENCH_MODEL", "test-weaver-model")
        monkeypatch.setenv("NORTHRL_SWEBENCH_TOKENIZER", "Qwen/Qwen3.5-35B-A3B")
        monkeypatch.setenv("NORTHRL_SWEBENCH_TASK_FILES_ROOT", "/data/swe/evaluation")
        monkeypatch.setenv("NORTHRL_SWEBENCH_TRAIN_DATA", "/data/swe-training.parquet")
        monkeypatch.setenv("EXPERIMENT_PATH", str(tmp_path))
        monkeypatch.setenv("NORTHRL_SWEBENCH_TOTAL_STEPS", "1")
        monkeypatch.setenv("NORTHRL_SAMPLER_TTL_S", "604800")
        config = OmegaConf.load(REPO_ROOT / "recipe/swebench/e2b_claude_grpo.yaml")
        validated = NorthRLConfig.model_validate(OmegaConf.to_container(config, resolve=True))
        assert validated.runtime.debug.dump_every_n_steps == 1
        assert validated.runtime.debug.remote_api_dump_options == {"datums": True}
        assert validated.algorithm.optim.batch_size is None
        assert validated.algorithm.optim.mini_batch_size == 16


class TestAgenticTaskMixture:
    @staticmethod
    def _configure(base_config):
        base_config["data"]["train"].pop("files")
        base_config["data"]["train"]["sources"] = [
            {
                "type": "local",
                "name": "code",
                "files": ["code.jsonl"],
                "ratio": 1.0,
                "rollout_profile": "code",
            },
            {
                "type": "local",
                "name": "swe",
                "files": ["swe.jsonl"],
                "ratio": 2.0,
                "rollout_profile": "swe",
            },
        ]
        base_config["rollout"].update(
            {
                "type": "e2b",
                "execution": {
                    "timeout_seconds": 7200,
                    "env_error_retries": 2,
                    "profiles": {
                        "code": {
                            "entrypoint": {
                                "target": "/tasks/code.py",
                                "task_files_root": "/evaluation/code",
                            },
                            "input_fields": ["instruction", "cwd"],
                        },
                        "swe": {
                            "entrypoint": {"target": "/tasks/swe.py"},
                            "input_fields": ["instruction", "interface"],
                            "timeout_seconds": 3600,
                        },
                    },
                },
            }
        )

    def test_weighted_sources_and_profiles_round_trip(self, base_config):
        self._configure(base_config)
        cfg = NorthRLConfig.model_validate(base_config)
        assert cfg.data.train.total_samples is None
        assert cfg.data.train.sources[0].name == "code"
        assert cfg.data.train.sources[1].rollout_profile == "swe"
        profiles = cfg.rollout.execution["profiles"]
        assert profiles["code"]["entrypoint"]["task_files_root"] == "/evaluation/code"
        assert "task_files_root" not in profiles["swe"]["entrypoint"]

    def test_local_grpo_sources_without_execution_profiles(self, base_config):
        self._configure(base_config)
        for source in base_config["data"]["train"]["sources"]:
            source.pop("rollout_profile")
        base_config["rollout"].update({"type": "single_turn", "execution": {}})
        NorthRLConfig.model_validate(base_config)

    def test_row_profiles_need_no_source_default(self, base_config):
        self._configure(base_config)
        for source in base_config["data"]["train"]["sources"]:
            source.pop("rollout_profile")
        NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("name", None, "requires a name"),
            ("name", "", "nonblank"),
            ("name", " code ", "boundary whitespace"),
            ("name", "swe", "duplicate RL source name"),
            ("rollout_profile", "unknown", "unknown E2B rollout_profile"),
            ("ratio", 0, "greater than 0"),
            ("ratio", float("inf"), "finite"),
        ],
    )
    def test_invalid_sources_rejected(self, base_config, field, value, message):
        self._configure(base_config)
        base_config["data"]["train"]["sources"][0][field] = value
        with pytest.raises(ValidationError, match=message):
            NorthRLConfig.model_validate(base_config)

    def test_rl_rejects_ignored_sft_sample_budget(self, base_config):
        self._configure(base_config)
        base_config["data"]["train"]["total_samples"] = 100
        with pytest.raises(ValidationError, match="total_samples is only valid for SFT"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("profiles", [None, {}, [], "code"])
    def test_invalid_profile_registry(self, base_config, profiles):
        self._configure(base_config)
        base_config["rollout"]["execution"]["profiles"] = profiles
        with pytest.raises(ValidationError, match="nonempty mapping"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("field", ["entrypoint", "input_fields"])
    def test_profile_requires_own_fields(self, base_config, field):
        self._configure(base_config)
        del base_config["rollout"]["execution"]["profiles"]["code"][field]
        with pytest.raises(ValidationError, match=f"{field} is required"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("input_fields", ["instruction", "instruction"], "input_fields must be a list"),
            ("entrypoint", {"target": "relative.py"}, "absolute host path"),
            ("timeout_seconds", True, "positive integer"),
            ("timeout_seconds", 0, "positive integer"),
            ("env_error_retries", -1, "non-negative integer"),
            ("sampling", {}, "unsupported fields"),
        ],
    )
    def test_profile_options_are_validated(self, base_config, field, value, message):
        self._configure(base_config)
        base_config["rollout"]["execution"]["profiles"]["code"][field] = value
        with pytest.raises(ValidationError, match=message):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("entrypoint", {"target": "/tasks/global.py"}),
            ("input_fields", ["instruction"]),
        ],
    )
    def test_global_and_profile_routes_are_mutually_exclusive(self, base_config, field, value):
        self._configure(base_config)
        base_config["rollout"]["execution"][field] = value
        with pytest.raises(ValidationError, match="cannot be combined"):
            NorthRLConfig.model_validate(base_config)

    def test_validated_mixture_reaches_loader_and_worker(self, base_config, tmp_path, monkeypatch):
        import json
        from unittest.mock import AsyncMock, Mock

        from northrl.controller import NorthRLController
        from northrl.data_loader import MixtureDataLoader
        from northrl.northrl_types import NorthRLRole
        from northrl.remote_rollout.contracts import RemoteRolloutResult, RemoteSegment
        from northrl.rollout_worker import E2BRolloutWorker
        from northrl.rollout_worker import e2b_rollout_worker as worker_module
        from northrl.utils.async_utils import run_coro

        self._configure(base_config)
        base_config["data"]["train"].update({"batch_size": 8, "rollout_repeat_n": 2})
        for source in base_config["data"]["train"]["sources"]:
            path = tmp_path / f'{source["name"]}.jsonl'
            path.write_text(
                json.dumps(
                    {
                        "instruction": "Fix it",
                        "task_id": "same-task",
                        "template": source["name"],
                        "cwd": "/workspace",
                        "interface": "public interface",
                    }
                )
                + "\n"
            )
            source["files"] = [str(path)]
        validated = NorthRLConfig.model_validate(base_config)
        config = OmegaConf.create(validated.model_dump(by_alias=True))
        controller = object.__new__(NorthRLController)
        controller._config = config
        loader_class = controller._get_module_class_for_role(NorthRLRole.DATA_LOADER)
        assert loader_class is MixtureDataLoader
        loader = loader_class(controller._derive_train_dataloader_config())
        worker = E2BRolloutWorker(config)
        holder = Mock()
        holder.get_current_gateway_snapshot = AsyncMock(
            return_value={
                "base_url": "https://gateway.example",
                "model": "actor",
                "sampling_session_id": "session",
                "weight_version": "v1",
            }
        )
        worker.init_inference_service_client(holder)
        executor = AsyncMock(
            side_effect=lambda request, *_args, **_kwargs: RemoteRolloutResult(
                rollout_id=request.rollout_id,
                reward=0.0,
                segments=[
                    RemoteSegment(
                        tokens=[1, 2],
                        loss_mask=[0, 1],
                        log_probs=[0.0, -0.1],
                        old_log_probs=[0.0, -0.1],
                    )
                ],
            )
        )
        monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
        worker._put_trajectory = Mock(return_value="success")
        try:
            requests = [loader.get_next_item() for _ in range(16)]
            assert {item["data_source"] for item in requests} == {"code", "swe"}
            for item in requests:
                assert run_coro(worker.rollout(item)) == "success"
            for pair_start in range(0, 16, 2):
                pair = requests[pair_start : pair_start + 2]
                assert pair[0]["group_id"] == pair[1]["group_id"]
                assert pair[0]["rollout_profile"] == pair[1]["rollout_profile"]
            assert len({item["group_id"] for item in requests}) == 8
            for item, call in zip(requests, executor.await_args_list):
                request, effective = call.args
                assert request.metadata["task_id"] == "same-task"
                assert (
                    effective.rollout.execution.entrypoint.target
                    == f'/tasks/{item["rollout_profile"]}.py'
                )
                assert set(request.task) == set(effective.rollout.execution.input_fields)
        finally:
            loader.close()


class TestE2BExecutionBudgets:
    @staticmethod
    def _configure(base_config, location):
        if location == "single":
            TestE2BEntrypointConfig._set_e2b_rollout(base_config, {"target": "/tasks/agent.py"})
        else:
            TestAgenticTaskMixture._configure(base_config)
        execution = base_config["rollout"]["execution"]
        if location == "profile":
            return execution["profiles"]["code"], "rollout.execution.profiles.code"
        return execution, "rollout.execution"

    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize(
        "key",
        [
            "preparation_timeout_seconds",
            "agent_timeout_seconds",
            "agent_finalize_timeout_seconds",
            "evaluation_timeout_seconds",
        ],
    )
    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "600", None])
    def test_stage_budget_requires_positive_integer(self, base_config, location, key, value):
        options, _ = self._configure(base_config, location)
        options[key] = value

        with pytest.raises(ValidationError, match=f"{key} must be a positive integer"):
            NorthRLConfig.model_validate(base_config)

    def test_single_task_stage_budgets_survive_complete_schema_validation(self, base_config):
        from northrl.remote_rollout.execution_budgets import ExecutionBudgets

        options, _ = self._configure(base_config, "single")
        options.update(
            preparation_timeout_seconds=100,
            agent_timeout_seconds=900,
            agent_finalize_timeout_seconds=200,
            evaluation_timeout_seconds=1800,
        )
        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))
        budgets = ExecutionBudgets.from_config(downstream)

        assert budgets.preparation_seconds == 100
        assert budgets.agent_seconds == 900
        assert budgets.finalize_seconds == 200
        assert budgets.evaluation_seconds == 1800
        assert budgets.recovery_seconds == 900
        assert budgets.io_seconds == 60

    def test_profile_budgets_validate_merge_and_keep_agent_precedence(self, base_config):
        from northrl.remote_rollout.execution_budgets import ExecutionBudgets
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs

        execution, _ = self._configure(base_config, "common")
        execution.update(
            preparation_timeout_seconds=50,
            agent_timeout_seconds=600,
            agent_finalize_timeout_seconds=60,
            evaluation_timeout_seconds=80,
        )
        execution["profiles"]["code"].update(
            preparation_timeout_seconds=70,
            agent_timeout_seconds=900,
            agent_finalize_timeout_seconds=90,
            evaluation_timeout_seconds=100,
        )
        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))
        effective = get_e2b_execution_configs(downstream)
        inherited = ExecutionBudgets.from_config(effective["swe"])
        overridden = ExecutionBudgets.from_config(effective["code"])

        # The explicit common agent budget still wins over a profile's legacy alias.
        assert effective["swe"].rollout.execution.timeout_seconds == 3600
        assert (
            inherited.preparation_seconds,
            inherited.agent_seconds,
            inherited.finalize_seconds,
            inherited.evaluation_seconds,
        ) == (50, 600, 60, 80)
        assert (
            overridden.preparation_seconds,
            overridden.agent_seconds,
            overridden.finalize_seconds,
            overridden.evaluation_seconds,
        ) == (70, 900, 90, 100)
        assert inherited.recovery_seconds == overridden.recovery_seconds == 900
        assert inherited.io_seconds == overridden.io_seconds == 60


class TestE2BNetworkPolicy:
    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize(
        "policy, expected",
        [
            (
                {
                    "deny_domains": ["*.GitHub.COM.", "github.com"],
                    "task_overrides": {"special": {"deny_domains": []}},
                },
                {
                    "deny_domains": ["github.com"],
                    "task_overrides": {"special": {"deny_domains": []}},
                },
            ),
            (
                {"mode": "local_only", "task_overrides": {"open": {"deny_domains": []}}},
                {
                    "mode": "local_only",
                    "deny_domains": [],
                    "task_overrides": {"open": {"deny_domains": []}},
                },
            ),
            (
                {"task_overrides": {"isolated": {"mode": "local_only"}}},
                {
                    "deny_domains": [],
                    "task_overrides": {"isolated": {"mode": "local_only", "deny_domains": []}},
                },
            ),
        ],
    )
    def test_network_policy_is_normalized_in_each_execution_location(
        self, base_config, location, policy, expected
    ):
        options, _ = TestE2BExecutionBudgets._configure(base_config, location)
        options["network_policy"] = policy
        validated = NorthRLConfig.model_validate(base_config)
        execution = validated.rollout.execution
        if location == "profile":
            execution = execution["profiles"]["code"]
        assert execution["network_policy"] == expected

    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize(
        "policy",
        [None, {"deny_domains": ["https://github.com"]}, {"task_overrides": {"special": {}}}],
    )
    def test_invalid_policy_identifies_its_location(self, base_config, location, policy):
        options, expected_location = TestE2BExecutionBudgets._configure(base_config, location)
        options["network_policy"] = policy
        with pytest.raises(ValidationError, match=re.escape(f"{expected_location}.network_policy")):
            NorthRLConfig.model_validate(base_config)

    def test_profile_override_cannot_hide_invalid_common_policy(self, base_config):
        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["network_policy"] = {"deny_domain": ["github.com"]}
        for profile in execution["profiles"].values():
            profile["network_policy"] = {"deny_domains": []}
        with pytest.raises(ValidationError, match=r"rollout\.execution\.network_policy"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("validate_schema", [True, False])
    @pytest.mark.parametrize(
        "profile_policy",
        [
            {},
            {"deny_domains": []},
            {"deny_domains": ["gitlab.com"]},
            {"task_overrides": {"special": {"deny_domains": ["bitbucket.org"]}}},
        ],
    )
    def test_profile_replaces_entire_policy_without_leaking_other_task_rules(
        self, base_config, profile_policy, validate_schema
    ):
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs
        from northrl.remote_rollout.network_policy_config import resolve_network_policy

        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["network_policy"] = {
            "deny_domains": ["github.com"],
            "task_overrides": {"shared-task": {"deny_domains": ["example.org"]}},
        }
        execution["profiles"]["code"]["network_policy"] = profile_policy
        original = copy.deepcopy(base_config)
        downstream = OmegaConf.create(
            NorthRLConfig.model_validate(base_config).model_dump(by_alias=True)
            if validate_schema
            else base_config
        )
        original_downstream = OmegaConf.to_container(downstream, resolve=False)
        effective = get_e2b_execution_configs(downstream)
        inherited = OmegaConf.to_container(effective["swe"].rollout.execution.network_policy)
        replaced = OmegaConf.to_container(effective["code"].rollout.execution.network_policy)
        assert resolve_network_policy(inherited, "shared-task") == {"deny_domains": ["example.org"]}
        assert resolve_network_policy(replaced, "shared-task") == {
            "deny_domains": profile_policy.get("deny_domains", [])
        }
        assert resolve_network_policy(replaced, "special") == resolve_network_policy(
            profile_policy, "special"
        )
        effective["swe"].rollout.execution.network_policy.deny_domains.append("new.example")
        assert "new.example" not in repr(replaced)
        assert OmegaConf.to_container(downstream, resolve=False) == original_downstream
        assert base_config == original


class TestE2BResultProcessing:
    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize("mode", ["thread", "process"])
    def test_result_processing_validates_in_each_execution_location(
        self, base_config, location, mode
    ):
        execution, _ = TestE2BExecutionBudgets._configure(base_config, location)
        execution["result_processing"] = mode
        config = NorthRLConfig.model_validate(base_config)
        options = config.rollout.execution
        if location == "profile":
            options = options["profiles"]["code"]
        assert options["result_processing"] == mode

    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize("mode", [None, True, 1, "fork", "", []])
    def test_invalid_processing_mode_is_rejected(self, base_config, location, mode):
        execution, _ = TestE2BExecutionBudgets._configure(base_config, location)
        execution["result_processing"] = mode
        with pytest.raises(ValidationError, match="result_processing"):
            NorthRLConfig.model_validate(base_config)

    def test_profile_mode_overrides_without_mutating_other_profiles(self, base_config):
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs

        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["result_processing"] = "process"
        execution["profiles"]["code"]["result_processing"] = "thread"
        config = NorthRLConfig.model_validate(base_config)
        effective = get_e2b_execution_configs(OmegaConf.create(config.model_dump(by_alias=True)))
        assert effective["code"].rollout.execution.result_processing == "thread"
        assert effective["swe"].rollout.execution.result_processing == "process"


class TestE2BTokenPrefixReuse:
    @pytest.mark.parametrize("location", ["single", "common"])
    def test_defaults_to_false_in_effective_execution_configs(self, base_config, location):
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs

        TestE2BExecutionBudgets._configure(base_config, location)
        original = copy.deepcopy(base_config)
        config = NorthRLConfig.model_validate(base_config)
        assert config.rollout.execution["token_prefix_reuse"] is False
        effective = get_e2b_execution_configs(OmegaConf.create(config.model_dump(by_alias=True)))
        assert all(
            item.rollout.execution.token_prefix_reuse is False for item in effective.values()
        )
        assert base_config == original

    @pytest.mark.parametrize("enabled", [False, True])
    def test_explicit_boolean_survives_validation(self, base_config, enabled):
        options, _ = TestE2BExecutionBudgets._configure(base_config, "single")
        options["token_prefix_reuse"] = enabled
        config = NorthRLConfig.model_validate(base_config)
        execution = config.rollout.execution
        assert execution["token_prefix_reuse"] is enabled

    @pytest.mark.parametrize(
        "location,value", [("single", "false"), ("single", 1), ("profile", None)]
    )
    def test_non_boolean_is_rejected(self, base_config, location, value):
        options, path = TestE2BExecutionBudgets._configure(base_config, location)
        options["token_prefix_reuse"] = value
        with pytest.raises(
            ValidationError, match=re.escape(f"{path}.token_prefix_reuse must be a boolean")
        ):
            NorthRLConfig.model_validate(base_config)

    def test_invalid_common_value_is_rejected_even_when_all_profiles_override(self, base_config):
        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["token_prefix_reuse"] = "false"
        for profile in execution["profiles"].values():
            profile["token_prefix_reuse"] = False
        with pytest.raises(
            ValidationError, match=r"rollout\.execution\.token_prefix_reuse must be a boolean"
        ):
            NorthRLConfig.model_validate(base_config)

    def test_profile_override_is_preserved_without_changing_other_profiles(self, base_config):
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs

        enabled = True
        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["token_prefix_reuse"] = enabled
        execution["profiles"]["code"]["token_prefix_reuse"] = not enabled
        original = copy.deepcopy(base_config)
        config = NorthRLConfig.model_validate(base_config)
        effective = get_e2b_execution_configs(OmegaConf.create(config.model_dump(by_alias=True)))
        assert effective["code"].rollout.execution.token_prefix_reuse is not enabled
        assert effective["swe"].rollout.execution.token_prefix_reuse is enabled
        assert base_config == original


class TestE2BVolumeMounts:
    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize("mounts", [None, {}, {"/mnt/shared": "shared-data"}])
    def test_mounts_pass_schema_in_all_execution_locations(self, base_config, location, mounts):
        options, _ = TestE2BExecutionBudgets._configure(base_config, location)
        options["volume_mounts"] = mounts

        config = NorthRLConfig.model_validate(base_config)
        execution = config.rollout.execution
        if location == "profile":
            execution = execution["profiles"]["code"]
        assert execution["volume_mounts"] == mounts

    @pytest.mark.parametrize("location", ["single", "common", "profile"])
    @pytest.mark.parametrize("mounts", [[], {"relative": "volume"}, {"/mnt/shared": 123}])
    def test_bad_mounts_fail_schema_at_their_location(self, base_config, location, mounts):
        options, expected_location = TestE2BExecutionBudgets._configure(base_config, location)
        options["volume_mounts"] = mounts

        with pytest.raises(ValidationError, match=re.escape(f"{expected_location}.volume_mounts")):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize("override", [None, {}, {"/mnt/profile": "profile-data"}])
    def test_profile_overrides_cannot_hide_invalid_common_mounts(self, base_config, override):
        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        execution["volume_mounts"] = {"relative": "invalid-default"}
        for profile in execution["profiles"].values():
            profile["volume_mounts"] = override

        with pytest.raises(ValidationError, match=r"rollout\.execution\.volume_mounts"):
            NorthRLConfig.model_validate(base_config)

    @pytest.mark.parametrize(
        "profile_options, expected",
        [
            ({}, {"/mnt/common": "common-data", "/mnt/shared": "shared-data"}),
            ({"volume_mounts": {"/mnt/shared": "profile-data"}}, {"/mnt/shared": "profile-data"}),
            ({"volume_mounts": {}}, {}),
            ({"volume_mounts": None}, None),
        ],
    )
    def test_profile_mounts_inherit_replace_or_clear_without_mutating_config(
        self, base_config, profile_options, expected
    ):
        from northrl.remote_rollout.execution_profiles import get_e2b_execution_configs

        execution, _ = TestE2BExecutionBudgets._configure(base_config, "common")
        common_mounts = {"/mnt/common": "common-data", "/mnt/shared": "shared-data"}
        execution["volume_mounts"] = common_mounts
        execution["agent_timeout_seconds"] = 600
        execution["profiles"]["code"].update(profile_options, agent_timeout_seconds=900)
        original_input = copy.deepcopy(base_config)
        validated = NorthRLConfig.model_validate(base_config)
        downstream = OmegaConf.create(validated.model_dump(by_alias=True))
        original_config = OmegaConf.to_container(downstream, resolve=False)

        effective = get_e2b_execution_configs(downstream)

        assert effective["code"].rollout.execution.volume_mounts == expected
        assert effective["swe"].rollout.execution.volume_mounts == common_mounts
        assert effective["code"].rollout.execution.agent_timeout_seconds == 900
        assert effective["swe"].rollout.execution.agent_timeout_seconds == 600
        effective["swe"].rollout.execution.volume_mounts["/mnt/added"] = "new-data"
        assert "/mnt/added" not in (effective["code"].rollout.execution.volume_mounts or {})
        assert OmegaConf.to_container(downstream, resolve=False) == original_config
        assert base_config == original_input


def test_weaver_metrics_store_schema():
    from northrl.utils.config_schema import BackendConfig

    assert BackendConfig().metrics_store.model_dump() == {"enabled": True, "path": None}
    assert BackendConfig(metrics_store={"enabled": False}).metrics_store.enabled is False
    assert (
        BackendConfig(metrics_store={"path": "/tmp/metrics"}).metrics_store.path == "/tmp/metrics"
    )
    with pytest.raises(ValueError):
        BackendConfig(metrics_store={"enable": False})
    with pytest.raises(ValueError):
        BackendConfig(metrics_store={"enabled": "false"})
