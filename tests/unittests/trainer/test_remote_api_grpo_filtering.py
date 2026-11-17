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

"""GRPO trajectory-retention and length-stop regression tests."""

from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from northrl.algorithm.advantage import build_estimator
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.trainer.router_replay_handler import RouterReplayHandler
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums


def _trainer() -> RemoteApiGrpoTrainer:
    trainer = RemoteApiGrpoTrainer.__new__(RemoteApiGrpoTrainer)
    trainer._config = OmegaConf.create(
        {"data": {"max_prompt_length": 16, "max_sequence_length": 64}}
    )
    trainer._loss_agg_mode = "token-mean"
    trainer._router_replay = RouterReplayHandler(
        {
            "sampling_logprob_source": "recomputed",
            "router_replay": {"enabled": False, "mode": "R3"},
        }
    )
    trainer._kl_enabled = False
    trainer._algorithm_config = AlgorithmConfig(type="grpo", optim={"lr": 1e-6})
    trainer._advantage_estimator = build_estimator(trainer._algorithm_config.advantage.estimator)
    return trainer


def _trajectory(group_id: str, run_id: int, reward: float, finish_reason: str = "stop"):
    return Trajectory(
        tokens=[10, 11, 12],
        loss_mask=[0, 1, 1],
        reward=reward,
        is_val=False,
        extra_fields={
            "group_id": group_id,
            "run_id": run_id,
            "finish_reason": finish_reason,
            "old_log_probs": [0.0, -0.1, -0.2],
        },
    )


def test_constant_groups_and_mixed_group_zero_advantages_are_retained():
    trainer = _trainer()
    constant_group = [_trajectory("constant", run_id, 0.1) for run_id in range(16)]
    mixed_rewards = [0.0] * 9 + [0.1] * 6 + [1.0]
    mixed_group = [
        _trajectory("mixed", run_id, reward) for run_id, reward in enumerate(mixed_rewards)
    ]
    metrics = {}

    prepared = run_coro(trainer._prepare_trajectories(constant_group + mixed_group, metrics))
    prepared = trainer._advantage_estimator.estimate(prepared, {}, metrics)

    assert len(prepared) == 32
    assert {trajectory.get("group_id") for trajectory in prepared} == {"constant", "mixed"}
    assert sum(float(trajectory.get("advantage")) == 0.0 for trajectory in prepared) == 22
    assert "grpo/constant_reward_groups_filtered" not in metrics
    assert "grpo/constant_reward_trajectories_filtered" not in metrics

    datums = trainer._convert_to_datums(prepared)
    assert len(datums) == 32
    assert (
        sum(
            all(advantage == 0.0 for advantage in datum["loss_fn_inputs"]["advantages"])
            for datum in datums
        )
        == 22
    )


def test_length_stop_keeps_loss_mask_and_datum_without_old_truncation_metrics():
    trainer = _trainer()
    length_trajectory = _trajectory("mixed", 0, 0.0, finish_reason="length")
    successful_trajectory = _trajectory("mixed", 1, 1.0)
    original_loss_mask = list(length_trajectory.loss_mask)
    metrics = {}

    prepared = run_coro(
        trainer._prepare_trajectories(
            [length_trajectory, successful_trajectory],
            metrics,
        )
    )

    prepared = trainer._advantage_estimator.estimate(prepared, {}, metrics)
    assert len(prepared) == 2
    assert length_trajectory.loss_mask == original_loss_mask
    assert "truncation/count" not in metrics
    assert "truncation/ratio" not in metrics
    assert len(trainer._convert_to_datums(prepared)) == 2


def test_empty_batch_skips_recompute():
    trainer = _trainer()
    metrics = {}

    result = run_coro(trainer._recompute_logprobs([], metrics))

    assert result == []
    assert not any(key.startswith("recompute_logprobs/") for key in metrics)
    assert metrics["timing/recompute_logprobs"] == 0


@pytest.mark.parametrize(("skipped", "expected_sync_count"), [(1, 0), (0, 1)])
def test_actor_to_ref_sync_only_runs_after_a_weight_update(
    monkeypatch: pytest.MonkeyPatch,
    skipped: int,
    expected_sync_count: int,
):
    trainer = RemoteApiGrpoTrainer.__new__(RemoteApiGrpoTrainer)
    trainer._kl_enabled = True
    trainer._actor_to_ref_sync_freq = 5
    trainer._ref_training_client = object()
    trainer._train_step = 4
    metrics = {"training/skipped_flag": skipped}
    sampling_path = None if skipped else "weaver://actor-step-5"
    weight_version = None if skipped else "northrl_actor_v000005"

    async def train_round(_trainer, _trajectories):
        return metrics, sampling_path, weight_version

    sync_actor_to_ref = AsyncMock()
    monkeypatch.setattr(RemoteApiTrainer, "train", train_round)
    monkeypatch.setattr(trainer, "_sync_actor_to_ref", sync_actor_to_ref)

    returned_metrics, returned_path, returned_version = run_coro(trainer.train([]))
    assert returned_metrics is metrics
    assert (returned_path, returned_version) == (sampling_path, weight_version)
    assert sync_actor_to_ref.await_count == expected_sync_count
    if expected_sync_count:
        sync_actor_to_ref.assert_awaited_once_with(5, metrics)


def test_generic_converter_keeps_legacy_zero_advantage_filtering():
    trajectory = _trajectory("mixed", 0, 0.1)
    trajectory.extra_fields["advantage"] = 0.0

    assert convert_trajectories_to_datums([trajectory]) == []


def test_grpo_converter_omits_all_zero_loss_mask():
    trainer = _trainer()
    trajectory = _trajectory("constant", 0, 1.0)
    trajectory.loss_mask = [0, 0, 0]
    trajectory.extra_fields["advantage"] = 1.0

    datums = trainer._convert_to_datums([trajectory])

    assert datums == []
    assert trainer._last_datum_sources == []


def test_grpo_empty_mask_filter_keeps_group_rewards_and_surviving_sources():
    trainer = _trainer()
    empty = _trajectory("mixed", 0, 0.0)
    empty.loss_mask = [0, 0, 0]
    active = _trajectory("mixed", 1, 1.0)
    metrics = {}

    prepared = run_coro(trainer._prepare_trajectories([active, empty], metrics))
    prepared = trainer._advantage_estimator.estimate(prepared, {}, metrics)
    expected = 0.5 / (0.5**0.5 + 1e-6)
    datums = trainer._convert_to_datums(prepared)

    assert prepared == [empty, active]
    assert float(empty["advantage"]) == pytest.approx(-expected)
    assert float(active["advantage"]) == pytest.approx(expected)
    assert len(datums) == 1
    assert datums[0]["loss_fn_inputs"]["advantages"] == pytest.approx([expected, expected])
    assert trainer._last_datum_sources == [(1, 0)]


def test_grpo_preparation_uses_zero_segment_trajectory_reward():
    trainer = _trainer()
    winner = Trajectory(
        segments=[{"tokens": [10, 11], "loss_mask": [0, 1]} for _ in range(3)],
        reward=1.0,
        extra_fields={"group_id": "group", "run_id": 0, "rollout_id": "a", "expected_rollouts": 2},
    )
    loser = Trajectory(
        segments=[],
        reward=0.0,
        extra_fields={"group_id": "group", "run_id": 1, "rollout_id": "b", "expected_rollouts": 2},
    )
    metrics = {}
    prepared = run_coro(trainer._prepare_trajectories([winner, loser], metrics))
    prepared = trainer._advantage_estimator.estimate(prepared, {}, metrics)
    expected = 0.5 / (0.5**0.5 + 1e-6)
    assert prepared == [winner, loser]
    assert [float(segment["advantage"]) for segment in winner.segments] == pytest.approx(
        [expected] * 3
    )
    assert float(loser["advantage"]) == pytest.approx(-expected)
    assert metrics["group/reward_std_min"] == pytest.approx(0.5**0.5)
    assert metrics["group/reward_std_max"] == pytest.approx(0.5**0.5)
    assert "critic/grpo_std/mean" not in metrics
    assert metrics["trajectory/advantage_positive_count"] == 1
    assert metrics["trajectory/advantage_negative_count"] == 1
    assert metrics["trajectory/advantage_mean"] == pytest.approx(0.0)
