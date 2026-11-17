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

"""Replacement training samples preserve original execution accounting."""

import copy

import pytest
from omegaconf import OmegaConf

from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.trajectory_pool import GroupedTrajectoryStore
from northrl.utils.async_utils import run_coro
from northrl.utils.logging_utils import log_rollout_metrics
from northrl.utils.trajectory_sequences import iter_training_sequences

from .test_remote_api_trainer_mis import _make_trainer


def _group(group="group", rewards=(1.0, 0.0, None, None), *, model="actor"):
    return [
        Trajectory(
            segments=(
                [
                    {
                        "tokens": [10 + run, 20, 30],
                        "loss_mask": [0, 1, 1],
                        "old_log_probs": [0.0, -4.0, -4.0],
                        "moe_topk_indices_ref": {"relative_path": f"routing-{run}.bin"},
                    }
                ]
                if reward is not None
                else []
            ),
            reward=reward,
            status="success" if reward is not None else "failed",
            extra_fields={
                "model_tag": model,
                "group_id": group,
                "run_id": run,
                "rollout_id": f"{model}-{group}-{run}",
                "expected_rollouts": len(rewards),
                "group_skipped": reward is None,
                "runtime_metrics": {"model_requests": run + 1, "duration_seconds": run + 1.0},
                "e2b_status": "success" if reward is not None else "env_error",
                "score": {"reward": reward} if reward is not None else {},
            },
        )
        for run, reward in enumerate(rewards)
    ]


def test_half_failed_group_uses_each_donor_once_with_independent_segments_and_slot_identities():
    originals = _group()
    metrics = {}
    repaired = RemoteApiTrainer._resample_failed_groups(originals, metrics)

    assert len(repaired) == 4
    assert repaired[:2] == originals[:2]
    assert [item["run_id"] for item in repaired] == [0, 1, 2, 3]
    assert [item["rollout_id"] for item in repaired] == [item["rollout_id"] for item in originals]
    assert all(item.status == "success" and not item.get("group_skipped") for item in repaired)
    assert {item["resampled_from_rollout_id"] for item in repaired[2:]} == {
        item["rollout_id"] for item in originals[:2]
    }
    assert metrics["batch/resampled_group_count"] == 1
    assert metrics["batch/resampled_trajectory_count"] == 2
    assert all(item.status == "failed" and item.segments == [] for item in originals[2:])

    donors = {item["rollout_id"]: item for item in originals[:2]}
    for clone in repaired[2:]:
        donor = donors[clone["resampled_from_rollout_id"]]
        assert clone.reward == donor.reward
        assert clone.segments == donor.segments
        clone.segments[0]["tokens"][0] = 999
        clone.segments[0]["old_log_probs"][1] = -99.0
        clone.segments[0]["moe_topk_indices_ref"]["relative_path"] = "changed.bin"
        clone["runtime_metrics"]["model_requests"] = 999
        assert donor.segments[0]["tokens"][0] != 999
        assert donor.segments[0]["old_log_probs"][1] == -4.0
        assert donor.segments[0]["moe_topk_indices_ref"]["relative_path"] != "changed.bin"
        assert donor["runtime_metrics"]["model_requests"] != 999


def test_repaired_rewards_define_grpo_statistics_and_segment_advantages():
    repaired = RemoteApiTrainer._resample_failed_groups(_group(), {})
    compute_grpo_advantage_for_trajectories(repaired, use_run_ids=True)
    sequences = list(iter_training_sequences(repaired))

    # Half-failure repair gives [1, 0, 1, 0] in some replacement order.
    expected = 0.5 / ((1.0 / 3.0) ** 0.5 + 1e-6)
    assert len(sequences) == 4
    assert [float(item["advantage"]) for item in repaired] == pytest.approx(
        [expected if item.reward == 1.0 else -expected for item in repaired]
    )
    assert [seq["rollout_id"] for seq in sequences] == [item["rollout_id"] for item in repaired]
    assert [seq["run_id"] for seq in sequences] == [0, 1, 2, 3]


def test_zero_reward_and_zero_segment_members_remain_eligible_donors():
    originals = _group(rewards=(0.0, None))
    originals[0].segments = []
    originals[0]["e2b_status"] = "task_failed"
    repaired = RemoteApiTrainer._resample_failed_groups(originals, {})
    assert [item.reward for item in repaired] == [0.0, 0.0]
    assert all(item.status == "success" and item.segments == [] for item in repaired)
    compute_grpo_advantage_for_trajectories(repaired)
    assert [item["advantage"] for item in repaired] == [0.0, 0.0]


def test_majority_failed_group_stays_skipped_without_replacements():
    originals = _group(rewards=(1.0, None, None))
    for item in originals:
        item["group_skipped"] = True
    metrics = {}
    repaired = RemoteApiTrainer._resample_failed_groups(originals, metrics)

    assert repaired == originals
    assert not any("resampled_from_rollout_id" in item for item in repaired)
    assert metrics["batch/resampled_group_count"] == 0
    assert metrics["batch/resampled_trajectory_count"] == 0
    compute_grpo_advantage_for_trajectories(repaired)
    assert list(iter_training_sequences(repaired)) == []


def test_donors_never_cross_group_or_model_boundaries():
    groups = [
        _group("shared", (1.0, None), model="actor-a"),
        _group("shared", (2.0, None), model="actor-b"),
        _group("other", (3.0, None), model="actor-a"),
    ]
    originals = [group[run] for run in range(2) for group in groups]
    metrics = {}
    repaired = RemoteApiTrainer._resample_failed_groups(originals, metrics)
    sources = {item["rollout_id"]: item for item in originals}

    assert len(repaired) == len(originals)
    for item in repaired:
        if "resampled_from_rollout_id" in item:
            source = sources[item["resampled_from_rollout_id"]]
            assert (item["model_tag"], item["group_id"], item.reward) == (
                source["model_tag"],
                source["group_id"],
                source.reward,
            )
    assert metrics["batch/resampled_group_count"] == 3
    assert metrics["batch/resampled_trajectory_count"] == 3


@pytest.mark.parametrize("invalid_reward", [None, True, float("nan"), float("inf"), "1.0"])
def test_pool_marks_invalid_rewards_before_raw_metrics_and_resampling(invalid_reward):
    store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 2}), "actor")
    originals = _group(rewards=(1.0, None))
    originals[1].status = "success"
    originals[1].reward = invalid_reward
    originals[1]["group_skipped"] = False
    for item in originals:
        assert store.put_trajectory(item)
    delivered = store.take_groups(num_groups=1)
    metrics = {}
    log_rollout_metrics(delivered, metrics)
    repaired = RemoteApiTrainer._resample_failed_groups(delivered, metrics)

    assert metrics["trajectory/reward_observed_count"] == 1
    assert metrics["trajectory/reward_mean"] == 1.0
    assert metrics["model_request/count"] == 3
    assert [item["group_skipped"] for item in originals] == [False, True]
    assert [item.reward for item in repaired] == [1.0, 1.0]


def test_training_copies_do_not_replace_original_retry_signatures_or_extend_history():
    store = GroupedTrajectoryStore(OmegaConf.create({"group_size": 2}), "actor")
    store._MAX_RECENT_GROUPS = 1
    originals = _group(rewards=(1.0, None))
    original_retries = copy.deepcopy(originals)
    for item in originals + original_retries:
        assert store.put_trajectory(item)
    delivered = store.take_groups(num_groups=1)
    repaired = RemoteApiTrainer._resample_failed_groups(delivered, {})
    assert repaired[1]["rollout_id"] == originals[1]["rollout_id"]
    for item in original_retries:
        assert store.put_trajectory(copy.deepcopy(item))
    assert store.take_groups(num_groups=1) is None

    next_group = _group("next", rewards=(1.0, 0.0))
    for item in next_group:
        assert store.put_trajectory(item)
    assert store.take_groups(num_groups=1) == next_group
    assert set(store._rollout_signatures) == {item["rollout_id"] for item in next_group}


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_train_logs_original_outcomes_before_resampling_and_recompute(monkeypatch, algorithm):
    trainer, holder = _make_trainer(monkeypatch, [[-4.0, -4.0]] * 4)
    trainer._config.algorithm.type = algorithm
    originals = _group()
    prepared = []

    async def prepare(trajectories, metrics):
        prepared.extend(trajectories)
        assert metrics["batch/failed_trajectory_count"] == 2
        assert metrics["outcome/env_error_count"] == 2
        assert metrics["trajectory/reward_observed_count"] == 2
        assert metrics["trajectory/reward_mean"] == 0.5
        assert metrics["model_request/count"] == 10
        assert metrics["batch/segment_count"] == 2
        assert metrics["verifier/result_count"] == 2
        assert all(item.status == "success" for item in trajectories)
        return compute_grpo_advantage_for_trajectories(trajectories, use_run_ids=True)

    monkeypatch.setattr(trainer, "_prepare_trajectories", prepare)
    metrics, _, _ = run_coro(trainer.train(originals))

    assert metrics["batch/resampled_group_count"] == 1
    assert metrics["batch/resampled_trajectory_count"] == 2
    assert len(prepared) == 4
    assert [event[0] for event in holder.events] == ["recompute", "forward_backward", "update"]
    assert len(holder.events[0][1]) == len(holder.calls[0]["datums_data"]) == 4
    assert [item.status for item in originals] == ["success", "success", "failed", "failed"]
