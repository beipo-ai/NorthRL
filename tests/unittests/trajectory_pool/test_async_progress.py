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

"""Pool inventory progress while sampling and training overlap."""

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPool, TrajectoryPoolInstance


def _config():
    return OmegaConf.create(
        {
            "key_list": ["group_id"],
            "group_size": 2,
            "groups_per_batch": 2,
            "need_sampling": False,
        }
    )


def _result(group, run_id, *, model_tag="default", segments=1, status="success"):
    return Trajectory(
        segments=[{"tokens": [1, 2, index], "loss_mask": [0, 1, 1]} for index in range(segments)],
        reward=1.0,
        status=status,
        extra_fields={
            "model_tag": model_tag,
            "group_id": group,
            "run_id": run_id,
            "rollout_id": f"{model_tag}-{group}-{run_id}",
            "expected_rollouts": 2,
        },
    )


def test_partial_groups_become_ready_then_leave_inventory_on_consumption():
    instance = TrajectoryPoolInstance(_config(), model_tag="default")
    initial = instance.get_progress_info()
    assert initial["ready_groups"] == initial["incomplete_groups"] == 0
    assert initial["groups_per_batch"] == 2

    first = _result("first", 0, segments=3)
    second = _result("second", 0, segments=2)
    assert instance.put_trajectory(first) == "success"
    assert instance.put_trajectory(second) == "success"
    assert instance.put_trajectory(first) == "success"
    partial = instance.get_progress_info()
    assert partial["incomplete_groups"] == 2
    assert partial["ready_groups"] == partial["ready_segments"] == 0
    assert instance.get_trajectories() is None

    first_last = _result("first", 1, segments=1)
    assert instance.put_trajectory(first_last) == "success"
    mixed = instance.get_progress_info()
    assert mixed["incomplete_groups"] == 1
    assert mixed["ready_groups"] == 1
    assert mixed["ready_trajectories"] == 2
    assert mixed["ready_segments"] == mixed["ready_sequences"] == 4
    assert instance._store.progress_counts() == (4, 2, 1)
    assert instance.get_trajectories() is None

    second_last = _result("second", 1, segments=0)
    assert instance.put_trajectory(second_last) == "success"
    ready = instance.get_progress_info()
    assert ready["incomplete_groups"] == 0
    assert ready["ready_groups"] == ready["groups_per_batch"] == 2
    assert ready["ready_segments"] == 6
    assert instance.get_trajectories() == [first, first_last, second, second_last]
    assert instance.get_progress_info() == initial


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_repairable_complete_group_counts_only_original_valid_segments(status):
    instance = TrajectoryPoolInstance(_config(), model_tag="default")
    assert instance.put_trajectory(_result("skipped", 0, segments=3)) == "success"
    assert instance.get_progress_info()["incomplete_groups"] == 1
    assert instance.put_trajectory(_result("skipped", 1, segments=0, status=status)) == "success"
    info = instance.get_progress_info()
    assert info["incomplete_groups"] == 0
    assert info["ready_groups"] == 1
    assert info["ready_trajectories"] == 2
    assert info["ready_segments"] == 3
    assert instance._store.progress_counts() == (3, 2, 1)


def test_multiple_models_keep_independent_batch_targets_and_aggregate_inventory():
    pool = TrajectoryPool(_config())
    pool.set_module_references(dataloader=None, weight_sync_controller=None)
    empty = pool.get_progress_info()
    assert empty["ready_groups"] == empty["incomplete_groups"] == 0
    assert empty["instances"] == {}

    for model_tag in ("actor-a", "actor-b"):
        for run_id in (0, 1):
            assert pool.put_trajectory(_result("ready", run_id, model_tag=model_tag)) == "success"
        assert pool.put_trajectory(_result("partial", 0, model_tag=model_tag)) == "success"
    assert pool.put_trajectory(_result("extra", 0, model_tag="actor-b")) == "success"

    info = pool.get_progress_info()
    assert info["ready_groups"] == 2
    assert info["incomplete_groups"] == 3
    assert info["ready_segments"] == 4
    for model_tag, partial_count in (("actor-a", 1), ("actor-b", 2)):
        model = info["instances"][model_tag]
        assert model["groups_per_batch"] == 2
        assert model["ready_groups"] == 1
        assert model["incomplete_groups"] == partial_count
        assert pool.get_trajectories(model_tag) is None


def test_simple_store_reports_no_group_inventory_or_group_target():
    instance = TrajectoryPoolInstance(
        OmegaConf.create({"batch_size": 2, "need_sampling": False}), model_tag="default"
    )
    assert instance.put_trajectory(_result("ungrouped", 0, segments=3)) == "success"
    info = instance.get_progress_info()
    assert info["ready_segments"] == 3
    assert info["ready_trajectories"] == 1
    assert info["ready_groups"] == info["incomplete_groups"] == info["groups_per_batch"] == 0
    assert instance._store.progress_counts() == (3, 1, 0)
