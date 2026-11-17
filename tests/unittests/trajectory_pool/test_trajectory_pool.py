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

"""
Tests for TrajectoryPool and TrajectoryPoolInstance
"""

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPool, TrajectoryPoolInstance

# =============================================================================
# Tests for TrajectoryPoolInstance
# =============================================================================


@pytest.fixture
def trajectory_pool_instance_config():
    """Configuration for TrajectoryPoolInstance testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
        }
    )


def test_trajectory_pool_instance_initialization(trajectory_pool_instance_config):
    """Test TrajectoryPoolInstance initialization"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    assert instance.model_tag == "default"
    assert instance._store is not None


def test_trajectory_pool_instance_put_trajectory(trajectory_pool_instance_config):
    """Test putting trajectories into instance"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    trajectory = Trajectory(
        tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": "default"}
    )

    result = instance.put_trajectory(trajectory)
    assert result == "success"


def test_trajectory_pool_instance_is_empty(trajectory_pool_instance_config):
    """Test is_empty for instance"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    assert instance.is_empty()

    trajectory = Trajectory(
        tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": "default"}
    )
    instance.put_trajectory(trajectory)

    assert not instance.is_empty()


def test_trajectory_pool_instance_set_module_references(trajectory_pool_instance_config):
    """Test setting module references"""
    instance = TrajectoryPoolInstance(trajectory_pool_instance_config, model_tag="default")

    class MockDataLoader:
        def can_return_item(self):
            return False

    class MockWeightSyncController:
        def trajectory_pool_notify_batch_ready(self, model_tag):
            pass

    class MockActivityTracker:
        def is_rollout_worker_quiescent(self):
            return True

        def experiment_logger_post(self, backend, data, step=None):
            pass

    dataloader = MockDataLoader()
    controller = MockWeightSyncController()
    tracker = MockActivityTracker()

    instance.set_module_references(dataloader, controller, tracker)

    assert instance._dataloader is dataloader
    assert instance._weight_sync_controller is controller
    assert instance._activity_tracker is tracker


@pytest.mark.parametrize("need_sampling", [True, False])
@pytest.mark.parametrize("num_trajectories", [3, 8, 11])
def test_completed_loader_batch_returns_all_variable_count_trajectories(
    need_sampling: bool, num_trajectories: int
) -> None:
    config = OmegaConf.create(
        {
            "batch_size": 8,
            "need_sampling": need_sampling,
        }
    )
    instance = TrajectoryPoolInstance(config, model_tag="default")

    class _DataLoader:
        has_items = True

        def can_return_item(self):
            return self.has_items

    class _WeightSyncController:
        def __init__(self):
            self.ready = []

        def trajectory_pool_notify_batch_ready(self, model_tag):
            self.ready.append(model_tag)

    class _ActivityTracker:
        quiescent = False

        def is_rollout_worker_quiescent(self):
            return self.quiescent

        @staticmethod
        def experiment_logger_post(**kwargs):
            del kwargs

    weight_sync = _WeightSyncController()
    dataloader = _DataLoader()
    activity_tracker = _ActivityTracker()
    instance.set_module_references(dataloader, weight_sync, activity_tracker)
    # Token-budget loader batches can contain fewer or more parents than nominal B.
    trajectories = [
        Trajectory(tokens=[index], loss_mask=[1], reward=1.0, extra_fields={"model_tag": "default"})
        for index in range(num_trajectories)
    ]
    for trajectory in trajectories[:-1]:
        assert instance.put_trajectory(trajectory) == "success"

    assert instance.get_trajectories() is None
    activity_tracker.quiescent = True
    assert instance.get_trajectories() is None  # Idle workers alone cannot end the loader batch.
    activity_tracker.quiescent = False
    dataloader.has_items = False
    assert instance.get_trajectories() is None
    # A claimed task can still finish after the dataloader buffer is empty.
    assert instance.put_trajectory(trajectories[-1]) == "success"
    assert instance.get_trajectories() is None
    assert weight_sync.ready == []

    activity_tracker.quiescent = True
    assert instance.get_trajectories() == trajectories
    assert instance.is_empty()
    assert instance.get_trajectories() is None
    assert weight_sync.ready == (["default"] if need_sampling else [])


def test_finished_source_requires_complete_rollout_groups() -> None:
    config = OmegaConf.create(
        {
            "batch_size": 8,
            "group_size": 4,
            "groups_per_batch": 2,
            "key_list": ["group_id"],
            "need_sampling": False,
        }
    )
    instance = TrajectoryPoolInstance(config, model_tag="default")

    class _DataLoader:
        @staticmethod
        def can_return_item():
            return False

        @staticmethod
        def is_finished():
            return True

    class _WeightSyncController:
        @staticmethod
        def trajectory_pool_notify_batch_ready(model_tag):
            del model_tag

    class _ActivityTracker:
        @staticmethod
        def is_rollout_worker_quiescent():
            return True

        @staticmethod
        def experiment_logger_post(**kwargs):
            del kwargs

    instance.set_module_references(_DataLoader(), _WeightSyncController(), _ActivityTracker())
    trajectories = [
        Trajectory(
            segments=[{"tokens": [run_id, index], "loss_mask": [0, 1]} for index in range(count)],
            reward=1.0,
            extra_fields={
                "group_id": "group-1",
                "model_tag": "default",
                "run_id": run_id,
                "rollout_id": f"rollout-{run_id}",
                "expected_rollouts": 4,
            },
        )
        for run_id, count in enumerate([2, 2, 2, 1])
    ]
    for trajectory in trajectories[:-1]:
        assert instance.put_trajectory(trajectory) == "success"
    # Quiescence cannot manufacture a missing complete rollout result.
    with pytest.raises(RuntimeError, match="unfinished rollout groups"):
        instance.get_trajectories()
    assert instance.put_trajectory(trajectories[-1]) == "success"
    assert instance.get_trajectories() == trajectories
    assert instance.is_empty()


# =============================================================================
# Tests for TrajectoryPool
# =============================================================================


@pytest.fixture
def trajectory_pool_config():
    """Configuration for TrajectoryPool testing"""
    return OmegaConf.create(
        {
            "batch_size": 4,
        }
    )


@pytest.fixture
def mock_module_references():
    """Create mock module references for TrajectoryPool"""

    class MockDataLoader:
        def can_return_item(self):
            return False

    class MockWeightSyncController:
        def trajectory_pool_notify_batch_ready(self, model_tag):
            pass

    class MockActivityTracker:
        def is_rollout_worker_quiescent(self):
            return True

        def experiment_logger_post(self, backend, data, step=None):
            pass

    return {
        "dataloader": MockDataLoader(),
        "weight_sync_controller": MockWeightSyncController(),
        "activity_tracker": MockActivityTracker(),
    }


def test_trajectory_pool_initialization(trajectory_pool_config):
    """Test TrajectoryPool initialization"""
    pool = TrajectoryPool(trajectory_pool_config)

    assert pool._config is not None
    assert pool._instances == {}


def test_trajectory_pool_put_trajectory(trajectory_pool_config, mock_module_references):
    """Test putting trajectory into pool"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    trajectory = Trajectory(
        tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": "default"}
    )

    result = pool.put_trajectory(trajectory)
    assert result == "success"
    assert "default" in pool._instances


def test_trajectory_pool_multiple_models(trajectory_pool_config, mock_module_references):
    """Test managing multiple model tags"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectories for different models
    for model_tag in ["model1", "model2"]:
        trajectory = Trajectory(
            tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": model_tag}
        )
        pool.put_trajectory(trajectory)

    assert len(pool._instances) == 2
    assert "model1" in pool._instances
    assert "model2" in pool._instances


def test_trajectory_pool_get_model_tags(trajectory_pool_config, mock_module_references):
    """Test getting all model tags"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectories for different models
    for model_tag in ["model1", "model2", "model3"]:
        trajectory = Trajectory(
            tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": model_tag}
        )
        pool.put_trajectory(trajectory)

    model_tags = pool.get_model_tags()
    assert len(model_tags) == 3
    assert "model1" in model_tags
    assert "model2" in model_tags
    assert "model3" in model_tags


def test_trajectory_pool_is_empty(trajectory_pool_config, mock_module_references):
    """Test is_empty for pool"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Empty pool
    assert pool.is_empty()

    # Add trajectory
    trajectory = Trajectory(
        tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": "default"}
    )
    pool.put_trajectory(trajectory)

    assert not pool.is_empty()


def test_trajectory_pool_is_empty_specific_model(trajectory_pool_config, mock_module_references):
    """Test is_empty for specific model tag"""
    pool = TrajectoryPool(trajectory_pool_config)
    pool.set_module_references(
        dataloader=mock_module_references["dataloader"],
        weight_sync_controller=mock_module_references["weight_sync_controller"],
    )
    pool._activity_tracker = mock_module_references["activity_tracker"]

    # Add trajectory to model1
    trajectory = Trajectory(
        tokens=[1, 2, 3], loss_mask=[0, 1, 1], reward=1.0, extra_fields={"model_tag": "model1"}
    )
    pool.put_trajectory(trajectory)

    # model1 should not be empty
    assert not pool.is_empty(model_tag="model1")

    # model2 doesn't exist, should be empty
    assert pool.is_empty(model_tag="model2")


def test_trajectory_pool_module_name(trajectory_pool_config):
    """Test trajectory pool module name functionality"""
    pool = TrajectoryPool(trajectory_pool_config)

    pool.set_module_name("test_trajectory_pool")
    assert pool.get_module_name() == "test_trajectory_pool"
