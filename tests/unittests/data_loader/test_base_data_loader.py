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
Tests for BaseDataLoader and SequentialDataLoader (via MockDataLoader)
"""

from northrl.data_loader import BaseDataLoader
from northrl.mock import MockDataLoader

# =============================================================================
# Tests for BaseDataLoader
# =============================================================================


def test_base_data_loader_repeat_item():
    """Test static repeat_item method"""
    original_item = {"id": 1, "prompt": "test"}
    n = 3

    repeated_items = BaseDataLoader.repeat_item(original_item, n)

    # Should have n items
    assert len(repeated_items) == n

    # All should have the same group_id
    group_ids = [item["group_id"] for item in repeated_items]
    assert len(set(group_ids)) == 1

    # Each should have different run_id
    run_ids = [item["run_id"] for item in repeated_items]
    assert run_ids == list(range(n))

    # All should have original fields
    for item in repeated_items:
        assert item["id"] == 1
        assert item["prompt"] == "test"


def test_base_data_loader_module_name(data_loader_config):
    """Test data loader module name functionality"""
    loader = MockDataLoader(data_loader_config)

    # Set module name
    loader.set_module_name("test_data_loader")
    assert loader.get_module_name() == "test_data_loader"


# =============================================================================
# Tests for SequentialDataLoader (via MockDataLoader)
# =============================================================================


def test_sequential_data_loader_initialization(data_loader_config):
    """Test SequentialDataLoader initialization"""
    loader = MockDataLoader(data_loader_config)

    assert loader._batch_size == 4
    assert len(loader._data) > 0
    assert loader._data_index == 0
    assert loader._buffer_index == 0


def test_sequential_data_loader_get_next_item(data_loader_config):
    """Test getting next item from sequential data loader"""
    loader = MockDataLoader(data_loader_config)

    # Get first item
    item = loader.get_next_item()
    assert item is not None
    assert "prompt" in item
    assert "id" in item
    assert loader._data_index == 1


def test_sequential_data_loader_can_return_item(data_loader_config):
    """Test checking if sequential data loader can return items"""
    loader = MockDataLoader(data_loader_config)

    # Initially can't return (buffer not loaded)
    # After first get_next_item, buffer is loaded
    item = loader.get_next_item()
    assert item is not None
    assert loader.can_return_item() or loader._buffer_index < len(loader._data_buffer)


def test_sequential_data_loader_is_finished(data_loader_config):
    """Test checking if sequential data loader is finished"""
    loader = MockDataLoader(data_loader_config)

    # Initially not finished
    assert not loader.is_finished()

    # Fetch all data
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            loader.release_next_batch()
            continue

    # Now should be finished
    assert loader.is_finished()


def test_sequential_data_loader_reset(data_loader_config):
    """Test resetting sequential data loader"""
    loader = MockDataLoader(data_loader_config)

    # Fetch some data
    loader.get_next_item()
    loader.get_next_item()
    assert loader._data_index > 0

    # Reset
    loader.reset()
    assert loader._data_index == 0
    assert loader._buffer_index == 0


def test_sequential_data_loader_release_next_batch(data_loader_config):
    """Ordered loaders expose the next batch only after release."""
    data_loader_config.keep_batch_order = True
    loader = MockDataLoader(data_loader_config)

    first_batch = [loader.get_next_item() for _ in range(data_loader_config.batch_size)]
    assert [item["id"] for item in first_batch if item is not None] == [0, 1, 2, 3]
    assert loader.get_next_item() is None

    loader.release_next_batch()
    item = loader.get_next_item()
    assert item is not None and item["id"] == 4


def test_repeated_draws_keep_task_identity_but_isolate_execution_groups():
    task = {"task_id": "same-question", "prompt": "test", "group_id": "old-group"}
    first = BaseDataLoader.repeat_item(task, 2)
    second = BaseDataLoader.repeat_item(task, 2)
    assert first[0]["group_id"] != second[0]["group_id"]
    assert len({item["rollout_id"] for item in first + second}) == 4
    assert all(item["expected_rollouts"] == 2 for item in first + second)
    assert all(item["task_id"] == "same-question" for item in first + second)
    assert task["group_id"] == "old-group"


def test_one_rollout_still_has_completion_identity():
    (item,) = BaseDataLoader.repeat_item({"prompt": "single"}, 1)
    assert item["group_id"] and item["rollout_id"]
    assert item["run_id"] == 0
    assert item["expected_rollouts"] == 1


def test_batches_preserve_sampling_group_and_rollout_identities():
    from omegaconf import OmegaConf

    from northrl.mock import MockDataLoader

    loader = MockDataLoader(
        OmegaConf.create(
            {
                "batch_size": 2,
                "rollout_repeat_n": 3,
                "keep_batch_order": True,
            }
        )
    )
    first = [loader.get_next_item() for _ in range(6)]
    assert len({item["group_id"] for item in first}) == 2
    assert len({item["rollout_id"] for item in first}) == 6
    assert [item["run_id"] for item in first] == [0, 1, 2, 0, 1, 2]
    assert first[0]["group_id"] == first[2]["group_id"]
    assert loader.get_next_item() is None
    loader.release_next_batch()
    next_item = loader.get_next_item()
    assert next_item["group_id"] not in {item["group_id"] for item in first}
    assert next_item["rollout_id"] not in {item["rollout_id"] for item in first}
    assert all(item["expected_rollouts"] == 3 for item in first)
