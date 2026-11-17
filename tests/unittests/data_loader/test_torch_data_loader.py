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
Tests for TorchDataLoader
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from northrl.data_loader import TorchDataLoader
from northrl.northrl_types import Trajectory
from northrl.rewards.math import math_reward
from northrl.trajectory_pool import TrajectoryPoolInstance


@pytest.fixture
def torch_data_loader_config():
    """Configuration for TorchDataLoader testing"""
    test_data_path = Path(__file__).parent / "test_data" / "test_dataset.parquet"

    return OmegaConf.create(
        {
            "files": [str(test_data_path)],
            "batch_size": 2,
            "rollout_repeat_n": 1,
            "prompt_key": "prompt",
            "filter_prompts": False,
            "shuffle": False,
            "drop_last": False,
            "max_prompt_length": 1024,
            "tokenizer": None,
            "seed": 42,
            "keep_batch_order": False,
        }
    )


def test_torch_data_loader_initialization(torch_data_loader_config):
    """Test TorchDataLoader initialization"""
    loader = TorchDataLoader(torch_data_loader_config)

    assert loader._batch_size == 2
    assert loader._dataset is not None
    assert loader._dataloader is not None


def test_torch_data_loader_get_next_item(torch_data_loader_config):
    """Test getting items from TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get first item
    item = loader.get_next_item()
    assert item is not None
    assert "prompt" in item
    assert "id" in item


def test_torch_data_loader_iteration(torch_data_loader_config):
    """Test iterating through TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    items = []
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            break
        items.append(item)

    # Should have loaded all 8 items from test dataset
    assert len(items) == 8


def test_torch_data_loader_reset(torch_data_loader_config):
    """Test resetting TorchDataLoader"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Fetch some items
    loader.get_next_item()
    loader.get_next_item()
    assert loader._data_index > 0

    # Reset
    loader.reset()
    assert loader._data_index == 0

    # Should be able to fetch items again
    item = loader.get_next_item()
    assert item is not None


def test_torch_data_loader_is_finished(torch_data_loader_config):
    """Test TorchDataLoader is_finished method"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Initially not finished
    assert not loader.is_finished()

    # Consume all items
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is None:
            break

    # Should be finished
    assert loader.is_finished()


def test_torch_data_loader_with_shuffle(torch_data_loader_config):
    """Test TorchDataLoader with shuffle enabled"""
    torch_data_loader_config.shuffle = True
    loader = TorchDataLoader(torch_data_loader_config)

    # Should initialize successfully with shuffle
    assert loader._dataset is not None

    # Should be able to get items
    item = loader.get_next_item()
    assert item is not None


def test_torch_data_loader_can_return_item(torch_data_loader_config):
    """Test can_return_item method"""
    loader = TorchDataLoader(torch_data_loader_config)

    # Get an item to load buffer
    item = loader.get_next_item()
    assert item is not None

    # Should be able to return items
    assert loader.can_return_item() or loader._buffer_index < len(loader._data_buffer)


def test_torch_data_loader_rejects_sft_mode(torch_data_loader_config):
    torch_data_loader_config.is_sft = True

    with pytest.raises(ValueError, match="only through SFTStreamingDataLoader"):
        TorchDataLoader(torch_data_loader_config)


def test_torch_data_loader_preserves_task_template_across_rollout_repeats(
    torch_data_loader_config, tmp_path
):
    data_path = tmp_path / "swebench.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "instruction": "Fix the bug",
                "template": "instance-specific-template",
                "task_id": "instance-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.prompt_key = "instruction"
    torch_data_loader_config.batch_size = 1
    torch_data_loader_config.rollout_repeat_n = 2

    loader = TorchDataLoader(torch_data_loader_config)

    items = [loader.get_next_item(), loader.get_next_item()]
    assert all(item is not None for item in items)
    assert [item["template"] for item in items] == [
        "instance-specific-template",
        "instance-specific-template",
    ]
    assert [item["run_id"] for item in items] == [0, 1]


def _write_cycle_dataset(path: Path, size: int) -> None:
    path.write_text(
        "".join(json.dumps({"prompt": f"prompt-{index}"}) + "\n" for index in range(size)),
        encoding="utf-8",
    )


def test_ordered_finite_short_batch_reaches_pool_without_another_release(
    torch_data_loader_config, tmp_path
):
    data_path = tmp_path / "finite-tail.jsonl"
    _write_cycle_dataset(data_path, size=2)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 3
    torch_data_loader_config.rollout_repeat_n = 2
    torch_data_loader_config.keep_batch_order = True
    loader = TorchDataLoader(torch_data_loader_config)
    activity = SimpleNamespace(quiescent=False)
    pool = TrajectoryPoolInstance(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": 2,
                "groups_per_batch": 3,
                "need_sampling": False,
            }
        ),
        "actor",
    )
    pool.set_module_references(
        loader,
        None,
        SimpleNamespace(is_rollout_worker_quiescent=lambda: activity.quiescent),
    )

    first = loader.get_next_item()
    assert first is not None
    assert not loader.is_finished()  # Three repeated items are still buffered.
    tasks = [first, *[loader.get_next_item() for _ in range(3)]]
    assert [task["index"] for task in tasks] == [0, 0, 1, 1]
    assert loader.get_next_item() is None
    assert loader.is_finished()  # No release is needed to discover this short tail.

    results = []
    for task in tasks:
        trajectory = Trajectory(
            tokens=[1, 2],
            loss_mask=[0, 1],
            reward=1.0,
            extra_fields={
                "model_tag": "actor",
                **{
                    key: task[key]
                    for key in ("group_id", "run_id", "rollout_id", "expected_rollouts")
                },
            },
        )
        assert pool.put_trajectory(trajectory) == "success"
        results.append(trajectory)
    assert pool.get_trajectories() is None  # The workers still own their activity windows.
    activity.quiescent = True
    assert pool.get_trajectories() == results
    assert pool.is_empty()
    assert pool.get_trajectories() is None

    loader.reset()
    assert not loader.is_finished()
    assert [loader.get_next_item()["index"] for _ in range(4)] == [0, 0, 1, 1]
    assert loader.is_finished()


def test_training_cycle_fills_batch_across_epoch_boundary(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "cycle.jsonl"
    _write_cycle_dataset(data_path, size=5)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 3
    torch_data_loader_config.shuffle = False
    torch_data_loader_config.drop_last = True
    torch_data_loader_config.cycle = True

    loader = TorchDataLoader(torch_data_loader_config)
    items = [loader.get_next_item() for _ in range(9)]

    assert all(item is not None for item in items)
    assert [item["index"] for item in items] == [0, 1, 2, 3, 4, 0, 1, 2, 3]
    assert loader._epoch_id == 1
    assert not loader.is_finished()


def test_training_cycle_shuffle_is_deterministic_per_epoch(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "shuffle-cycle.jsonl"
    _write_cycle_dataset(data_path, size=10)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 5
    torch_data_loader_config.shuffle = True
    torch_data_loader_config.drop_last = True
    torch_data_loader_config.cycle = True
    torch_data_loader_config.seed = 17

    first_loader = TorchDataLoader(torch_data_loader_config)
    second_loader = TorchDataLoader(torch_data_loader_config)
    first_stream = [first_loader.get_next_item()["index"] for _ in range(20)]
    second_stream = [second_loader.get_next_item()["index"] for _ in range(20)]

    assert first_stream == second_stream
    assert sorted(first_stream[:10]) == list(range(10))
    assert sorted(first_stream[10:]) == list(range(10))
    assert first_stream[:10] != first_stream[10:]
    assert first_loader._epoch_id == 1

    first_loader.reset()
    reset_epoch = [first_loader.get_next_item()["index"] for _ in range(10)]
    assert reset_epoch == first_stream[:10]
    assert first_loader._epoch_id == 0


def test_training_cycle_skip_batches_can_cross_epochs(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "resume-cycle.jsonl"
    _write_cycle_dataset(data_path, size=5)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 2
    torch_data_loader_config.shuffle = False
    torch_data_loader_config.cycle = True

    loader = TorchDataLoader(torch_data_loader_config)
    loader.skip_batches(4)

    assert loader._epoch_id == 1
    assert loader.get_next_item()["index"] == 3


def test_training_cycle_shuffle_resume_reconstructs_same_cursor(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "shuffle-resume-cycle.jsonl"
    _write_cycle_dataset(data_path, size=10)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 4
    torch_data_loader_config.shuffle = True
    torch_data_loader_config.cycle = True
    torch_data_loader_config.seed = 23

    reference = TorchDataLoader(torch_data_loader_config)
    expected_stream = [reference.get_next_item()["index"] for _ in range(17)]

    resumed = TorchDataLoader(torch_data_loader_config)
    resumed.skip_batches(4)

    assert resumed.get_next_item()["index"] == expected_stream[16]


def test_training_cycle_rejects_empty_filtered_dataset(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "empty.csv"
    data_path.write_text("prompt\n", encoding="utf-8")
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.cycle = True

    with pytest.raises(ValueError, match="Cannot cycle an empty training dataset"):
        TorchDataLoader(torch_data_loader_config)


def test_validation_loader_remains_finite_when_cycle_is_requested(
    torch_data_loader_config,
    tmp_path,
):
    data_path = tmp_path / "validation.jsonl"
    _write_cycle_dataset(data_path, size=5)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.batch_size = 3
    torch_data_loader_config.shuffle = False
    torch_data_loader_config.drop_last = False
    torch_data_loader_config.cycle = True

    loader = TorchDataLoader(torch_data_loader_config, is_validate=True)
    items = []
    while not loader.is_finished():
        item = loader.get_next_item()
        if item is not None:
            items.append(item)

    assert [item["index"] for item in items] == list(range(5))
    assert loader._epoch_id == 0
    assert loader.is_finished()


@pytest.mark.parametrize("file_format", ["jsonl", "parquet", "csv"])
def test_torch_data_loader_preserves_reward_fn_across_rollout_repeats(
    torch_data_loader_config, tmp_path, file_format
):
    records = [
        {"prompt": "explicit", "reward_fn": "math_with_format"},
        {"prompt": "missing"},
        {"prompt": "null-reward", "reward_fn": None},
        {"prompt": "empty", "reward_fn": ""},
        {"prompt": "unknown", "reward_fn": "unknown_reward"},
    ]
    data_path = tmp_path / f"rewards.{file_format}"
    if file_format == "jsonl":
        data_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    elif file_format == "parquet":
        pd.DataFrame(records).to_parquet(data_path, index=False)
    else:
        pd.DataFrame(records).to_csv(data_path, index=False)
    torch_data_loader_config.files = [str(data_path)]
    torch_data_loader_config.rollout_repeat_n = 2

    loader = TorchDataLoader(torch_data_loader_config)
    items = [loader.get_next_item() for _ in range(len(records) * 2)]

    for index, record in enumerate(records):
        for run_id in range(2):
            item = items[index * 2 + run_id]
            assert item["prompt"] == record["prompt"]
            assert item["run_id"] == run_id
            # CSV already interprets blank cells as missing during parsing.
            if record.get("reward_fn") is None or (
                file_format == "csv" and record["reward_fn"] == ""
            ):
                assert "reward_fn" not in item
            else:
                assert item["reward_fn"] == record["reward_fn"]
    assert loader.get_next_item() is None


@pytest.mark.parametrize("missing_value", [None, float("nan"), pd.NA])
def test_torch_data_loader_omits_pandas_missing_reward_fn(torch_data_loader_config, missing_value):
    loader = TorchDataLoader(torch_data_loader_config)
    loader._dataset.dataframe = pd.DataFrame(
        [{"prompt": "missing", "reward_fn": missing_value, "label": None}], dtype=object
    )

    item = loader.get_next_item()

    assert "reward_fn" not in item
    assert "label" in item
    assert item["label"] is None


@pytest.mark.parametrize("reward_fn", [["math_with_format"], [], {"name": "math_with_format"}])
def test_torch_data_loader_keeps_non_scalar_reward_fn(
    torch_data_loader_config, tmp_path, reward_fn
):
    data_path = tmp_path / "invalid-reward.jsonl"
    data_path.write_text(
        json.dumps({"prompt": "invalid", "reward_fn": reward_fn}) + "\n", encoding="utf-8"
    )
    torch_data_loader_config.files = [str(data_path)]

    item = TorchDataLoader(torch_data_loader_config).get_next_item()

    assert item["reward_fn"] == reward_fn


@pytest.mark.parametrize("file_format", ["jsonl", "parquet"])
def test_mixed_answer_formats_reach_math_judges_unchanged(
    torch_data_loader_config, tmp_path, file_format
):
    prompt = [{"role": "user", "content": "Return the expected number."}]
    records = [
        {"prompt": prompt, "label": "42", "reward_fn": "tinker_aime_scalar_match"},
        {"prompt": prompt, "ground_truth": "43"},
        {"prompt": prompt, "reward_model": {"ground_truth": "44"}},
    ]
    if file_format == "jsonl":
        data_path = tmp_path / "mixed-answers.jsonl"
        data_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        data_files = [str(data_path)]
    else:
        # Separate schemas acquire NaN cells only when the frames are concatenated.
        data_files = []
        for index, record in enumerate(records):
            data_path = tmp_path / f"answers-{index}.parquet"
            pd.DataFrame([record]).to_parquet(data_path, index=False)
            data_files.append(str(data_path))
    torch_data_loader_config.files = data_files
    loader = TorchDataLoader(torch_data_loader_config)

    for index, expected in enumerate(("42", "43", "44")):
        item = loader.get_next_item()
        assert item is not None
        if index == 0:
            assert item["label"] == expected
            assert item.get("ground_truth") is None
            assert item.get("reward_model") is None
        elif index == 1:
            assert item.get("label") is None
            assert item["ground_truth"] == expected
            assert item.get("reward_model") is None
        else:
            assert item.get("label") is None
            assert item.get("ground_truth") is None
            assert item["reward_model"]["ground_truth"] == expected
        assert (
            asyncio.run(
                math_reward(
                    item,
                    {"response": rf"\boxed{{{expected}}}"},
                    judge_mode="tinker_aime_scalar_match",
                )
            )
            == 1.0
        )
    assert loader.get_next_item() is None


@pytest.mark.parametrize("missing_value", [None, float("nan"), pd.NA])
def test_pandas_missing_answer_fields_become_none(torch_data_loader_config, missing_value):
    loader = TorchDataLoader(torch_data_loader_config)
    loader._dataset.dataframe = pd.DataFrame(
        [
            {
                "prompt": "missing",
                "label": missing_value,
                "ground_truth": missing_value,
                "reward_model": missing_value,
            }
        ],
        dtype=object,
    )

    item = loader.get_next_item()

    for field in ("label", "ground_truth", "reward_model"):
        assert field in item
        assert item[field] is None


def test_answer_cleanup_preserves_values_and_absent_fields(torch_data_loader_config):
    loader = TorchDataLoader(torch_data_loader_config)
    for value in (0, "", [], ["42"], {}, {"ground_truth": "42"}):
        loader._dataset.dataframe = pd.DataFrame(
            [{"prompt": "present", "label": value}], dtype=object
        )
        item = loader._dataset[0]
        assert item["label"] == value
        assert "ground_truth" not in item
        assert "reward_model" not in item

    loader._dataset.dataframe = pd.DataFrame([{"prompt": "absent"}])
    item = loader._dataset[0]
    for field in ("label", "ground_truth", "reward_model"):
        assert field not in item
