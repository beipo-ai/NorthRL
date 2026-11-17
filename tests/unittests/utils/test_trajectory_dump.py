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

"""Complete-trajectory dumps preserve outcome and sequence boundaries on replay."""

import copy
import json
from array import array

import pytest
import torch
from omegaconf import OmegaConf

from northrl.mock import MockRolloutWorker
from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPool
from northrl.utils.async_utils import run_coro
from northrl.utils.data_dumper import DataDumper
from northrl.utils.trajectory_sequences import iter_trajectory_sequences


def _worker(config, path, dump_format):
    config = copy.deepcopy(config)
    config.trajectory_load_path = str(path)
    config.trajectory_format = dump_format
    worker = MockRolloutWorker(config)
    worker._mock_delay = 0
    return worker


def _write(path, rows, dump_format, *, format_version=None):
    data = {"trajectories": rows}
    if format_version is not None:
        data["format_version"] = format_version
    if dump_format == "pt":
        torch.save(data, path)
    else:
        path.write_text(json.dumps(data) + "\n")


def _grouped_pool(repeat=2):
    pool = TrajectoryPool(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": repeat,
                "groups_per_batch": 1,
                "batch_size": repeat,
                "need_sampling": False,
            }
        )
    )
    pool.set_module_references(None, None)
    return pool


@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
@pytest.mark.parametrize("packed", [False, True])
def test_dump_load_replay_preserves_multisegment_and_successful_empty_trajectory(
    tmp_path, rollout_worker_config, dump_format, packed
):
    identity = {
        "group_id": "saved-group",
        "expected_rollouts": 2,
        "model_tag": "actor",
        "task_id": "saved-task",
    }
    segments = [
        {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            **{
                "old_log_probs": torch.tensor([0.0, -0.25, -0.5]),
                "sampling_mask": [1, 1],
                "correction_weights": [0.0, 0.5, 0.75],
                "moe_topk_indices_ref": {"shards": [{"offset": torch.tensor(3)}]},
            },
        },
        {"tokens": [4, 5], "loss_mask": [0, 1], **{"old_log_probs": [0.0, -0.75]}},
        {"tokens": [6, 7], "loss_mask": [0, 1], **{"old_log_probs": [0.0, -0.125]}},
    ]
    if packed:
        for segment in segments:
            for field, code in (("tokens", "q"), ("loss_mask", "b"), ("old_log_probs", "d")):
                segment[field] = array(code, segment[field])
    trajectories = [
        Trajectory(
            segments=segments,
            reward=1.0,
            extra_fields={**identity, "run_id": 0, "rollout_id": "saved-0"},
        ),
        Trajectory(
            segments=[],
            reward=0.25,
            extra_fields={**identity, "run_id": 1, "rollout_id": "saved-1"},
        ),
    ]
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": dump_format,
                    "remote_api_dump_options": {"prepared_trajectories": True},
                }
            }
        }
    )
    dumper.dump_prepared_trajectories(
        7, trajectories, extra_info={"nested": [{"tensor": torch.tensor([2, 3])}]}
    )
    path = tmp_path / "prepared_trajectories" / f"step_000007.{dump_format}"
    data = torch.load(path) if dump_format == "pt" else json.loads(path.read_text())
    assert data["format_version"] == 3
    assert data["num_trajectories"] == 2
    assert data["num_sequences"] == data["num_segments"] == 3
    assert [row["reward"] for row in data["trajectories"]] == [1.0, 0.25]
    assert all(
        "reward" not in segment and "is_val" not in segment
        for row in data["trajectories"]
        for segment in row["segments"]
    )
    if dump_format == "jsonl":
        assert data["extra_info"]["nested"] == [{"tensor": [2, 3]}]
    worker = _worker(rollout_worker_config, path, dump_format)
    assert [
        sum(1 for _ in iter_trajectory_sequences(trajectory))
        for trajectory in worker._loaded_trajectories
    ] == [3, 0]
    worker._trajectory_pool = pool = _grouped_pool()
    run_coro(worker.execute_rollout({"expected_rollouts": 2, "group_id": "outer"}))
    restored = pool.get_trajectories("actor")
    assert [trajectory.reward for trajectory in restored] == [1.0, 0.25]
    assert all(
        trajectory.status == "success" and not trajectory["group_skipped"]
        for trajectory in restored
    )
    assert [segment["tokens"] for segment in restored[0].segments] == [[1, 2, 3], [4, 5], [6, 7]]
    assert [segment["loss_mask"] for segment in restored[0].segments] == [[0, 1, 1], [0, 1], [0, 1]]
    assert restored[0].segments[0]["old_log_probs"] == [0.0, -0.25, -0.5]
    assert restored[0].segments[0]["moe_topk_indices_ref"] == {"shards": [{"offset": 3}]}
    assert restored[0].segments[0]["correction_weights"] == [0.0, 0.5, 0.75]
    assert restored[0]["task_id"] == "saved-task"
    assert restored[1].segments == []


def _legacy_row(run, token=2):
    return {
        "tokens": [1, token],
        "loss_mask": [0, 1],
        "reward": float(run),
        "is_val": False,
        "extra_fields": {
            "group_id": "old-group",
            "run_id": run,
            "rollout_id": f"old-{run}",
            "old_log_probs": [0.0, -0.5],
        },
    }


@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
def test_legacy_flat_segments_are_restored_by_group_and_run(
    tmp_path, rollout_worker_config, dump_format
):
    rows = [_legacy_row(0, 2), _legacy_row(1, 4), _legacy_row(0, 3)]
    path = tmp_path / f"legacy.{dump_format}"
    if dump_format == "jsonl":
        # A legacy execution may have its rows on separate lines of the same file.
        path.write_text("".join(json.dumps({"trajectories": [row]}) + "\n" for row in rows))
    else:
        _write(path, rows, dump_format)
    worker = _worker(rollout_worker_config, path, dump_format)
    assert len(worker._loaded_trajectories) == 2
    assert [
        sum(1 for _ in iter_trajectory_sequences(trajectory))
        for trajectory in worker._loaded_trajectories
    ] == [2, 1]
    assert [trajectory.reward for trajectory in worker._loaded_trajectories] == [0.0, 1.0]
    worker._trajectory_pool = pool = _grouped_pool()
    run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    restored = pool.get_trajectories("actor")
    assert [segment["tokens"] for segment in restored[0].segments] == [[1, 2], [1, 3]]
    assert restored[1].segments is None
    assert restored[1].tokens == [1, 4]


@pytest.mark.parametrize(
    "conflict,message",
    [
        ("reward", "final outcome"),
        ("rollout_id", "multiple rollout identities"),
        ("e2b_rollout_id", "multiple rollout identities"),
        ("missing_group", "require group_id"),
    ],
)
def test_invalid_legacy_dump_raises_without_falling_back_to_synthetic(
    tmp_path, rollout_worker_config, conflict, message
):
    rows = [_legacy_row(0), _legacy_row(0, 3), _legacy_row(1)]
    if conflict == "reward":
        rows[1]["reward"] = 0.5
    elif conflict == "missing_group":
        del rows[1]["extra_fields"]["group_id"]
    else:
        rows[1]["extra_fields"][conflict] = "different-rollout"
    path = tmp_path / "legacy.jsonl"
    _write(path, rows, "jsonl")
    with pytest.raises(ValueError, match=message):
        _worker(rollout_worker_config, path, "jsonl")


def test_pt_nested_tensor_segment_data_is_normalized_recursively(tmp_path, rollout_worker_config):
    row = {
        "segments": [
            {
                "tokens": torch.tensor([1, 2]),
                "loss_mask": torch.tensor([0, 1]),
                "extra_fields": {
                    "old_log_probs": torch.tensor([0.0, -0.5]),
                    "moe_topk_indices_ref": [{"ids": torch.tensor([3, 4])}],
                },
            }
        ],
        "reward": torch.tensor(0.5),
        "is_val": False,
        "extra_fields": {"group_id": "nested", "run_id": 0},
    }
    path = tmp_path / "nested.pt"
    _write(path, [row], "pt", format_version=2)
    [trajectory] = _worker(rollout_worker_config, path, "pt")._loaded_trajectories
    assert trajectory.reward == 0.5
    [segment] = trajectory.segments
    assert segment["tokens"] == [1, 2] and segment["loss_mask"] == [0, 1]
    assert segment["old_log_probs"] == [0.0, -0.5]
    assert segment["moe_topk_indices_ref"] == [{"ids": [3, 4]}]


@pytest.mark.parametrize("contents", ["", "{}\n", "not json"])
def test_empty_or_malformed_replay_file_does_not_enable_synthetic_generation(
    tmp_path, rollout_worker_config, contents
):
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError):
        _worker(rollout_worker_config, path, "jsonl")


def test_missing_replay_file_is_reported(tmp_path, rollout_worker_config):
    with pytest.raises(FileNotFoundError):
        _worker(rollout_worker_config, tmp_path / "missing.jsonl", "jsonl")


@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
def test_mixed_dump_preserves_ordinary_flat_shape_and_distinct_parent_counts(
    tmp_path, rollout_worker_config, dump_format
):
    identity = {"group_id": "mixed", "expected_rollouts": 3, "model_tag": "actor"}
    ordinary = Trajectory(
        tokens=[1, 2],
        loss_mask=[0, 1],
        reward=0.75,
        extra_fields={**identity, "run_id": 0, "rollout_id": "plain", "old_log_probs": [0.0, -0.5]},
    )
    multiple = Trajectory(
        segments=[
            {"tokens": [3, 4 + index], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.25]}
            for index in range(3)
        ],
        reward=1.0,
        extra_fields={**identity, "run_id": 1, "rollout_id": "multi"},
    )
    empty = Trajectory(
        segments=[], reward=0.25, extra_fields={**identity, "run_id": 2, "rollout_id": "empty"}
    )
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": dump_format,
                    "remote_api_dump_options": {"prepared_trajectories": True},
                }
            }
        }
    )
    dumper.dump_prepared_trajectories(1, [ordinary, multiple, empty])
    path = tmp_path / "prepared_trajectories" / f"step_000001.{dump_format}"
    data = torch.load(path) if dump_format == "pt" else json.loads(path.read_text())
    assert data["format_version"] == 3
    assert data["num_trajectories"] == 3
    assert data["num_sequences"] == data["num_segments"] == 4
    ordinary_row, multiple_row, empty_row = data["trajectories"]
    assert "segments" not in ordinary_row
    assert ordinary_row["tokens"] == [1, 2]
    assert "tokens" not in multiple_row
    assert all(
        "extra_fields" not in segment and "reward" not in segment
        for segment in multiple_row["segments"]
    )
    assert empty_row["segments"] == [] and empty_row["reward"] == 0.25
    assert ordinary.segments is None
    worker = _worker(rollout_worker_config, path, dump_format)
    restored_ordinary = worker._loaded_trajectories[0]
    assert restored_ordinary.segments is None
    assert restored_ordinary.tokens == [1, 2]
    worker._trajectory_pool = pool = _grouped_pool(repeat=3)
    run_coro(worker.execute_rollout({"expected_rollouts": 3}))
    restored = pool.get_trajectories("actor")
    assert [trajectory.reward for trajectory in restored] == [0.75, 1.0, 0.25]
    assert restored[0].segments is None
    assert restored[0]["old_log_probs"] == [0.0, -0.5]
    assert len(restored[1].segments) == 3
    assert restored[2].segments == []


def test_current_flat_dump_does_not_merge_duplicate_complete_trajectories(
    tmp_path, rollout_worker_config
):
    path = tmp_path / "current.jsonl"
    _write(path, [_legacy_row(0), _legacy_row(0, 3), _legacy_row(1)], "jsonl", format_version=3)
    worker = _worker(rollout_worker_config, path, "jsonl")
    assert len(worker._loaded_trajectories) == 3
    assert all(trajectory.segments is None for trajectory in worker._loaded_trajectories)
    worker._trajectory_pool = _grouped_pool()
    with pytest.raises(ValueError, match="duplicate trajectories"):
        run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    assert worker._trajectory_pool.is_empty()


@pytest.mark.parametrize("invalid_reward", [True, "1"])
def test_dump_replay_does_not_coerce_invalid_rewards_into_valid_outcomes(
    tmp_path, rollout_worker_config, invalid_reward
):
    identity = {"group_id": "invalid", "expected_rollouts": 2, "model_tag": "actor"}
    trajectories = [
        Trajectory(
            tokens=[1, 2],
            loss_mask=[0, 1],
            reward=reward,
            extra_fields={**identity, "run_id": run, "rollout_id": f"invalid-{run}"},
        )
        for run, reward in enumerate([1.0, invalid_reward])
    ]
    initial_pool = _grouped_pool()
    for trajectory in trajectories:
        initial_pool.put_trajectory(trajectory)
    trajectories = initial_pool.get_trajectories("actor")
    assert [trajectory["group_skipped"] for trajectory in trajectories] == [False, True]
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": "jsonl",
                    "remote_api_dump_options": {"prepared_trajectories": True},
                }
            }
        }
    )
    dumper.dump_prepared_trajectories(0, trajectories)
    worker = _worker(
        rollout_worker_config, tmp_path / "prepared_trajectories" / "step_000000.jsonl", "jsonl"
    )
    worker._trajectory_pool = pool = _grouped_pool()
    run_coro(worker.execute_rollout({"expected_rollouts": 2}))
    restored = pool.get_trajectories("actor")
    assert type(restored[1].reward) is type(invalid_reward)
    assert [trajectory["group_skipped"] for trajectory in restored] == [False, True]
