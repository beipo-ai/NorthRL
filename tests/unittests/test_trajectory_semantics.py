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

"""Direct trajectory fields, nested sequence ownership and complete batch rows."""

import pickle

import pytest
import torch

import northrl
import northrl.northrl_types as types_module
from northrl.northrl_types import Batch, Trajectory
from northrl.utils.trajectory_sequences import iter_training_sequences, iter_trajectory_sequences


def _sequence(token: int = 12) -> dict:
    return {"tokens": [11, token], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.5]}


def _metadata(rollout_id: str = "r0") -> dict:
    return {"model_tag": "actor", "temperature": 0.7, "rollout_id": rollout_id}


def test_ordinary_trajectory_stores_sequence_fields_without_wrapping() -> None:
    tokens, mask = [11, 12], [0, 1]
    extra = {"old_log_probs": [0.0, -0.5], "task_id": "t"}
    trajectory = Trajectory(tokens, mask, 0.75, True, extra)
    assert trajectory.__dict__["tokens"] is tokens
    assert trajectory.__dict__["loss_mask"] is mask
    assert trajectory.extra_fields is extra
    assert trajectory.segments is None
    assert trajectory.reward == 0.75 and trajectory.is_val is True
    assert trajectory.status == "success"
    assert list(iter_trajectory_sequences(trajectory)) == [trajectory]
    assert next(iter_training_sequences([trajectory])) is trajectory
    assert not hasattr(types_module, "Segment")
    assert not hasattr(types_module, "flatten_trajectories")
    assert not hasattr(northrl, "Segment")
    assert not hasattr(northrl, "flatten_trajectories")
    assert not hasattr(Batch, "from_segments")
    assert not hasattr(Batch, "to_segments")


def test_ordinary_dictionary_and_direct_field_updates_remain_compatible() -> None:
    trajectory = Trajectory([11, 12], [0, 1], 0.5)
    trajectory.tokens = [21, 22]
    trajectory["loss_mask"] = [0, 0]
    trajectory["reward"] = 1.0
    trajectory["advantage"] = 0.25
    assert trajectory["tokens"] == [21, 22]
    assert trajectory.loss_mask == [0, 0]
    assert trajectory.get("reward") == 1.0
    assert trajectory.get("advantage") == 0.25
    assert trajectory.get("missing", "fallback") == "fallback"
    assert "advantage" in trajectory and "missing" not in trajectory
    assert {"tokens", "loss_mask", "reward", "is_val", "advantage"} <= set(trajectory.keys())
    assert dict(trajectory.items())["tokens"] == [21, 22]
    with pytest.raises(KeyError):
        _ = trajectory["missing"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": [], "loss_mask": [], "segments": []},
        {"tokens": [1]},
        {"loss_mask": [0]},
        {"segments": [{"tokens": [1]}]},
    ],
)
def test_invalid_or_ambiguous_sequence_forms_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Trajectory(**kwargs)


def test_nested_trajectory_preserves_flat_dicts_and_parent_outcome() -> None:
    sequences = [_sequence(12), _sequence(13)]
    extra = {"group_id": "g", "score": {"correct": 1.0}}
    trajectory = Trajectory(segments=sequences, reward=0.75, extra_fields=extra)
    assert trajectory.segments is sequences
    assert trajectory.tokens is None and trajectory.loss_mask is None
    assert trajectory.extra_fields is extra
    assert list(iter_trajectory_sequences(trajectory)) == sequences
    assert next(iter_trajectory_sequences(trajectory)) is sequences[0]
    assert all("reward" not in sequence and "status" not in sequence for sequence in sequences)
    assert all("extra_fields" not in sequence for sequence in sequences)
    assert dict(trajectory.items())["segments"] is sequences


def test_training_iteration_broadcasts_identity_advantage_and_preserves_sequence_metadata() -> None:
    sequences = [_sequence(12), _sequence(13)]
    identity = {
        "group_id": "g",
        "run_id": 0,
        "rollout_id": "r0",
        "expected_rollouts": 2,
        "task_id": "t",
        "model_tag": "actor",
    }
    sequences[0].update({key: "stale" for key in identity})
    sequences[0]["advantage"] = -99.0
    trajectory = Trajectory(
        segments=sequences,
        reward=0.75,
        extra_fields={
            **identity,
            "advantage": 0.5,
            "old_log_probs": [0.0, -99.0],
            "score": {"correct": 1.0},
            "reward": 99.0,
            "status": "failed",
            "is_val": True,
            "group_skipped": False,
        },
    )
    prepared = list(iter_training_sequences([trajectory]))
    assert prepared[0] is sequences[0] and prepared[1] is sequences[1]
    for sequence in sequences:
        for key, value in identity.items():
            assert sequence[key] == value
        assert sequence["advantage"] == 0.5
        assert sequence["old_log_probs"] == [0.0, -0.5]
        assert sequence["score"] == {"correct": 1.0}
        assert not {"reward", "status", "is_val", "group_skipped"}.intersection(sequence)
    assert trajectory.reward == 0.75 and trajectory.status == "success"


def test_raw_iteration_does_not_filter_or_inject_parent_metadata() -> None:
    sequence = _sequence()
    trajectory = Trajectory(segments=[sequence], status="failed", extra_fields={"group_id": "g"})
    assert list(iter_trajectory_sequences(trajectory)) == [sequence]
    assert "group_id" not in sequence
    assert list(iter_training_sequences([trajectory])) == []


def test_training_iteration_skips_failed_groups_and_keeps_original_order() -> None:
    ordinary = Trajectory([1, 2], [0, 1], 0.0)
    sequence = _sequence()
    trajectories = [
        ordinary,
        Trajectory([3], [0], reward=1.0, status="failed"),
        Trajectory(segments=[_sequence()], reward=None, status="skipped"),
        Trajectory([4], [0], reward=1.0, extra_fields={"group_skipped": True}),
        Trajectory(segments=[], reward=0.5),
        Trajectory(segments=[sequence], reward=1.0),
    ]
    prepared = list(iter_training_sequences(trajectories))
    assert len(prepared) == 2
    assert prepared[0] is ordinary and prepared[1] is sequence
    assert trajectories[4].reward == 0.5


@pytest.mark.parametrize(
    "trajectory", [Trajectory(segments=[], reward=1.0), Trajectory(status="skipped")]
)
def test_zero_sequence_outcomes_have_no_training_sequences(trajectory: Trajectory) -> None:
    assert list(iter_trajectory_sequences(trajectory)) == []
    assert list(iter_training_sequences([trajectory])) == []


def test_legacy_flat_pickle_restores_direct_fields() -> None:
    legacy = object.__new__(Trajectory)
    legacy.__dict__.update(
        tokens=[11, 12],
        loss_mask=[0, 1],
        reward=0.75,
        is_val=True,
        extra_fields={"task_id": "t", "old_log_probs": [0.0, -0.5]},
    )
    restored = pickle.loads(pickle.dumps(legacy, protocol=0))
    assert restored.__dict__["tokens"] == [11, 12]
    assert restored.segments is None
    assert restored.reward == 0.75 and restored.is_val is True
    assert restored.status == "success"
    assert restored["old_log_probs"] == [0.0, -0.5]
    assert pickle.loads(pickle.dumps(restored)) == restored


def test_nested_pickle_roundtrip() -> None:
    trajectory = Trajectory(
        segments=[_sequence(12), _sequence(13)], reward=0.5, extra_fields={"group_id": "g"}
    )
    restored = pickle.loads(pickle.dumps(trajectory))
    assert restored == trajectory
    assert restored.tokens is None
    assert isinstance(restored.segments[0], dict)


def test_batch_counts_complete_parent_rows_and_roundtrips_mixed_forms() -> None:
    trajectories = [
        Trajectory([1, 2], [0, 1], 0.0, extra_fields=_metadata("r0")),
        Trajectory(
            segments=[_sequence(i) for i in range(3)], reward=1.0, extra_fields=_metadata("r1")
        ),
        Trajectory(segments=[], reward=None, status="failed", extra_fields=_metadata("r2")),
    ]
    batch = Batch.from_trajectories(trajectories)
    assert len(batch) == 3  # Complete executions, although there are four sequences.
    assert batch.values["reward"] == [0.0, 1.0, None]
    assert batch.values["segments"][0] is None
    assert batch.values["segments"][1] is trajectories[1].segments
    assert batch.to_trajectories() == trajectories


def test_batch_preserves_distinct_row_metadata() -> None:
    trajectories = [
        Trajectory([1], [0], 0.0, extra_fields={"model_tag": "a", "temperature": 0.3}),
        Trajectory([2], [0], 1.0, extra_fields={"model_tag": "b", "temperature": 0.9}),
    ]
    assert Batch.from_trajectories(trajectories).to_trajectories() == trajectories


def test_padding_counts_rows_but_cannot_be_restored_as_fake_trajectories() -> None:
    trajectory = Trajectory(
        segments=[_sequence(i) for i in range(4)], reward=0.5, extra_fields=_metadata()
    )
    batch = Batch.from_trajectories([trajectory]).pad_to_world_size(8)
    assert len(batch) == 8 and batch.pad_size == 7
    with pytest.raises(ValueError, match="unpad"):
        batch.to_trajectories()
    # Existing copy() loses pad_size; nullable padding must still be rejected.
    with pytest.raises(ValueError, match="unpad"):
        batch.copy().to_trajectories()
    batch.unpad()
    assert len(batch) == 1
    assert batch.to_trajectories() == [trajectory]


def test_generic_batch_remains_a_row_container() -> None:
    batch = Batch({"data": torch.tensor([1, 2, 3])}, {"batch_size": 3})
    assert len(batch) == 3
    assert batch.reorder([2, 0, 1]).values["data"].tolist() == [3, 1, 2]
    assert batch.to_dict()["batch_size"] == 3
    with pytest.raises(ValueError, match="trajectory sequence fields"):
        batch.to_trajectories()


def test_training_iterator_rejects_a_bare_sequence_without_parent() -> None:
    with pytest.raises(TypeError, match="complete Trajectory"):
        list(iter_training_sequences([_sequence()]))  # type: ignore[list-item]


def test_raw_iterator_rejects_a_bare_sequence_without_parent() -> None:
    with pytest.raises(TypeError, match="complete Trajectory"):
        list(iter_trajectory_sequences(_sequence()))  # type: ignore[arg-type]


def test_batch_roundtrips_real_mixed_sequence_metadata_without_inventing_missing_fields() -> None:
    ordinary_extra = {"old_log_probs": [0.0, -0.5], "rollout_id": "ordinary"}
    nested_extra = {"rollout_id": "nested", "score": {"correct": 0.0}}
    sequences = [_sequence(12), _sequence(13)]
    ordinary = Trajectory([11, 12], [0, 1], 0.75, extra_fields=ordinary_extra)
    nested = Trajectory(segments=sequences, reward=None, status="failed", extra_fields=nested_extra)
    batch = Batch.from_trajectories([ordinary, nested])
    assert len(batch) == 2
    assert set(batch.values) == {
        "tokens",
        "loss_mask",
        "segments",
        "reward",
        "is_val",
        "status",
        "extra_fields",
    }
    restored = batch.to_trajectories()
    assert restored == [ordinary, nested]
    assert restored[0].reward == 0.75 and restored[1].status == "failed"
    assert "old_log_probs" not in restored[1].extra_fields
    assert "score" not in restored[0].extra_fields
    assert "model_tag" not in restored[0].extra_fields
    assert restored[1].segments == sequences
    restored[0]["new_key"] = "restored-only"
    restored[1]["another_key"] = "restored-only"
    assert ordinary_extra == {"old_log_probs": [0.0, -0.5], "rollout_id": "ordinary"}
    assert nested_extra == {"rollout_id": "nested", "score": {"correct": 0.0}}
    assert batch.values["extra_fields"] == [ordinary_extra, nested_extra]
    assert all("rollout_id" not in sequence for sequence in sequences)


def test_batch_uniform_schema_keeps_legacy_flat_metadata_columns() -> None:
    trajectories = [
        Trajectory([1], [0], 0.0, extra_fields={"old_log_probs": [-0.1], **_metadata("r0")}),
        Trajectory([2], [0], 1.0, extra_fields={"old_log_probs": [-0.2], **_metadata("r1")}),
    ]
    batch = Batch.from_trajectories(trajectories)
    assert "extra_fields" not in batch.values
    assert batch.values["old_log_probs"] == [[-0.1], [-0.2]]
    assert batch.to_trajectories() == trajectories
