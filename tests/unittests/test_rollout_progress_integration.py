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

"""Exercise round progress across real loader claims and pool submissions."""

from typing import Any

import pytest
from omegaconf import OmegaConf

from northrl.data_loader.data_loader import SequentialDataLoader
from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPoolInstance


class _FiniteLoader(SequentialDataLoader):
    def __init__(self, *, keep_batch_order: bool = True, is_validate: bool = False):
        self._items = [{"prompt": f"prompt-{index}"} for index in range(4)]
        self._cursor = 0
        super().__init__(
            OmegaConf.create(
                {
                    "batch_size": 2,
                    "rollout_repeat_n": 2,
                    "keep_batch_order": keep_batch_order,
                }
            ),
            is_validate=is_validate,
        )

    def _fetch_batch_data(self) -> list[dict[str, Any]]:
        batch = self._items[self._cursor : self._cursor + self._batch_size]
        self._cursor += len(batch)
        return batch

    def _reset_iterator(self) -> None:
        self._cursor = 0

    def is_finished(self) -> bool:
        return self._cursor == len(self._items) and not self.can_return_item()


def _pool(loader: _FiniteLoader) -> TrajectoryPoolInstance:
    instance = TrajectoryPoolInstance(
        OmegaConf.create(
            {
                "key_list": ["group_id"],
                "group_size": 2,
                "groups_per_batch": 2,
                "need_sampling": False,
            }
        ),
        model_tag="default",
    )
    instance.set_module_references(loader, None, None)
    return instance


def _result(item: dict[str, Any], *, segments: int = 1, status: str = "success") -> Trajectory:
    return Trajectory(
        segments=[{"tokens": [1, 2, index], "loss_mask": [0, 1, 1]} for index in range(segments)],
        reward=1.0,
        status=status,
        extra_fields={**item, "model_tag": "default"},
    )


def _claim(loader: _FiniteLoader) -> dict[str, Any]:
    item = loader.get_next_item()
    assert item is not None
    return item


def test_round_progress_survives_pool_consumption_and_resets_on_release():
    loader = _FiniteLoader()
    pool = _pool(loader)
    assert loader.get_rollout_progress() is None

    first = _claim(loader)
    assert loader.get_rollout_progress() == {
        "total": 4,
        "finished": 0,
        "in_progress": 1,
        "queued": 3,
        "groups_total": 2,
        "groups_complete": 0,
        "ready_segments": 0,
    }
    first_result = _result(first, segments=3)
    assert pool.put_trajectory(first_result) == "success"
    partial = loader.get_rollout_progress()
    assert partial["finished"] == 1
    assert partial["in_progress"] == 0
    assert partial["groups_complete"] == 0
    assert partial["ready_segments"] == 0
    assert pool.get_progress_info()["ready_segments"] == 0

    # A retry of the same complete result must not increase either counter.
    assert pool.put_trajectory(first_result) == "success"
    assert loader.get_rollout_progress() == partial

    second_result = _result(_claim(loader), segments=2)
    assert pool.put_trajectory(second_result) == "success"
    assert loader.get_rollout_progress()["groups_complete"] == 1
    assert loader.get_rollout_progress()["ready_segments"] == 5
    assert pool.get_progress_info()["ready_segments"] == 5
    assert pool.get_trajectories() is None

    third_result = _result(_claim(loader), segments=0)
    fourth_result = _result(_claim(loader), segments=4)
    assert loader.get_next_item() is None
    assert loader.get_rollout_progress()["in_progress"] == 2
    assert pool.put_trajectory(third_result) == "success"
    assert pool.put_trajectory(fourth_result) == "success"
    completed = loader.get_rollout_progress()
    assert completed == {
        "total": 4,
        "finished": 4,
        "in_progress": 0,
        "queued": 0,
        "groups_total": 2,
        "groups_complete": 2,
        "ready_segments": 9,
    }

    assert pool.get_trajectories() == [
        first_result,
        second_result,
        third_result,
        fourth_result,
    ]
    assert pool.is_empty()
    assert pool.get_progress_info()["ready_segments"] == 0
    assert loader.get_rollout_progress() == completed

    loader.release_next_batch()
    new_round = loader.get_rollout_progress()
    assert new_round == {
        "total": 4,
        "finished": 0,
        "in_progress": 0,
        "queued": 4,
        "groups_total": 2,
        "groups_complete": 0,
        "ready_segments": 0,
    }
    # Old-result delivery after the trainer advances cannot complete a new task.
    assert pool.put_trajectory(first_result) == "success"
    assert loader.get_rollout_progress() == new_round
    assert pool.is_empty()
    assert _claim(loader)["rollout_id"] != first["rollout_id"]


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_terminal_failure_completes_group_but_excludes_its_segments(status):
    loader = _FiniteLoader()
    pool = _pool(loader)
    results = [
        _result(_claim(loader), segments=3),
        _result(_claim(loader), segments=0, status=status),
        _result(_claim(loader), segments=0),
        _result(_claim(loader), segments=2),
    ]
    for result in results:
        assert pool.put_trajectory(result) == "success"

    progress = loader.get_rollout_progress()
    assert progress["finished"] == 4
    assert progress["in_progress"] == progress["queued"] == 0
    assert progress["groups_complete"] == 2
    assert progress["ready_segments"] == pool.get_progress_info()["ready_segments"] == 5
    assert pool.get_trajectories() == results
    assert loader.get_rollout_progress() == progress


def test_rejected_pool_submission_does_not_mark_rollout_finished():
    loader = _FiniteLoader()
    pool = _pool(loader)
    item = _claim(loader)
    malformed = _result(item)
    malformed["expected_rollouts"] = 3
    before = loader.get_rollout_progress()

    with pytest.raises(ValueError, match="expected_rollouts"):
        pool.put_trajectory(malformed)
    assert loader.get_rollout_progress() == before

    assert pool.put_trajectory(_result(item)) == "success"
    assert loader.get_rollout_progress()["finished"] == 1


def test_loader_reset_clears_round_and_rejects_prior_execution_identity():
    loader = _FiniteLoader()
    pool = _pool(loader)
    old_result = _result(_claim(loader))
    assert pool.put_trajectory(old_result) == "success"

    loader.reset()
    assert loader.get_rollout_progress() is None
    fresh = _claim(loader)
    assert fresh["rollout_id"] != old_result["rollout_id"]
    before = loader.get_rollout_progress()
    assert pool.put_trajectory(old_result) == "success"
    assert loader.get_rollout_progress() == before


@pytest.mark.parametrize(("keep_batch_order", "is_validate"), [(False, False), (True, True)])
def test_unordered_and_validation_loaders_do_not_report_training_rounds(
    keep_batch_order, is_validate
):
    loader = _FiniteLoader(keep_batch_order=keep_batch_order, is_validate=is_validate)
    assert loader.get_rollout_progress() is None
    while (item := loader.get_next_item()) is not None:
        loader.record_rollout_result(_result(item))
        assert loader.get_rollout_progress() is None


def test_progress_callback_failure_does_not_fail_accepted_rollout(monkeypatch, caplog):
    loader = _FiniteLoader()
    pool = _pool(loader)
    results = [_result(_claim(loader)) for _ in range(4)]

    def unavailable_progress(trajectory):
        raise RuntimeError("progress observer unavailable")

    monkeypatch.setattr(loader, "record_rollout_result", unavailable_progress)
    for result in results:
        assert pool.put_trajectory(result) == "success"

    assert pool.get_trajectories() == results
    assert "Could not record rollout progress" in caplog.text
    assert "progress observer unavailable" in caplog.text


@pytest.mark.parametrize(("keep_batch_order", "is_validate"), [(False, False), (True, True)])
def test_disabled_round_tracking_does_not_acquire_loader_lock(
    keep_batch_order, is_validate, monkeypatch
):
    loader = _FiniteLoader(keep_batch_order=keep_batch_order, is_validate=is_validate)
    result = _result(_claim(loader))

    class UnavailableLock:
        def __enter__(self):
            raise AssertionError("Disabled progress tracking must not acquire the loader lock")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(loader, "_lock", UnavailableLock())
    loader.record_rollout_result(result)
    assert loader.get_rollout_progress() is None
