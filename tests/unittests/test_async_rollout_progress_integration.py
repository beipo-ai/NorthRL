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

"""Continuous progress follows outstanding rollouts across input and pool batches."""

from typing import Any

import pytest
from omegaconf import OmegaConf

from northrl.data_loader.data_loader import SequentialDataLoader
from northrl.northrl_types import Trajectory
from northrl.trajectory_pool import TrajectoryPoolInstance


class _FiniteLoader(SequentialDataLoader):
    def __init__(self, *, keep_batch_order: bool = False, is_validate: bool = False):
        self._cursor = 0
        self._items = [{"prompt": f"prompt-{index}"} for index in range(4)]
        super().__init__(
            OmegaConf.create(
                {"batch_size": 2, "rollout_repeat_n": 2, "keep_batch_order": keep_batch_order}
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


def _claim(loader: _FiniteLoader) -> dict[str, Any]:
    item = loader.get_next_item()
    assert item is not None
    return item


def _result(item: dict[str, Any], *, segments: int = 1, status: str = "success") -> Trajectory:
    return Trajectory(
        segments=[{"tokens": [1, 2, index], "loss_mask": [0, 1, 1]} for index in range(segments)],
        reward=1.0,
        status=status,
        extra_fields={**item, "model_tag": "default"},
    )


def _assert_rollouts(loader, *, finished: int, active: int, queued: int) -> None:
    assert loader.get_continuous_rollout_progress() == {
        "finished_total": finished,
        "in_progress": active,
        "queued": queued,
    }


def _assert_pool(pool, *, incomplete: int, ready: int, segments: int) -> None:
    counts = pool.get_progress_info()
    assert counts["incomplete_groups"] == incomplete
    assert counts["ready_groups"] == ready
    assert counts["ready_segments"] == segments
    assert counts["groups_per_batch"] == 2


def test_completion_crosses_input_batches_and_survives_pool_consumption():
    loader = _FiniteLoader()
    pool = _pool(loader)
    _assert_rollouts(loader, finished=0, active=0, queued=0)
    first_batch = [_result(_claim(loader), segments=n) for n in (3, 1, 0, 4)]
    next_result = _result(_claim(loader), segments=2)
    _assert_rollouts(loader, finished=0, active=5, queued=3)

    assert pool.put_trajectory(first_batch[0]) == "success"
    assert pool.put_trajectory(first_batch[0]) == "success"
    _assert_rollouts(loader, finished=1, active=4, queued=3)
    assert pool.put_trajectory(next_result) == "success"
    _assert_pool(pool, incomplete=2, ready=0, segments=0)

    assert pool.put_trajectory(first_batch[1]) == "success"
    _assert_pool(pool, incomplete=1, ready=1, segments=4)
    assert pool.get_trajectories() is None
    for result in first_batch[2:]:
        assert pool.put_trajectory(result) == "success"
    _assert_rollouts(loader, finished=5, active=0, queued=3)
    _assert_pool(pool, incomplete=1, ready=2, segments=8)

    assert pool.get_trajectories() == first_batch
    _assert_pool(pool, incomplete=1, ready=0, segments=0)
    # A consumed group's repeated delivery is accepted idempotently by the pool.
    assert pool.put_trajectory(first_batch[0]) == "success"
    _assert_rollouts(loader, finished=5, active=0, queued=3)
    _assert_pool(pool, incomplete=1, ready=0, segments=0)

    rest = [_result(_claim(loader)) for _ in range(3)]
    assert loader.get_next_item() is None
    _assert_rollouts(loader, finished=5, active=3, queued=0)
    for result in rest:
        assert pool.put_trajectory(result) == "success"
    _assert_rollouts(loader, finished=8, active=0, queued=0)
    _assert_pool(pool, incomplete=0, ready=2, segments=5)
    assert pool.get_trajectories() == [next_result, *rest]
    _assert_rollouts(loader, finished=8, active=0, queued=0)
    _assert_pool(pool, incomplete=0, ready=0, segments=0)


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_terminal_failures_and_zero_segments_still_complete_rollouts(status):
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
    _assert_rollouts(loader, finished=4, active=0, queued=0)
    _assert_pool(pool, incomplete=0, ready=2, segments=5)
    assert pool.get_trajectories() == results
    _assert_rollouts(loader, finished=4, active=0, queued=0)


def test_rejected_pool_result_keeps_its_rollout_in_progress():
    loader = _FiniteLoader()
    pool = _pool(loader)
    item = _claim(loader)
    malformed = _result(item)
    malformed["expected_rollouts"] = 3
    with pytest.raises(ValueError, match="expected_rollouts"):
        pool.put_trajectory(malformed)
    _assert_rollouts(loader, finished=0, active=1, queued=3)
    assert pool.put_trajectory(_result(item)) == "success"
    _assert_rollouts(loader, finished=1, active=0, queued=3)


def test_reset_discards_prior_active_ids_and_cumulative_progress():
    loader = _FiniteLoader()
    pool = _pool(loader)
    completed = _result(_claim(loader))
    stale = _result(_claim(loader))
    assert pool.put_trajectory(completed) == "success"
    loader.reset()
    _assert_rollouts(loader, finished=0, active=0, queued=0)
    fresh = _result(_claim(loader))
    assert fresh["rollout_id"] != stale["rollout_id"]
    assert pool.put_trajectory(stale) == "success"
    _assert_rollouts(loader, finished=0, active=1, queued=3)
    assert pool.put_trajectory(fresh) == "success"
    _assert_rollouts(loader, finished=1, active=0, queued=3)


def test_resume_skips_do_not_become_queued_or_finished_rollouts():
    loader = _FiniteLoader()
    loader.skip_batches(1)
    _assert_rollouts(loader, finished=0, active=0, queued=0)
    item = _claim(loader)
    assert item["prompt"] == "prompt-2"
    _assert_rollouts(loader, finished=0, active=1, queued=3)


@pytest.mark.parametrize(("keep_batch_order", "is_validate"), [(True, False), (False, True)])
def test_ordered_and_validation_loaders_do_not_report_continuous_progress(
    keep_batch_order, is_validate
):
    loader = _FiniteLoader(keep_batch_order=keep_batch_order, is_validate=is_validate)
    assert loader.get_continuous_rollout_progress() is None
    result = _result(_claim(loader))
    loader.record_rollout_result(result)
    assert loader.get_continuous_rollout_progress() is None


def test_result_callback_and_snapshot_do_not_wait_for_input_loading(monkeypatch):
    loader = _FiniteLoader()
    result = _result(_claim(loader))

    class UnavailableLock:
        def __enter__(self):
            raise AssertionError("Continuous progress must not acquire the data loader lock")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(loader, "_lock", UnavailableLock())
    loader.record_rollout_result(result)
    _assert_rollouts(loader, finished=1, active=0, queued=3)
