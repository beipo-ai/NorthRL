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

"""Tests for the asynchronous BaseTrainer lifecycle."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.trainer import base_trainer as base_trainer_module
from northrl.trainer.base_trainer import BaseTrainer
from northrl.utils.async_utils import await_coro
from northrl.utils.logging_utils import log_rollout_metrics


async def _current_thread_name() -> str:
    await asyncio.sleep(0)
    return threading.current_thread().name


class _SingleBatchPool:
    def __init__(self, batch_count: int = 1) -> None:
        self._returned_batch = False
        self._returned_batches = 0
        self._batch_count = batch_count

    def get_trajectories(self):
        if self._returned_batches >= self._batch_count:
            return None
        self._returned_batches += 1
        self._returned_batch = True
        return []

    def get_discarded_metrics(self):
        return {}


class _WeightSyncController:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.completed_steps: list[int | None] = []
        self.notifications: list[tuple[str, str, bool, bool, str | None, str | None]] = []

    def is_waiting_for_validation(self) -> bool:
        return False

    def train_worker_notify_weight_update(
        self,
        *,
        worker_name: str,
        model_tag: str,
        sampling_path: str | None,
        weight_version: str | None,
        weights_updated: bool = True,
        continue_training: bool = True,
        completed_step: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.completed_steps.append(completed_step)
        self.notifications.append(
            (
                worker_name,
                model_tag,
                weights_updated,
                continue_training,
                sampling_path,
                weight_version,
            )
        )


class _ValidationWeightSyncController(_WeightSyncController):
    def __init__(self, *, initially_waiting: bool = False) -> None:
        super().__init__()
        self.validation_pending = threading.Event()
        self.waiting_observed = threading.Event()
        self.validation_checks = 0
        if initially_waiting:
            self.validation_pending.set()

    def is_waiting_for_validation(self) -> bool:
        waiting = self.validation_pending.is_set()
        if waiting:
            self.validation_checks += 1
            # Observe a second poll so assertions cover a full wait iteration.
            if self.validation_checks >= 2:
                self.waiting_observed.set()
        return waiting

    def train_worker_notify_weight_update(self, **kwargs) -> None:
        super().train_worker_notify_weight_update(**kwargs)
        if len(self.notifications) == 1:
            self.validation_pending.set()


class _BlockingWeightSyncController(_WeightSyncController):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def train_worker_notify_weight_update(self, **kwargs) -> None:
        self.started.set()
        assert self.release.wait(timeout=5)
        super().train_worker_notify_weight_update(**kwargs)


class _ActivityTracker:
    def __init__(self) -> None:
        self.training_steps: list[int] = []
        self.experiment_posts: list[dict] = []
        self.flushed_steps: list[int] = []

    def track(self, module, work):
        del module, work
        return nullcontext()

    def experiment_logger_post(self, **kwargs) -> None:
        self.experiment_posts.append(kwargs)

    def experiment_logger_flush(self, step: int) -> None:
        self.flushed_steps.append(step)

    def set_training_step(self, step: int) -> None:
        self.training_steps.append(step)


class _ClaimTrackingActivityTracker(_ActivityTracker):
    def __init__(self) -> None:
        super().__init__()
        self.active_count = 0
        self.track_count = 0

    @contextmanager
    def track(self, module, work):
        del module, work
        self.active_count += 1
        self.track_count += 1
        try:
            yield
        finally:
            self.active_count -= 1

    def is_quiescent(self) -> bool:
        return self.active_count == 0


class _ClaimObservingPool(_SingleBatchPool):
    def __init__(self, activity_tracker: _ClaimTrackingActivityTracker) -> None:
        super().__init__()
        self._activity_tracker = activity_tracker
        self.claim_was_active: bool | None = None

    def get_trajectories(self):
        trajectories = super().get_trajectories()
        if trajectories is not None:
            self.claim_was_active = not self._activity_tracker.is_quiescent()
        return trajectories


class _AsyncTrainer(BaseTrainer):
    def __init__(
        self,
        algorithm_type="grpo",
        metrics: dict | None = None,
        total_steps: int = 5,
    ) -> None:
        super().__init__(
            OmegaConf.create(
                {
                    "training": {"total_steps": total_steps},
                    "algorithm": {"type": algorithm_type},
                }
            )
        )
        self.train_started = threading.Event()
        self.train_finished = threading.Event()
        self.trainer_thread_name: str | None = None
        self.sdk_thread_name: str | None = None
        self.training_complete = threading.Event()
        self._metrics = {"weaver/loss": 0.0} if metrics is None else metrics

    async def train(self, trajectories):
        assert trajectories == []
        self.trainer_thread_name = threading.current_thread().name
        self.train_started.set()

        # Keep the step in flight long enough for stop() to exercise graceful
        # draining instead of cancelling the trainer task.
        await asyncio.sleep(0.05)
        self.sdk_thread_name = await await_coro(_current_thread_name())
        self.train_finished.set()
        return dict(self._metrics), None, None

    async def _on_training_complete(self) -> None:
        self.training_complete.set()


def test_base_trainer_runs_on_dedicated_loop_and_stop_drains_inflight_step():
    trainer = _AsyncTrainer(
        metrics={
            "weaver/loss": 0.0,
            "training/window_shifted_tokens": 100.0,
            "training/loss_agg_denominator": 40.0,
        }
    )
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.train_started.wait(timeout=5)
    trainer.stop()

    assert trainer.training_complete.is_set()
    assert trainer.train_finished.is_set()
    assert trainer.trainer_thread_name == "northrl-trainer-loop"
    assert trainer.sdk_thread_name == "northrl-async-loop"
    assert trainer.get_train_step() == 1
    assert activity_tracker.training_steps == [1]
    assert activity_tracker.flushed_steps == [1]
    assert weight_sync.notifications == [("trainer", "actor", True, False, None, None)]
    assert weight_sync.completed_steps == [1]

    assert len(activity_tracker.experiment_posts) == 1
    post = activity_tracker.experiment_posts[0]
    assert post["backend"] == "wandb"
    assert post["step"] == 1
    assert post["data"]["training/train_step"] == 1
    assert "training/global_step" not in post["data"]
    assert post["data"]["weaver/loss"] == 0.0
    assert post["data"]["timing/trainer_wait_for_batch_seconds"] == 0.0
    assert post["data"]["timing_train/batch_training_seconds"] >= 0.0
    assert post["data"]["throughput/input_tokens_per_second"] == pytest.approx(
        100.0 / post["data"]["timing_train/batch_training_seconds"]
    )
    assert post["data"]["throughput/effective_tokens_per_second"] == pytest.approx(
        40.0 / post["data"]["timing_train/batch_training_seconds"]
    )


@pytest.mark.parametrize("sync_mode", ["sync", "batch-async", "fully-async", "no-sync"])
@pytest.mark.parametrize("skipped", [False, True])
def test_batch_wait_belongs_to_claimed_batch_and_excludes_training_sync_and_validation(
    monkeypatch, sync_mode, skipped
):
    trainer = _AsyncTrainer(total_steps=2)
    tracker = _ActivityTracker()
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    weight_sync.sync_mode = sync_mode
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(pool, weight_sync)
    now = [0.0]
    validation_pending = [True]
    claims = [0]
    monkeypatch.setattr(base_trainer_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(weight_sync, "is_waiting_for_validation", lambda: validation_pending[0])

    def claim():
        claims[0] += 1
        return None if claims[0] == 1 else []

    async def sleep(_delay):
        if validation_pending[0]:
            now[0] += 100.0
            validation_pending[0] = False
        else:
            now[0] += 4.0

    async def train(_trajectories):
        now[0] += 30.0
        return {"training/skipped_flag": int(skipped)}, None, None

    def sync(**kwargs):
        now[0] += 20.0
        validation_pending[0] = kwargs["completed_step"] == 1

    monkeypatch.setattr(pool, "get_trajectories", claim)
    monkeypatch.setattr(base_trainer_module, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr(weight_sync, "train_worker_notify_weight_update", sync)
    asyncio.run(trainer._main_loop())

    assert claims[0] == 3
    assert [post["step"] for post in tracker.experiment_posts] == [1, 2]
    assert [
        post["data"]["timing/trainer_wait_for_batch_seconds"] for post in tracker.experiment_posts
    ] == [4.0, 0.0]
    assert [
        post["data"]["timing_train/batch_training_seconds"] for post in tracker.experiment_posts
    ] == [
        30.0,
        30.0,
    ]
    assert all(
        key not in post["data"]
        for post in tracker.experiment_posts
        for key in ("timing/batch_release_to_claim_seconds", "timing/batch_claim_interval_seconds")
    )


def test_batch_wait_pauses_during_validation_and_retains_prior_data_wait(monkeypatch):
    trainer = _AsyncTrainer(total_steps=1)
    tracker = _ActivityTracker()
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(pool, weight_sync)
    now = [0.0]
    validation_pending = [False]
    claims = [0]
    monkeypatch.setattr(base_trainer_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(weight_sync, "is_waiting_for_validation", lambda: validation_pending[0])

    def claim():
        claims[0] += 1
        return [] if claims[0] == 3 else None

    async def sleep(_delay):
        if validation_pending[0]:
            now[0] += 100.0
            validation_pending[0] = False
        elif claims[0] == 1:
            now[0] += 2.0
            validation_pending[0] = True
        else:
            now[0] += 3.0

    async def train(_trajectories):
        now[0] += 1000.0
        return {}, None, None

    monkeypatch.setattr(pool, "get_trajectories", claim)
    monkeypatch.setattr(base_trainer_module, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(trainer, "train", train)
    asyncio.run(trainer._main_loop())

    assert len(tracker.experiment_posts) == 1
    assert tracker.experiment_posts[0]["data"]["timing/trainer_wait_for_batch_seconds"] == 5.0
    assert tracker.experiment_posts[0]["data"]["timing_train/batch_training_seconds"] == 1000.0


def test_stop_while_waiting_for_batch_does_not_emit_a_completed_step(monkeypatch):
    trainer = _AsyncTrainer(total_steps=1)
    tracker = _ActivityTracker()
    pool = _SingleBatchPool(batch_count=0)
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(pool, _WeightSyncController())

    async def sleep(_delay):
        trainer._stop_event.set()

    monkeypatch.setattr(base_trainer_module, "asyncio", SimpleNamespace(sleep=sleep))
    asyncio.run(trainer._main_loop())

    assert tracker.experiment_posts == []
    assert tracker.flushed_steps == []
    assert trainer.get_train_step() == 0


def test_base_trainer_is_active_before_claiming_a_batch():
    trainer = _AsyncTrainer(total_steps=1)
    activity_tracker = _ClaimTrackingActivityTracker()
    pool = _ClaimObservingPool(activity_tracker)
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, _WeightSyncController())  # type: ignore[arg-type]

    trainer.run()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert pool.claim_was_active is True


def test_base_trainer_logs_skipped_round_once_with_rollout_metrics():
    trainer = _AsyncTrainer(
        metrics={
            "trajectory/reward_mean": 0.25,
            "training/skipped_flag": 1,
        }
    )
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.train_started.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 1
    assert activity_tracker.training_steps == [1]
    assert activity_tracker.flushed_steps == [1]
    assert weight_sync.notifications == [("trainer", "actor", False, False, None, None)]

    assert len(activity_tracker.experiment_posts) == 1
    post = activity_tracker.experiment_posts[0]
    assert post["backend"] == "wandb"
    assert post["step"] == 1
    assert post["data"]["training/skipped_flag"] == 1
    assert "rollout/reward" not in post["data"]
    assert post["data"]["algorithm/trajectory/reward_mean"] == 0.25
    assert "trajectory/reward_mean" not in post["data"]
    assert post["data"]["timing/trainer_wait_for_batch_seconds"] == 0.0
    assert "weaver/loss" not in post["data"]


def test_resume_at_total_steps_does_not_consume_or_train_an_extra_round():
    trainer = _AsyncTrainer()
    trainer.set_train_step(5)
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 5
    assert not trainer.train_started.is_set()
    assert not pool._returned_batch  # pylint: disable=protected-access
    assert activity_tracker.training_steps == []
    assert activity_tracker.experiment_posts == []
    assert activity_tracker.flushed_steps == []
    assert weight_sync.notifications == []


def test_base_trainer_flushes_resumed_round_without_weight_sync():
    trainer = _AsyncTrainer(total_steps=8)
    trainer.set_train_step(7)
    weight_sync = _WeightSyncController(enabled=False)
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(_SingleBatchPool(), weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 8
    assert activity_tracker.flushed_steps == [8]
    assert activity_tracker.experiment_posts[0]["step"] == 8
    assert weight_sync.notifications == []


@pytest.mark.parametrize("resume_step", [190, 199, 200, 205])
def test_resumed_trainer_passes_global_completed_step_to_weight_sync(resume_step):
    trainer = _AsyncTrainer(total_steps=resume_step + 1)
    trainer.set_train_step(resume_step)
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)
    trainer.set_module_references(_SingleBatchPool(), weight_sync)

    trainer.run()
    try:
        assert trainer.training_complete.wait(timeout=5)
    finally:
        trainer.stop()

    assert weight_sync.completed_steps == [resume_step + 1]
    assert activity_tracker.training_steps == [resume_step + 1]
    assert activity_tracker.experiment_posts[0]["step"] == resume_step + 1
    assert trainer.get_train_step() == resume_step + 1


def test_base_trainer_allows_next_batch_until_the_final_round():
    trainer = _AsyncTrainer(total_steps=2)
    pool = _SingleBatchPool(batch_count=2)
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 2
    assert activity_tracker.flushed_steps == [1, 2]
    assert weight_sync.notifications == [
        ("trainer", "actor", True, True, None, None),
        ("trainer", "actor", True, False, None, None),
    ]


@pytest.mark.parametrize(
    ("algorithm_type", "sync_mode"),
    [
        ("grpo", "sync"),
        ("grpo", "batch-async"),
        ("grpo", "fully-async"),
        ("grpo", "no-sync"),
        ("sft", "sync"),
    ],
)
def test_completion_interval_spans_previous_sync_validation_and_batch_wait(
    monkeypatch, algorithm_type, sync_mode
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(base_trainer_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    sync_enabled = algorithm_type != "sft" and sync_mode != "no-sync"

    class _TimedPool(_SingleBatchPool):
        missed_batch = False

        def get_trajectories(self):
            if self._returned_batches == 1 and not self.missed_batch:
                self.missed_batch = True
                clock.now += 7.0
                return None
            return super().get_trajectories()

    class _TimedWeightSync(_WeightSyncController):
        validation_pending = False
        validation_polls = 0

        def train_worker_notify_weight_update(self, **kwargs):
            super().train_worker_notify_weight_update(**kwargs)
            if sync_enabled:
                # The current round's sync is after its training completion.
                clock.now += 3.0 if len(self.notifications) == 1 else 29.0
                self.validation_pending = len(self.notifications) == 1

        def is_waiting_for_validation(self):
            if not self.validation_pending:
                return False
            self.validation_polls += 1
            if self.validation_polls == 2:
                clock.now += 5.0
                self.validation_pending = False
            return True

    class _TimedTrainer(_AsyncTrainer):
        durations = iter((11.0, 13.0))

        async def train(self, trajectories):
            assert trajectories == []
            clock.now += next(self.durations)
            return {}, None, None

    trainer = _TimedTrainer(algorithm_type, total_steps=2)
    tracker = _ActivityTracker()
    weight_sync = _TimedWeightSync(enabled=sync_enabled)
    weight_sync.sync_mode = sync_mode
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(_TimedPool(batch_count=2), weight_sync)

    asyncio.run(trainer._main_loop())

    first, second = [post["data"] for post in tracker.experiment_posts]
    assert "timing/batch_completion_interval_seconds" not in first
    assert first["timing_train/batch_training_seconds"] == 11.0
    assert second["timing_train/batch_training_seconds"] == 13.0
    assert second["timing/batch_completion_interval_seconds"] == (
        3.0 + 5.0 + 7.0 + 13.0 if sync_enabled else 7.0 + 13.0
    )


def test_completion_interval_omits_first_resumed_round_and_counts_skipped_round(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(base_trainer_module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    class _TimedTrainer(_AsyncTrainer):
        rounds = iter(((11.0, True), (13.0, False), (17.0, False)))

        async def train(self, trajectories):
            assert trajectories == []
            duration, skipped = next(self.rounds)
            clock.now += duration
            return {"training/skipped_flag": int(skipped)}, None, None

    trainer = _TimedTrainer(total_steps=10)
    trainer.set_train_step(7)
    tracker = _ActivityTracker()
    weight_sync = _WeightSyncController()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(_SingleBatchPool(batch_count=3), weight_sync)

    asyncio.run(trainer._main_loop())

    first, second, third = [post["data"] for post in tracker.experiment_posts]
    assert [post["step"] for post in tracker.experiment_posts] == [8, 9, 10]
    assert first["training/skipped_flag"] == 1
    assert "timing/batch_completion_interval_seconds" not in first
    assert second["timing/batch_completion_interval_seconds"] == 13.0
    assert third["timing/batch_completion_interval_seconds"] == 17.0
    assert [notification[2] for notification in weight_sync.notifications] == [False, True, True]


def test_training_total_excludes_persistence_but_completion_interval_includes_it(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(base_trainer_module, "time", SimpleNamespace(monotonic=lambda: clock.now))

    class _PersistingTrainer(_AsyncTrainer):
        async def train(self, trajectories):
            assert trajectories == []
            clock.now += 11.0
            metrics = {
                "timing_train/batch_training_seconds": 11.0,
                "timing_weights/checkpoint_seconds": 50.0,
                "training/window_shifted_tokens": 110,
                "training/loss_agg_denominator": 22,
            }
            clock.now += 50.0
            return metrics, None, None

    trainer = _PersistingTrainer(total_steps=2)
    tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(_SingleBatchPool(batch_count=2), _WeightSyncController())

    asyncio.run(trainer._main_loop())

    first, second = [post["data"] for post in tracker.experiment_posts]
    assert first["timing_train/batch_training_seconds"] == 11.0
    assert second["timing_train/batch_training_seconds"] == 11.0
    assert "timing/batch_completion_interval_seconds" not in first
    assert second["timing/batch_completion_interval_seconds"] == 61.0
    assert second["throughput/input_tokens_per_second"] == 10.0
    assert second["throughput/effective_tokens_per_second"] == 2.0


def test_base_trainer_waits_for_validation_before_claiming_a_queued_batch():
    trainer = _AsyncTrainer(total_steps=2)
    pool = _SingleBatchPool(batch_count=2)
    weight_sync = _ValidationWeightSyncController()
    activity_tracker = _ClaimTrackingActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    try:
        assert weight_sync.waiting_observed.wait(timeout=5)
        assert trainer.get_train_step() == 1
        assert pool._returned_batches == 1  # pylint: disable=protected-access
        assert activity_tracker.is_quiescent()
        assert activity_tracker.track_count == 1
        assert activity_tracker.flushed_steps == []
        assert weight_sync.notifications == [("trainer", "actor", True, True, None, None)]

        weight_sync.validation_pending.clear()
        assert trainer.training_complete.wait(timeout=5)
    finally:
        weight_sync.validation_pending.clear()
        trainer.stop()

    assert trainer.get_train_step() == 2
    assert pool._returned_batches == 2  # pylint: disable=protected-access
    assert activity_tracker.track_count == 2
    assert activity_tracker.flushed_steps == [2]
    assert activity_tracker.is_quiescent()
    assert weight_sync.notifications == [
        ("trainer", "actor", True, True, None, None),
        ("trainer", "actor", True, False, None, None),
    ]


def test_base_trainer_stop_exits_while_validation_is_pending():
    trainer = _AsyncTrainer(total_steps=1)
    pool = _SingleBatchPool()
    weight_sync = _ValidationWeightSyncController(initially_waiting=True)
    activity_tracker = _ClaimTrackingActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    stop_thread = threading.Thread(target=trainer.stop, daemon=True)
    trainer.run()
    try:
        assert weight_sync.waiting_observed.wait(timeout=5)
        stop_thread.start()
        stop_thread.join(timeout=5)
        assert not stop_thread.is_alive()
        assert weight_sync.validation_pending.is_set()
    finally:
        weight_sync.validation_pending.clear()
        if stop_thread.ident is not None:
            stop_thread.join(timeout=5)
        trainer.stop()

    assert trainer.training_complete.is_set()
    assert not trainer.train_started.is_set()
    assert not pool._returned_batch  # pylint: disable=protected-access
    assert trainer.get_train_step() == 0
    assert activity_tracker.track_count == 0
    assert activity_tracker.is_quiescent()
    assert weight_sync.notifications == []


def test_base_trainer_finishes_the_final_round_with_validation_pending():
    trainer = _AsyncTrainer(total_steps=1)
    pool = _SingleBatchPool(batch_count=2)
    weight_sync = _ValidationWeightSyncController()
    trainer.set_module_name("trainer")
    activity_tracker = _ActivityTracker()
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    try:
        assert trainer.training_complete.wait(timeout=5)
        assert weight_sync.validation_pending.is_set()
        assert activity_tracker.flushed_steps == []
        assert trainer.get_train_step() == 1
        assert pool._returned_batches == 1  # pylint: disable=protected-access
        assert weight_sync.notifications == [("trainer", "actor", True, False, None, None)]
    finally:
        weight_sync.validation_pending.clear()
        trainer.stop()


def test_base_trainer_publishes_final_step_after_weight_sync_is_ready():
    trainer = _AsyncTrainer(total_steps=1)
    pool = _SingleBatchPool()
    weight_sync = _BlockingWeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert weight_sync.started.wait(timeout=5)
    assert trainer.get_train_step() == 0
    assert activity_tracker.flushed_steps == []

    weight_sync.release.set()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 1
    assert activity_tracker.flushed_steps == [1]


def test_base_trainer_does_not_continue_after_a_skipped_final_round():
    trainer = _AsyncTrainer(
        metrics={
            "trajectory/reward_mean": 0.0,
            "training/skipped_flag": 1,
        },
        total_steps=1,
    )
    pool = _SingleBatchPool()
    weight_sync = _WeightSyncController()
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.training_complete.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 1
    assert activity_tracker.flushed_steps == [1]
    assert weight_sync.notifications == [("trainer", "actor", False, False, None, None)]


def test_sft_trainer_does_not_notify_weight_sync():
    trainer = _AsyncTrainer("sft")
    pool = _SingleBatchPool()
    weight_sync = _ValidationWeightSyncController(initially_waiting=True)
    activity_tracker = _ActivityTracker()
    trainer.set_module_name("trainer")
    trainer.set_activity_tracker(activity_tracker)  # type: ignore[arg-type]
    trainer.set_module_references(pool, weight_sync)  # type: ignore[arg-type]

    trainer.run()
    assert trainer.train_started.wait(timeout=5)
    trainer.stop()

    assert trainer.get_train_step() == 1
    assert not weight_sync.notifications
    assert weight_sync.validation_checks == 0
    assert activity_tracker.flushed_steps == [1]


@pytest.mark.parametrize("fail_first_refresh", [False, True])
def test_trainer_stop_refreshes_on_caller_thread_without_interrupting_training(fail_first_refresh):
    class _ControlledTrainer(_AsyncTrainer):
        async def train(self, trajectories):
            loop = asyncio.get_running_loop()
            release = asyncio.Event()
            self.release_train = lambda: loop.call_soon_threadsafe(release.set)
            self.train_started.set()
            await asyncio.wait_for(release.wait(), timeout=5)
            self.train_finished.set()
            return dict(self._metrics), None, None

    trainer = _ControlledTrainer()
    trainer.set_module_name("trainer")
    tracker = _ActivityTracker()
    weight_sync = _WeightSyncController()
    trainer.set_activity_tracker(tracker)
    trainer.set_module_references(_SingleBatchPool(), weight_sync)
    trainer.run()
    assert trainer.train_started.wait(timeout=5)
    future = trainer._task_future
    callback_threads = []

    def refresh():
        callback_threads.append(threading.get_ident())
        if len(callback_threads) == 1 and fail_first_refresh:
            raise RuntimeError("progress output failed")
        if len(callback_threads) >= 2:
            trainer.release_train()

    try:
        trainer.stop(on_wait=refresh, wait_interval=0.005)
    finally:
        trainer.release_train()

    assert len(callback_threads) >= 2
    assert set(callback_threads) == {threading.get_ident()}
    assert future.done() and not future.cancelled()
    assert trainer._task_future is None
    assert trainer.train_finished.is_set()
    assert trainer.training_complete.is_set()
    assert trainer.get_train_step() == 1
    assert weight_sync.notifications == [("trainer", "actor", True, False, None, None)]


def test_trainer_stop_handles_task_timeout_error_without_polling_forever(caplog):
    trainer = _AsyncTrainer()
    future = Future()
    future.set_exception(TimeoutError("optimizer request timed out"))
    trainer._task_future = future

    trainer.stop(
        on_wait=lambda: pytest.fail("A completed task must not be polled again"),
        wait_interval=0.005,
    )

    assert trainer._task_future is None
    assert not future.cancelled()
    assert "stopped after task failure: optimizer request timed out" in caplog.text


def test_round_manifest_counts_complete_outcomes_and_nested_sequences(tmp_path):
    import json

    from northrl.northrl_types import Trajectory

    trainer = _AsyncTrainer()
    trainer._config.runtime = {
        "debug": {
            "enable_data_dump": True,
            "dump_dir": str(tmp_path),
            "remote_api_dump_options": {"prepared_trajectories": True},
        }
    }
    trainer._train_step = 1
    identities = {"group_id": "g", "task_id": "task", "model_tag": "actor"}
    trajectories = [
        Trajectory(
            segments=[{"tokens": [1, 2], "loss_mask": [0, 1]} for _ in range(3)],
            reward=0.0,
            extra_fields={**identities, "rollout_id": "r0", "e2b_status": "task_failed"},
        ),
        Trajectory(segments=[], reward=1.0, extra_fields={**identities, "rollout_id": "r1"}),
        Trajectory(
            tokens=[1, 2, 3],
            loss_mask=[0, 1, 1],
            reward=1.0,
            extra_fields={
                **identities,
                "group_id": "failed-group",
                "rollout_id": "r2",
                "group_skipped": True,
            },
        ),
        Trajectory(
            segments=[],
            reward=None,
            status="failed",
            extra_fields={
                **identities,
                "group_id": "failed-group",
                "rollout_id": "r3",
                "group_skipped": True,
                "e2b_status": "env_error",
            },
        ),
    ]
    records = trainer._rollout_round_records(trajectories)
    metrics = {}
    trainer._record_rollout_round(records, metrics, 2)
    assert not any(key.startswith("round/") for key in metrics)
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/group_count"] == 2
    assert metrics["batch/trajectory_count"] == 4
    assert metrics["batch/segment_count"] == 4
    assert metrics["batch/empty_trajectory_count"] == 2
    payload = json.loads((tmp_path / "rollout_rounds" / "step_000001.json").read_text())
    assert payload["step"] == 1
    assert payload["training_step"] == 2
    assert len(payload["rollouts"]) == 4
    assert payload["rollouts"][0]["sequence_count"] == 3
    assert payload["rollouts"][1]["reward"] == 1.0
    assert payload["rollouts"][3]["reward"] is None


@pytest.mark.parametrize(
    ("model_tags", "expected_groups"),
    [((None, None), 1), ((None, "actor"), 2), (("actor", "actor"), 1)],
)
def test_round_group_count_uses_main_scalar_and_model_identity(model_tags, expected_groups):
    import torch

    from northrl.northrl_types import Trajectory

    trainer = _AsyncTrainer()
    trajectories = [
        Trajectory(
            segments=[],
            reward=0.0,
            extra_fields={
                "group_id": torch.tensor(7),
                **({"model_tag": tag} if tag is not None else {}),
            },
        )
        for tag in model_tags
    ]
    metrics = {}
    trainer._record_rollout_round(trainer._rollout_round_records(trajectories), metrics, 1)
    assert "round/group_count" not in metrics
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/group_count"] == expected_groups


@pytest.mark.parametrize("prepared_enabled", [False, True])
def test_round_manifest_uses_prepared_dump_frequency_and_batch_step(tmp_path, prepared_enabled):
    import json

    import torch

    from northrl.northrl_types import Trajectory
    from northrl.utils.data_dumper import DataDumper

    trainer = _AsyncTrainer()
    trainer._config.runtime = {
        "debug": {
            "enable_data_dump": True,
            "dump_dir": str(tmp_path),
            "dump_every_n_steps": 2,
            "remote_api_dump_options": {"prepared_trajectories": prepared_enabled},
        }
    }
    trainer._data_dumper = DataDumper(trainer._config)
    trajectory = Trajectory(
        segments=[], reward=torch.tensor(1.0), extra_fields={"group_id": torch.tensor(7)}
    )
    for step in range(3):
        trainer._train_step = step
        trainer._data_dumper.dump_prepared_trajectories(step, [trajectory])
        metrics = {}
        trainer._record_rollout_round(
            trainer._rollout_round_records([trajectory]), metrics, step + 1
        )
        assert "round/group_count" not in metrics

    expected_stems = ["step_000000", "step_000002"] if prepared_enabled else []
    manifests = sorted((tmp_path / "rollout_rounds").glob("*.json"))
    prepared = sorted((tmp_path / "prepared_trajectories").glob("*.pt"))
    assert [path.stem for path in manifests] == [path.stem for path in prepared] == expected_stems
    for path in manifests:
        payload = json.loads(path.read_text())
        assert payload["training_step"] == payload["step"] + 1
        assert payload["rollouts"][0]["group_id"] == 7
        assert payload["rollouts"][0]["reward"] == 1.0
