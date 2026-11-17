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

"""Unit tests for WeightSyncController version-counter bookkeeping."""

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, call

import pytest
from omegaconf import OmegaConf

# Avoid importing northrl.__init__, which pulls optional runtime dependencies that
# are unrelated to these small weight-sync unit tests.
_orig_northrl = sys.modules.get("northrl")
northrl_stub = types.ModuleType("northrl")
northrl_stub.__path__ = [str(Path(__file__).resolve().parents[3] / "northrl")]
sys.modules["northrl"] = northrl_stub
try:
    from northrl.weight_sync.weight_sync_controller import WeightSyncController  # noqa: E402
finally:
    if _orig_northrl is not None:
        sys.modules["northrl"] = _orig_northrl
    else:
        sys.modules.pop("northrl", None)


def _make_config(*, mode="sync", algorithm_type="grpo", rollout_type="default", validate_freq=0):
    """Minimal new-schema config accepted by WeightSyncController.__init__."""
    config = {
        "training": {"weight_sync": {"mode": mode, "staleness_threshold": 0}},
        "backend": {"type": "mock"},
        "algorithm": {"type": algorithm_type},
        "rollout": {},
        "validate": {"schedule": {"every_n_steps": validate_freq}},
    }
    if algorithm_type != "sft":
        config["rollout"].update({"type": rollout_type})
    return OmegaConf.create(config)


class _SpyDataLoader:
    def __init__(self, events: list[str] | None = None):
        self.release_count = 0
        self._events = events

    def release_next_batch(self):
        self.release_count += 1
        if self._events is not None:
            self._events.append("release")


def _make_fully_async_weaver_controller(*, validate_freq=0, pause_mode=None):
    config = _make_config(mode="fully-async", validate_freq=validate_freq)
    config.backend.type = "weaver"
    config.training.weight_sync.pause_mode = pause_mode
    controller = WeightSyncController(config)
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    holder = types.SimpleNamespace(update_sampling_client=AsyncMock())
    controller.set_module_references(dataloader, trajectory_pool)
    controller.set_weaver_service_holder(holder)
    return controller, holder, dataloader


def test_no_sync_callbacks_do_not_coordinate_batches():
    controller = WeightSyncController(_make_config(mode="no-sync", rollout_type="default"))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None
    )

    assert dataloader.release_count == 0


@pytest.mark.parametrize("mode", ["sync", "batch-async", "no-sync"])
def test_initial_validation_releases_first_batch(mode):
    controller = WeightSyncController(_make_config(mode=mode))
    events: list[str] = []
    dataloader = _SpyDataLoader(events)
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    # Initial validation completes before any training rollout has reached the pool.
    controller.end_validate("actor")

    assert dataloader.release_count == 1
    assert events == ["release"]


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_sync_releases_data_after_weight_update(monkeypatch, mode):
    controller = WeightSyncController(_make_config(mode=mode))
    events: list[str] = []
    dataloader = _SpyDataLoader(events)
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)
    sync_calls = []
    original_sync = controller.sync_weight_to_rollout_service

    def _sync(model_tag, sampling_path, weight_version):
        sync_calls.append((model_tag, sampling_path, weight_version))
        result = original_sync(model_tag, sampling_path, weight_version)
        events.append("sync")
        return result

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", _sync)

    controller.trajectory_pool_notify_batch_ready("actor")

    assert controller.check_rollout_service_status("actor") == "block"
    assert dataloader.release_count == 0

    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None
    )

    assert dataloader.release_count == 1
    assert events == ["sync", "release"]
    assert sync_calls == [("actor", None, None)]

    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    assert service.state == "running"
    assert service.train_model_version == 1
    assert service.rollout_model_version == 1


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_sync_skip_releases_waiting_batch_without_advancing_version(monkeypatch, mode):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=1))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    service.train_model_version = 4
    service.rollout_model_version = 4

    def _unexpected_sync(model_tag, sampling_path, weight_version):
        pytest.fail(f"Skipped training unexpectedly synced {model_tag}")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", _unexpected_sync)

    controller.trajectory_pool_notify_batch_ready("actor")

    assert service.state == "need_sync"
    assert controller.check_rollout_service_status("actor") == "block"
    assert dataloader.release_count == 0

    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, weights_updated=False
    )

    assert service.state == "running"
    assert service.train_model_version == 4
    assert service.rollout_model_version == 4
    assert dataloader.release_count == 1
    assert not controller.is_waiting_for_validation()


@pytest.mark.parametrize("continue_training", [True, False])
def test_skip_does_not_release_again_when_service_is_already_running(
    monkeypatch, continue_training
):
    controller = WeightSyncController(_make_config(mode="fully-async"))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    service.train_model_version = 3
    service.rollout_model_version = 1

    def _unexpected_sync(model_tag, sampling_path, weight_version):
        pytest.fail(f"Skipped training unexpectedly synced {model_tag}")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", _unexpected_sync)

    # Fully-async fetches data continuously without a dataloader release.
    controller.trajectory_pool_notify_batch_ready("actor")
    assert service.state == "running"
    assert dataloader.release_count == 0

    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=False,
        continue_training=continue_training,
    )

    assert service.state == "running"
    assert service.train_model_version == 3
    assert service.rollout_model_version == 1
    assert dataloader.release_count == 0


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_final_sync_updates_weights_without_releasing_another_batch(monkeypatch, mode):
    controller = WeightSyncController(_make_config(mode=mode))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)
    sync_calls = []
    original_sync = controller.sync_weight_to_rollout_service

    def _sync(model_tag, sampling_path, weight_version):
        sync_calls.append((model_tag, sampling_path, weight_version))
        return original_sync(model_tag, sampling_path, weight_version)

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", _sync)

    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        continue_training=False,
    )

    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    assert service.state == "running"
    assert service.train_model_version == 1
    assert service.rollout_model_version == 1
    assert sync_calls == [("actor", None, None)]
    assert dataloader.release_count == 0


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_skipped_final_round_keeps_the_next_batch_blocked(monkeypatch, mode):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=1))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    service.train_model_version = 4
    service.rollout_model_version = 4

    def _unexpected_sync(model_tag, sampling_path, weight_version):
        pytest.fail(f"Skipped training unexpectedly synced {model_tag}")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", _unexpected_sync)

    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=False,
        continue_training=False,
    )

    assert service.state == "running"
    assert service.train_model_version == 4
    assert service.rollout_model_version == 4
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
@pytest.mark.parametrize("continue_training", [True, False])
def test_validation_releases_a_held_batch_only_after_completion(mode, continue_training):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=1))
    dataloader = _SpyDataLoader()
    trajectory_pool = object()
    controller.set_module_references(dataloader, trajectory_pool)

    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        continue_training=continue_training,
    )

    assert controller.is_waiting_for_validation()
    assert dataloader.release_count == 0

    controller.end_validate("actor")

    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == int(continue_training)


@pytest.mark.parametrize("continue_training", [True, False])
@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_fully_async_updates_sampler_every_round_without_dataloader_release(
    continue_training, pause_mode
):
    controller, holder, dataloader = _make_fully_async_weaver_controller(pause_mode=pause_mode)
    service = controller._rollout_services["actor"]  # pylint: disable=protected-access

    for version in (1, 2):
        controller.trajectory_pool_notify_batch_ready("actor")
        assert dataloader.release_count == 0
        assert service.state == "running"

        controller.train_worker_notify_weight_update(
            "trainer",
            "actor",
            sampling_path=f"/samplers/{version}",
            weight_version=f"export-{version}",
            continue_training=version == 1 or continue_training,
        )

        assert service.state == "running"
        assert service.train_model_version == service.rollout_model_version == version
        assert dataloader.release_count == 0
        assert not controller.is_waiting_for_validation()

    pause_kwargs = {} if pause_mode is None else {"pause_mode": pause_mode}
    assert holder.update_sampling_client.await_args_list == [
        call("/samplers/1", "export-1", **pause_kwargs),
        call("/samplers/2", "export-2", **pause_kwargs),
    ]


@pytest.mark.parametrize("continue_training", [True, False])
def test_fully_async_validation_does_not_release_dataloader(continue_training):
    controller, holder, dataloader = _make_fully_async_weaver_controller(validate_freq=1)
    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path="/samplers/1",
        weight_version="export-1",
        continue_training=continue_training,
    )

    holder.update_sampling_client.assert_awaited_once_with("/samplers/1", "export-1")
    assert controller.is_waiting_for_validation()
    assert controller.get_rollout_model_version("actor") == 1
    assert dataloader.release_count == 0

    controller.end_validate("actor")

    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize("pause_mode", [None, "retract", "in_place"])
def test_fully_async_sync_failure_preserves_last_published_version_and_propagates(pause_mode):
    controller, holder, dataloader = _make_fully_async_weaver_controller(pause_mode=pause_mode)
    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path="/samplers/1", weight_version="export-1"
    )
    failure = RuntimeError("sampler creation failed")
    holder.update_sampling_client.side_effect = failure
    controller.trajectory_pool_notify_batch_ready("actor")

    with pytest.raises(RuntimeError) as caught:
        controller.train_worker_notify_weight_update(
            "trainer", "actor", sampling_path="/samplers/2", weight_version="export-2"
        )

    assert caught.value is failure
    pause_kwargs = {} if pause_mode is None else {"pause_mode": pause_mode}
    assert holder.update_sampling_client.await_args_list == [
        call("/samplers/1", "export-1", **pause_kwargs),
        call("/samplers/2", "export-2", **pause_kwargs),
    ]
    service = controller._rollout_services["actor"]  # pylint: disable=protected-access
    assert service.train_model_version == 2
    assert service.rollout_model_version == 1
    assert controller.check_rollout_service_status("actor") == "block"
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize("weights_updated", [True, False])
def test_batch_async_final_round_does_not_repeat_its_early_release(monkeypatch, weights_updated):
    config = _make_config(mode="batch-async", validate_freq=1)
    config.training.weight_sync.staleness_threshold = 2
    controller = WeightSyncController(config)
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)

    def unexpected_sync(*args, **kwargs):
        pytest.fail("A below-threshold batch must not synchronize")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", unexpected_sync)
    controller.trajectory_pool_notify_batch_ready("actor")
    assert dataloader.release_count == 1

    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=weights_updated,
        continue_training=False,
    )

    service = controller._rollout_services["actor"]
    assert service.train_model_version == int(weights_updated)
    assert service.rollout_model_version == 0
    assert service.state == "running"
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 1


@pytest.mark.parametrize("mode", ["sync", "batch-async"])
def test_failed_sync_does_not_release_held_batch_or_start_validation(mode):
    config = _make_config(mode=mode, validate_freq=1)
    config.backend.type = "weaver"
    controller = WeightSyncController(config)
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)
    failure = RuntimeError("sampler creation failed")
    holder = types.SimpleNamespace(update_sampling_client=AsyncMock(side_effect=failure))
    controller.set_weaver_service_holder(holder)
    controller.trajectory_pool_notify_batch_ready("actor")

    with pytest.raises(RuntimeError) as caught:
        controller.train_worker_notify_weight_update(
            "trainer", "actor", sampling_path="/samplers/1", weight_version="export-1"
        )

    assert caught.value is failure
    holder.update_sampling_client.assert_awaited_once_with("/samplers/1", "export-1")
    service = controller._rollout_services["actor"]
    assert service.train_model_version == 1
    assert service.rollout_model_version == 0
    assert service.state == "syncing"
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize("validate_freq", [0, 1])
def test_sync_timing_uses_completed_round_before_validation_or_batch_release(
    monkeypatch, validate_freq
):
    controller = WeightSyncController(_make_config(validate_freq=validate_freq))
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)
    controller.trajectory_pool_notify_batch_ready("actor")
    now = [100.0]
    monkeypatch.setattr(
        sys.modules[WeightSyncController.__module__],
        "time",
        types.SimpleNamespace(monotonic=lambda: now[0]),
    )
    original_sync = controller.sync_weight_to_rollout_service
    posts = []

    def sync(*args):
        now[0] += 4.5
        original_sync(*args)

    def post(**kwargs):
        assert not controller.is_waiting_for_validation()
        assert dataloader.release_count == 0
        posts.append(kwargs)
        now[0] += 30.0  # Metric publication must not inflate synchronization time.

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", sync)
    controller.set_activity_tracker(types.SimpleNamespace(experiment_logger_post=post))
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, completed_step=11
    )
    assert posts == [
        {
            "backend": "wandb",
            "step": 11,
            "data": {"timing/weight_sync_seconds": 4.5, "training/weight_sync_success_flag": 1},
        }
    ]
    assert controller.get_rollout_model_version("actor") == 1
    assert controller.is_waiting_for_validation() is bool(validate_freq)
    assert dataloader.release_count == int(not validate_freq)
    if validate_freq:
        now[0] += 200.0
        controller.end_validate("actor")
        assert dataloader.release_count == 1
        assert len(posts) == 1


@pytest.mark.parametrize("metrics_fail", [False, True])
def test_sync_failure_timing_does_not_mask_failure_or_release_batch(monkeypatch, metrics_fail):
    controller = WeightSyncController(_make_config(validate_freq=1))
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)
    controller.trajectory_pool_notify_batch_ready("actor")
    now = [100.0]
    monkeypatch.setattr(
        sys.modules[WeightSyncController.__module__],
        "time",
        types.SimpleNamespace(monotonic=lambda: now[0]),
    )
    failure = RuntimeError("publish failed")
    posts = []

    def sync(*args):
        now[0] += 6.5
        raise failure

    def post(**kwargs):
        posts.append(kwargs)
        if metrics_fail:
            raise RuntimeError("logger unavailable")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", sync)
    controller.set_activity_tracker(types.SimpleNamespace(experiment_logger_post=post))
    with pytest.raises(RuntimeError) as caught:
        controller.train_worker_notify_weight_update(
            "trainer", "actor", sampling_path=None, weight_version=None, completed_step=11
        )
    assert caught.value is failure
    assert posts == [
        {
            "backend": "wandb",
            "step": 11,
            "data": {
                "timing/weight_sync_failed_seconds": 6.5,
                "training/weight_sync_success_flag": 0,
            },
        }
    ]
    assert controller.get_rollout_model_version("actor") == 0
    assert controller.check_rollout_service_status("actor") == "block"
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize(
    ("mode", "weights_updated"), [("no-sync", True), ("sync", False), ("batch-async", True)]
)
def test_unexecuted_sync_does_not_emit_zero_duration(monkeypatch, mode, weights_updated):
    config = _make_config(mode=mode)
    config.training.weight_sync.staleness_threshold = 2
    controller = WeightSyncController(config)
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)
    controller.trajectory_pool_notify_batch_ready("actor")
    posts = []
    controller.set_activity_tracker(
        types.SimpleNamespace(experiment_logger_post=lambda **kwargs: posts.append(kwargs))
    )
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        weights_updated=weights_updated,
        completed_step=11,
    )
    assert posts == []


def test_sync_metrics_failure_does_not_block_next_batch():
    controller = WeightSyncController(_make_config())
    dataloader, trajectory_pool = _SpyDataLoader(), object()
    controller.set_module_references(dataloader, trajectory_pool)
    controller.trajectory_pool_notify_batch_ready("actor")

    def fail_metrics(**kwargs):
        raise RuntimeError("logger unavailable")

    controller.set_activity_tracker(types.SimpleNamespace(experiment_logger_post=fail_metrics))
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, completed_step=1
    )
    assert controller.get_rollout_model_version("actor") == 1
    assert controller.check_rollout_service_status("actor") == "continue"
    assert dataloader.release_count == 1


@pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
@pytest.mark.parametrize(
    ("completed_step", "previous_version", "validate_freq", "expected_validation"),
    [
        (200, 9, 25, True),
        (215, 24, 25, False),
        (225, 34, 25, True),
        (0, 24, 25, False),
        (200, 24, 0, False),
    ],
)
def test_validation_uses_global_completed_step_after_sync(
    mode, completed_step, previous_version, validate_freq, expected_validation
):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=validate_freq))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    service = controller._rollout_services["actor"]
    service.train_model_version = service.rollout_model_version = previous_version
    controller.trajectory_pool_notify_batch_ready("actor")

    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, completed_step=completed_step
    )

    assert controller.is_waiting_for_validation() is expected_validation
    assert service.train_model_version == service.rollout_model_version == previous_version + 1
    assert service.state == "running"
    assert dataloader.release_count == int(not expected_validation and mode != "fully-async")
    if expected_validation:
        controller.end_validate("actor")
        assert not controller.is_waiting_for_validation()
        assert dataloader.release_count == int(mode != "fully-async")


@pytest.mark.parametrize(
    ("resume_step", "expected_validation_steps"),
    [(0, [25, 50]), (190, [200, 225]), (200, [225, 250]), (205, [225, 250])],
)
def test_resumed_validation_schedule_does_not_restart_its_global_interval(
    resume_step, expected_validation_steps
):
    controller = WeightSyncController(_make_config(validate_freq=25))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    controller.sync_weight_to_rollout_service(
        "actor", "mock://resumed", f"northrl_actor_v{resume_step:06d}"
    )
    assert controller.get_rollout_model_version("actor") == 0
    validation_steps = []

    for update_count, completed_step in enumerate(range(resume_step + 1, resume_step + 51), 1):
        controller.trajectory_pool_notify_batch_ready("actor")
        controller.train_worker_notify_weight_update(
            "trainer",
            "actor",
            sampling_path=None,
            weight_version=None,
            completed_step=completed_step,
        )
        assert controller.get_rollout_model_version("actor") == update_count
        if controller.is_waiting_for_validation():
            validation_steps.append(completed_step)
            controller.end_validate("actor")

    assert validation_steps == expected_validation_steps
    assert dataloader.release_count == 50


def test_skipped_round_does_not_shift_later_global_validation_boundaries():
    controller = WeightSyncController(_make_config(validate_freq=25))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    validation_steps = []

    for completed_step in range(191, 226):
        controller.trajectory_pool_notify_batch_ready("actor")
        controller.train_worker_notify_weight_update(
            "trainer",
            "actor",
            sampling_path=None,
            weight_version=None,
            completed_step=completed_step,
            weights_updated=completed_step != 200,
        )
        if controller.is_waiting_for_validation():
            validation_steps.append(completed_step)
            controller.end_validate("actor")

    assert validation_steps == [225]
    assert controller.get_rollout_model_version("actor") == 34
    assert dataloader.release_count == 35


@pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
def test_final_global_validation_does_not_release_another_training_batch(mode):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=25))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        completed_step=200,
        continue_training=False,
    )

    assert controller.is_waiting_for_validation()
    assert dataloader.release_count == 0
    controller.end_validate("actor")
    assert not controller.is_waiting_for_validation()
    assert dataloader.release_count == 0


@pytest.mark.parametrize("mode", ["sync", "batch-async", "fully-async"])
def test_skipped_global_validation_boundary_preserves_existing_skip_policy(mode):
    controller = WeightSyncController(_make_config(mode=mode, validate_freq=25))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    service = controller._rollout_services["actor"]
    service.train_model_version = service.rollout_model_version = 9
    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer",
        "actor",
        sampling_path=None,
        weight_version=None,
        completed_step=200,
        weights_updated=False,
    )

    assert not controller.is_waiting_for_validation()
    assert service.train_model_version == service.rollout_model_version == 9
    assert dataloader.release_count == int(mode != "fully-async")


def test_global_validation_boundary_does_not_force_batch_async_sync(monkeypatch):
    config = _make_config(mode="batch-async", validate_freq=25)
    config.training.weight_sync.staleness_threshold = 2
    controller = WeightSyncController(config)
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())

    def unexpected_sync(*_args):
        pytest.fail("Validation must not change the batch-async synchronization policy")

    monkeypatch.setattr(controller, "sync_weight_to_rollout_service", unexpected_sync)
    controller.trajectory_pool_notify_batch_ready("actor")
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, completed_step=200
    )

    assert not controller.is_waiting_for_validation()
    assert controller.get_rollout_model_version("actor") == 0
    assert dataloader.release_count == 1


def test_global_validation_boundary_does_not_change_no_sync_mode():
    controller = WeightSyncController(_make_config(mode="no-sync", validate_freq=25))
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    controller.train_worker_notify_weight_update(
        "trainer", "actor", sampling_path=None, weight_version=None, completed_step=200
    )

    assert not controller.is_waiting_for_validation()
    assert controller.get_rollout_model_version("actor") == 0
    assert dataloader.release_count == 0


def test_global_validation_boundary_waits_for_successful_sync():
    config = _make_config(validate_freq=25)
    config.backend.type = "weaver"
    controller = WeightSyncController(config)
    dataloader = _SpyDataLoader()
    controller.set_module_references(dataloader, object())
    service = controller._rollout_services["actor"]
    service.train_model_version = service.rollout_model_version = 9
    failure = RuntimeError("sampler publication failed")
    controller.set_weaver_service_holder(
        types.SimpleNamespace(update_sampling_client=AsyncMock(side_effect=failure))
    )
    controller.trajectory_pool_notify_batch_ready("actor")

    with pytest.raises(RuntimeError) as caught:
        controller.train_worker_notify_weight_update(
            "trainer",
            "actor",
            sampling_path="mock://weights/200",
            weight_version="northrl_actor_v000200",
            completed_step=200,
        )

    assert caught.value is failure
    assert not controller.is_waiting_for_validation()
    assert service.train_model_version == 10
    assert service.rollout_model_version == 9
    assert dataloader.release_count == 0


@pytest.mark.parametrize("completed_step", [None, 200])
def test_validation_log_distinguishes_global_step_from_legacy_sync_version(caplog, completed_step):
    controller = WeightSyncController(_make_config(validate_freq=25))
    controller.set_module_references(_SpyDataLoader(), object())
    previous_version = 24 if completed_step is None else 9
    service = controller._rollout_services["actor"]
    service.train_model_version = service.rollout_model_version = previous_version
    controller.trajectory_pool_notify_batch_ready("actor")

    with caplog.at_level("INFO", logger=WeightSyncController.__module__):
        controller.train_worker_notify_weight_update(
            "trainer",
            "actor",
            sampling_path=None,
            weight_version=None,
            completed_step=completed_step,
        )

    assert controller.is_waiting_for_validation()
    messages = [record.getMessage() for record in caplog.records]
    validation_messages = [message for message in messages if "Validation required:" in message]
    assert len(validation_messages) == 1
    message = validation_messages[0]
    assert f"global_step={completed_step}" in message
    assert f"sync_version={previous_version + 1}" in message
    source = "legacy_sync_version" if completed_step is None else "global_step"
    assert f"schedule_source={source}" in message
