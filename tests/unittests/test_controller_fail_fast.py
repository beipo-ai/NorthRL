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

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

import northrl.controller as controller_module
from northrl.activity_tracker import ActivityTracker, ActivityTrackerProxy
from northrl.controller import NorthRLController
from northrl.data_loader import SFTStreamingDataLoader
from northrl.exceptions import WeightVersionMismatchError
from northrl.northrl_types import NorthRLRole
from northrl.rollout_worker import E2BRolloutWorker, SFTRolloutWorker
from northrl.trainer.remote_api_cross_entropy_trainer import (
    RemoteApiCrossEntropyTrainer,
)
from northrl.validator import Validator


def _make_controller():
    controller = object.__new__(NorthRLController)
    controller.use_weaver = False
    controller._config = SimpleNamespace(
        experiment=SimpleNamespace(project="test-project", name="test-run"),
        algorithm=SimpleNamespace(type="grpo"),
        validate=SimpleNamespace(
            enabled=False,
            schedule=SimpleNamespace(before_train=False),
        ),
        runtime=SimpleNamespace(
            health=SimpleNamespace(
                exception_policy="stop_on_error",
                exception_check_interval_s=1.0,
                memory_log_interval_s=10.0,
            ),
            progress=SimpleNamespace(update_interval_s=1.0),
        ),
    )
    controller.trainer = SimpleNamespace(
        initialize_workers=AsyncMock(),
        run=MagicMock(),
    )
    controller.dataloader = SimpleNamespace(prepare=AsyncMock())
    controller.rollout_workers = []
    controller.trajectory_pool = MagicMock()
    controller.activity_tracker = MagicMock()
    controller.activity_tracker.get_error_health_status.return_value = {"error_level_count": 0}
    controller.activity_tracker.update_progress.return_value = float("inf")
    controller.activity_tracker.experiment_logger.wandb_url = "https://wandb.test/run"
    controller.weight_sync_controller = MagicMock()
    controller.weight_sync_controller.is_waiting_for_validation.return_value = False
    controller._load_initial_checkpoint = MagicMock()
    controller._stop = MagicMock()
    return controller


@pytest.mark.parametrize("sync_mode", ["sync", "batch-async", "fully-async", "no-sync"])
def test_controller_allows_sampler_version_mismatch_only_in_fully_async(
    monkeypatch, rollout_worker_config, sync_mode
) -> None:
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    controller = _make_controller()
    controller._config = rollout_worker_config
    controller._config.training.weight_sync.mode = sync_mode
    controller._init_weaver_service()
    sequence = {
        "tokens": [3],
        "logprobs": [-0.1],
        "text": "ok",
        "stop_reason": "stop",
        "weight_version": "northrl_actor_v000001",
    }
    holder = controller._weaver_service_holder

    if sync_mode == "fully-async":
        result = holder._weaver_sequence_to_sampling_result(sequence, [1, 2], "default")
        assert result["weight_version"] == "northrl_actor_v000001"
    else:
        with pytest.raises(WeightVersionMismatchError, match="expected='default'"):
            holder._weaver_sequence_to_sampling_result(sequence, [1, 2], "default")


@pytest.mark.parametrize(
    "backend_overrides,expected_tier",
    [({}, None), ({"performance_tier": None}, None), ({"performance_tier": "flash"}, "flash")],
    ids=["omitted", "null", "flash"],
)
def test_controller_forwards_backend_performance_tier(
    monkeypatch, rollout_worker_config, backend_overrides, expected_tier
) -> None:
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    holder_factory = MagicMock()
    holder_factory.return_value.configure_sampling_flags = AsyncMock()
    monkeypatch.setattr("northrl.weaver.WeaverServiceHolder", holder_factory)
    controller = _make_controller()
    controller._config = rollout_worker_config
    controller._config.backend.update(backend_overrides)

    controller._init_weaver_service()

    holder_factory.assert_called_once()
    assert holder_factory.call_args.kwargs["performance_tier"] == expected_tier


@pytest.mark.parametrize(
    ("rollout_type", "execution_config", "expected_timeout"),
    [
        ("e2b", {}, 300),
        ("e2b", {"gateway_publication_timeout_seconds": 900.5}, 900.5),
        ("single_turn", None, None),
        ("simple", {"gateway_publication_timeout_seconds": 900.5}, None),
    ],
    ids=["e2b-default", "e2b-configured", "single-turn", "simple"],
)
def test_controller_enables_gateway_publication_wait_only_for_e2b(
    monkeypatch, rollout_worker_config, rollout_type, execution_config, expected_timeout
) -> None:
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    holder_factory = MagicMock()
    holder_factory.return_value.configure_sampling_flags = AsyncMock()
    monkeypatch.setattr("northrl.weaver.WeaverServiceHolder", holder_factory)
    controller = _make_controller()
    controller._config = rollout_worker_config
    controller._config.rollout.type = rollout_type
    if execution_config is not None:
        controller._config.rollout.execution = execution_config

    controller._init_weaver_service()

    holder_factory.assert_called_once()
    assert (
        holder_factory.call_args.kwargs["gateway_publication_timeout_seconds"] == expected_timeout
    )


def test_controller_warms_shared_weaver_sampler() -> None:
    controller = _make_controller()
    controller.use_weaver = True
    controller._weaver_service_holder = SimpleNamespace(warmup_sampling=AsyncMock())

    controller._warmup_rollout_service()

    controller._weaver_service_holder.warmup_sampling.assert_awaited_once_with()


@pytest.mark.parametrize("scheduling_policy", [None, {"mode": "fixed", "clusters": ["cluster-1"]}])
def test_controller_forwards_weaver_scheduling_policy(
    monkeypatch, rollout_worker_config, scheduling_policy
) -> None:
    monkeypatch.setenv("WEAVER_API_KEY", "environment-key")
    monkeypatch.setenv("WEAVER_BASE_URL", "https://environment.weaver.example")
    rollout_worker_config.backend.scheduling_policy = scheduling_policy
    holder_factory = MagicMock()
    holder_factory.return_value.configure_sampling_flags = AsyncMock()
    monkeypatch.setattr("northrl.weaver.WeaverServiceHolder", holder_factory)
    controller = _make_controller()
    controller._config = rollout_worker_config

    controller._init_weaver_service()

    kwargs = holder_factory.call_args.kwargs
    assert kwargs["base_url"] == "https://environment.weaver.example"
    assert kwargs["scheduling_policy"] == scheduling_policy
    if scheduling_policy is not None:
        assert isinstance(kwargs["scheduling_policy"]["clusters"], list)


def test_controller_initializes_epoch_sft_lr_horizon_from_exact_window_plan() -> None:
    controller = object.__new__(NorthRLController)
    controller._config = OmegaConf.create(
        {
            "algorithm": {
                "type": "sft",
                "optim": {
                    "schedule": {
                        "style": "cosine",
                        "min_lr": 2.0e-6,
                        "warmup_fraction": 0.03,
                    }
                },
            },
            "training": {"total_steps": None, "epochs": 1},
        }
    )
    controller.dataloader = SimpleNamespace(count_optimizer_windows=AsyncMock(return_value=86))
    controller.trainer = SimpleNamespace(set_lr_schedule_total_steps=MagicMock())

    controller._initialize_lr_schedule_horizon()

    controller.dataloader.count_optimizer_windows.assert_awaited_once_with()
    controller.trainer.set_lr_schedule_total_steps.assert_called_once_with(86)


@pytest.mark.parametrize(
    ("use_weaver", "algorithm_type"),
    [(False, "grpo"), (True, "sft")],
)
def test_controller_skips_rollout_warmup_without_weaver_sampling(
    use_weaver, algorithm_type
) -> None:
    controller = _make_controller()
    controller.use_weaver = use_weaver
    controller._config.algorithm.type = algorithm_type
    controller._weaver_service_holder = SimpleNamespace(warmup_sampling=AsyncMock())

    controller._warmup_rollout_service()

    controller._weaver_service_holder.warmup_sampling.assert_not_awaited()


def test_controller_warms_rollout_before_starting_workers() -> None:
    controller = _make_controller()
    events: list[str] = []
    controller.trainer.initialize_workers = AsyncMock(
        side_effect=lambda: events.append("initialize-trainer")
    )
    controller.dataloader.prepare = AsyncMock(
        side_effect=lambda: events.append("prepare-dataloader")
    )
    controller._load_initial_checkpoint.side_effect = lambda: events.append("load-checkpoint")
    controller._prepare_e2b_artifacts = MagicMock(
        side_effect=lambda: events.append("prepare-e2b-artifacts")
    )
    controller._warmup_rollout_service = MagicMock(
        side_effect=lambda: events.append("warmup-rollout")
    )
    controller.trainer.run.side_effect = lambda: events.append("start-trainer")
    worker = SimpleNamespace(run=MagicMock(side_effect=lambda: events.append("start-worker")))
    controller.rollout_workers = [worker]
    controller._check_module_exceptions = MagicMock(return_value=True)
    controller._check_finish = MagicMock(return_value=True)

    controller.run()

    assert events == [
        "initialize-trainer",
        "prepare-dataloader",
        "load-checkpoint",
        "prepare-e2b-artifacts",
        "warmup-rollout",
        "start-trainer",
        "start-worker",
    ]
    controller._prepare_e2b_artifacts.assert_called_once_with()
    controller._warmup_rollout_service.assert_called_once_with()


def test_controller_stops_before_workers_when_gateway_publication_times_out() -> None:
    controller = _make_controller()
    controller.use_weaver = True
    publication_error = TimeoutError("NorthGate publication timed out for sampler sampler-1")
    controller._weaver_service_holder = SimpleNamespace(
        warmup_sampling=AsyncMock(side_effect=publication_error)
    )
    controller._prepare_e2b_artifacts = MagicMock()
    worker = SimpleNamespace(run=MagicMock())
    controller.rollout_workers = [worker]

    with pytest.raises(TimeoutError, match="NorthGate publication timed out for sampler sampler-1"):
        controller.run()

    controller._weaver_service_holder.warmup_sampling.assert_awaited_once_with()
    controller.trainer.run.assert_not_called()
    worker.run.assert_not_called()
    controller._stop.assert_called_once_with()


def test_controller_shares_prepared_e2b_artifacts_across_workers(monkeypatch) -> None:
    from northrl.remote_rollout import e2b_executor

    controller = _make_controller()
    controller._config.rollout = SimpleNamespace(execution={})
    workers = [object.__new__(E2BRolloutWorker), object.__new__(E2BRolloutWorker)]
    for worker in workers:
        worker.set_prepared_artifacts = MagicMock()
    other_worker = SimpleNamespace(set_prepared_artifacts=MagicMock())
    controller.rollout_workers = [workers[0], other_worker, workers[1]]
    artifacts = SimpleNamespace(source_archive=b"source", evaluation_archives={"task": b"eval"})
    preparation = MagicMock(return_value=artifacts)
    monkeypatch.setattr(e2b_executor, "prepare_e2b_artifacts", preparation)

    controller._prepare_e2b_artifacts()

    preparation.assert_called_once_with(controller._config)
    for worker in workers:
        worker.set_prepared_artifacts.assert_called_once_with(artifacts)
    other_worker.set_prepared_artifacts.assert_not_called()


def test_controller_prepares_profile_artifacts_once_and_shares_registry(monkeypatch) -> None:
    from northrl.remote_rollout import e2b_executor

    controller = _make_controller()
    controller._config = OmegaConf.create(
        {
            "rollout": {
                "execution": {
                    "profiles": {
                        "code": {
                            "entrypoint": {"target": "/tasks/code.py"},
                            "input_fields": ["instruction"],
                        },
                        "swe": {
                            "entrypoint": {"target": "/tasks/swe.py"},
                            "input_fields": ["instruction", "interface"],
                        },
                    }
                }
            }
        }
    )
    workers = [object.__new__(E2BRolloutWorker), object.__new__(E2BRolloutWorker)]
    for worker in workers:
        worker.set_prepared_artifacts_by_profile = MagicMock()
        worker.set_prepared_artifacts = MagicMock()
    controller.rollout_workers = workers
    code_assets = SimpleNamespace(
        source_archive=b"code", evaluation_archives={"same-task": b"code-test"}
    )
    swe_assets = SimpleNamespace(
        source_archive=b"swe", evaluation_archives={"same-task": b"swe-test"}
    )
    preparation = MagicMock(side_effect=[code_assets, swe_assets])
    monkeypatch.setattr(e2b_executor, "prepare_e2b_artifacts", preparation)

    controller._prepare_e2b_artifacts()

    assert preparation.call_count == 2
    assert [
        call.args[0].rollout.execution.entrypoint.target for call in preparation.call_args_list
    ] == [
        "/tasks/code.py",
        "/tasks/swe.py",
    ]
    for worker in workers:
        worker.set_prepared_artifacts_by_profile.assert_called_once_with(
            {"code": code_assets, "swe": swe_assets}
        )
        worker.set_prepared_artifacts.assert_not_called()
    assert (
        workers[0].set_prepared_artifacts_by_profile.call_args.args[0]
        is workers[1].set_prepared_artifacts_by_profile.call_args.args[0]
    )


def test_controller_skips_artifact_preparation_without_e2b_workers(monkeypatch) -> None:
    from northrl.remote_rollout import e2b_executor

    controller = _make_controller()
    controller.rollout_workers = [SimpleNamespace(set_prepared_artifacts=MagicMock())]
    preparation = MagicMock()
    monkeypatch.setattr(e2b_executor, "prepare_e2b_artifacts", preparation)

    controller._prepare_e2b_artifacts()

    preparation.assert_not_called()


def test_controller_stops_before_sampler_warmup_when_e2b_preparation_fails() -> None:
    controller = _make_controller()
    controller._prepare_e2b_artifacts = MagicMock(side_effect=OSError("archive preparation failed"))
    controller._warmup_rollout_service = MagicMock()
    worker = SimpleNamespace(run=MagicMock())
    controller.rollout_workers = [worker]

    with pytest.raises(OSError, match="archive preparation failed"):
        controller.run()

    controller._load_initial_checkpoint.assert_called_once_with()
    controller._warmup_rollout_service.assert_not_called()
    controller.trainer.run.assert_not_called()
    worker.run.assert_not_called()
    controller._stop.assert_called_once_with()


@pytest.mark.parametrize(
    ("algorithm", "sync_mode", "before_train"),
    [
        ("sft", "no-sync", False),
        ("grpo", "no-sync", False),
        ("grpo", "fully-async", False),
        ("grpo", "sync", True),
        ("grpo", "batch-async", True),
    ],
)
def test_controller_runs_initial_validation_when_configured(algorithm, sync_mode, before_train):
    controller = _make_controller()
    events: list[str] = []
    controller._config.algorithm.type = algorithm
    controller._config.validate.enabled = before_train
    controller._config.validate.schedule.before_train = before_train
    controller.weight_sync_controller.sync_mode = sync_mode
    controller._start_validate = MagicMock(side_effect=lambda _: events.append("start-validation"))
    controller._end_validate = MagicMock(side_effect=lambda _: events.append("end-validation"))
    controller.trainer.run.side_effect = lambda: events.append("start-trainer")
    controller.rollout_workers = [
        SimpleNamespace(run=MagicMock(side_effect=lambda: events.append("start-worker")))
    ]
    controller._check_module_exceptions = MagicMock(return_value=True)
    controller._check_finish = MagicMock(return_value=True)

    controller.run()

    if before_train:
        controller._start_validate.assert_called_once_with("actor")
        controller._end_validate.assert_called_once_with("actor")
        assert events == ["start-validation", "start-trainer", "start-worker", "end-validation"]
    else:
        controller._start_validate.assert_not_called()
        controller._end_validate.assert_not_called()
        assert events == ["start-trainer", "start-worker"]


def test_controller_preserves_startup_error_when_cleanup_also_fails():
    controller = _make_controller()
    controller.trainer.initialize_workers.side_effect = ValueError("initialization failed")
    controller._stop.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="initialization failed"):
        controller.run()

    controller._stop.assert_called_once_with()


def test_controller_propagates_cleanup_error_after_successful_run():
    controller = _make_controller()
    controller._run_training_process = MagicMock()
    controller._stop.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        controller.run()

    controller._run_training_process.assert_called_once_with()
    controller._stop.assert_called_once_with()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143)])
def test_controller_uses_interrupt_cleanup_on_process_signal(interrupt: BaseException):
    controller = _make_controller()
    controller._run_training_process = MagicMock(side_effect=interrupt)

    with pytest.raises(type(interrupt)):
        controller.run()

    controller._stop.assert_called_once_with(close_weaver_first=True)


def test_controller_raises_after_stop_on_error_cleanup():
    controller = _make_controller()
    controller._check_module_exceptions = MagicMock(return_value=False)
    controller._check_finish = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="System health check detected critical issues"):
        controller.run()

    controller._stop.assert_called_once_with()
    controller._check_finish.assert_not_called()
    titles = [
        call.kwargs["title"]
        for call in controller.activity_tracker.experiment_logger_post.call_args_list
    ]
    assert titles == ["Error!"]


def test_controller_finish_race_raises_without_complete_notification():
    controller = _make_controller()
    controller._check_module_exceptions = MagicMock(side_effect=[True, False])
    controller._check_finish = MagicMock(return_value=True)

    with pytest.raises(RuntimeError, match="System health check detected critical issues"):
        controller.run()

    controller._stop.assert_called_once_with()
    titles = [
        call.kwargs["title"]
        for call in controller.activity_tracker.experiment_logger_post.call_args_list
    ]
    assert titles == ["Error!"]


def test_controller_clean_finish_still_returns_normally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = _make_controller()
    controller._check_module_exceptions = MagicMock(return_value=True)
    controller._check_finish = MagicMock(return_value=True)

    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        controller.run()

    controller._stop.assert_called_once_with()
    assert controller._check_module_exceptions.call_count == 2
    titles = [
        call.kwargs["title"]
        for call in controller.activity_tracker.experiment_logger_post.call_args_list
    ]
    assert titles == ["Complete!"]
    assert "Training loop finished; entering controller shutdown" in caplog.messages


def test_controller_epoch_mode_finishes_when_pipeline_is_drained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    checks = []
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(training=SimpleNamespace(total_steps=None))
    controller.trainer = SimpleNamespace(get_train_step=lambda: 7)
    controller.trajectory_pool = SimpleNamespace(is_empty=lambda: checks.append("pool") or True)
    controller.dataloader = SimpleNamespace(is_finished=lambda: checks.append("dataloader") or True)
    controller.activity_tracker = SimpleNamespace(
        is_quiescent=lambda: checks.append("tracker") or True
    )
    controller.weight_sync_controller = SimpleNamespace(
        is_waiting_for_validation=lambda: checks.append("validation") or False
    )

    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        assert controller._check_finish()

    assert checks == ["dataloader", "tracker", "pool", "tracker", "validation"]
    assert "Finish check: input exhausted and pipeline drained -> stopping" in caplog.messages

    controller.dataloader = SimpleNamespace(is_finished=lambda: False)
    assert not controller._check_finish()


def test_controller_epoch_mode_rechecks_activity_after_pool_snapshot():
    checks = []
    tracker_states = iter([True, False])
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(training=SimpleNamespace(total_steps=None))
    controller.trainer = SimpleNamespace(get_train_step=lambda: 7)
    controller.dataloader = SimpleNamespace(is_finished=lambda: checks.append("dataloader") or True)
    controller.activity_tracker = SimpleNamespace(
        is_quiescent=lambda: checks.append("tracker") or next(tracker_states)
    )
    controller.trajectory_pool = SimpleNamespace(is_empty=lambda: checks.append("pool") or True)
    controller.weight_sync_controller = SimpleNamespace(
        is_waiting_for_validation=lambda: checks.append("validation") or False
    )

    assert not controller._check_finish()
    assert checks == ["dataloader", "tracker", "pool", "tracker", "validation"]


def test_controller_does_not_finish_while_final_validation_is_pending():
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(training=SimpleNamespace(total_steps=1))
    controller.trainer = SimpleNamespace(get_train_step=lambda: 1)
    controller.weight_sync_controller = SimpleNamespace(is_waiting_for_validation=lambda: True)

    assert not controller._check_finish()


def test_validation_resets_input_before_switching_workers():
    events = []
    controller = _make_controller()
    controller.validate_dataloader = SimpleNamespace(
        reset=MagicMock(side_effect=lambda: events.append("reset"))
    )
    controller.rollout_workers = [
        SimpleNamespace(
            begin_validate=MagicMock(side_effect=lambda: events.append("begin")),
            end_validate=MagicMock(side_effect=lambda: events.append("end")),
        )
    ]
    controller.validator = SimpleNamespace(
        is_complete=MagicMock(return_value=True),
        compute_and_log_metrics=MagicMock(side_effect=lambda: events.append("metrics")),
    )
    controller.weight_sync_controller = SimpleNamespace(
        get_rollout_model_version=MagicMock(return_value=1),
        end_validate=MagicMock(side_effect=lambda _: events.append("weight-sync-end")),
    )

    controller._run_validate("actor")

    assert events == ["reset", "begin", "metrics", "weight-sync-end", "end"]
    controller.validate_dataloader.reset.assert_called_once_with()


@pytest.mark.parametrize(("global_step", "sync_version"), [(190, 0), (200, 10)])
def test_validation_start_logs_global_step_and_sync_version(caplog, global_step, sync_version):
    controller = _make_controller()
    controller.activity_tracker.get_training_step.return_value = global_step
    controller.weight_sync_controller.get_rollout_model_version.return_value = sync_version
    # The trainer's public step is advanced after the sync callback returns.
    controller.trainer.get_train_step = MagicMock(return_value=global_step - 1)
    controller.validate_dataloader = SimpleNamespace(reset=MagicMock())
    worker = SimpleNamespace(begin_validate=MagicMock())
    controller.rollout_workers = [worker]

    with caplog.at_level(logging.INFO, logger="northrl.controller"):
        controller._start_validate("actor")

    messages = [
        record.getMessage()
        for record in caplog.records
        if "Starting validation:" in record.getMessage()
    ]
    assert len(messages) == 1
    assert f"global_step={global_step}" in messages[0]
    assert f"sync_version={sync_version}" in messages[0]
    controller.trainer.get_train_step.assert_not_called()
    controller.validate_dataloader.reset.assert_called_once_with()
    worker.begin_validate.assert_called_once_with()


@pytest.mark.parametrize("before_train", [False, True])
@pytest.mark.parametrize("last_task", [False, True])
@pytest.mark.parametrize("exception_policy", ["stop_on_error", "stop_on_critical"])
def test_controller_stops_after_validation_worker_error(
    monkeypatch, rollout_worker_config, before_train, last_task, exception_policy
):
    controller = _make_controller()
    controller._config.validate.enabled = True
    controller._config.validate.schedule.before_train = before_train
    controller._config.runtime.health.exception_policy = exception_policy
    controller.weight_sync_controller.is_waiting_for_validation.return_value = True
    tracker = ActivityTracker(rollout_worker_config)
    controller.activity_tracker = tracker
    monkeypatch.setattr(tracker, "update_progress", MagicMock(return_value=float("inf")))

    remaining_tasks = [1] if last_task else [1, 2]
    controller.validate_dataloader = SimpleNamespace(
        reset=MagicMock(), is_finished=lambda: not remaining_tasks
    )
    controller.validator = Validator(rollout_worker_config)
    controller.validator.set_module_references(controller.validate_dataloader)
    controller.validator.set_activity_tracker(ActivityTrackerProxy(tracker))
    controller.validator.compute_and_log_metrics = MagicMock()
    worker = MagicMock()
    controller.rollout_workers = [worker]
    controller.trainer.stop = MagicMock()
    controller.dataloader = SimpleNamespace(prepare=AsyncMock(), close=MagicMock())
    # Exercise run()'s real shutdown path, not just a mocked cleanup callback.
    del controller._stop

    failed = False

    def fail_worker_while_waiting(seconds):
        nonlocal failed
        assert seconds > 0
        assert not failed, "Validation kept waiting after the worker failed"
        failed = True
        remaining_tasks.pop(0)
        # Use the same error-reporting context as the rollout worker. On the
        # last task, validation becomes complete despite this failure.
        with pytest.raises(WeightVersionMismatchError):
            with ActivityTrackerProxy(tracker).track("rollout_worker_0", "rollout"):
                raise WeightVersionMismatchError("expected='v1', observed='v2'")

    monkeypatch.setattr(
        controller_module,
        "time",
        SimpleNamespace(monotonic=lambda: 0.0, sleep=fail_worker_while_waiting),
    )

    with pytest.raises(RuntimeError, match="critical issues"):
        controller.run()

    assert failed
    assert remaining_tasks == ([] if last_task else [2])
    assert tracker.get_error_health_status()["error_level_count"] == 1
    controller.validator.compute_and_log_metrics.assert_not_called()
    worker.end_validate.assert_not_called()
    controller.weight_sync_controller.end_validate.assert_not_called()
    worker.request_stop.assert_called_once_with()
    worker.wait_stopped.assert_called_once()
    controller.trainer.stop.assert_called_once()
    controller.dataloader.close.assert_called_once_with()


def test_controller_derives_sft_token_batch_gate_and_runtime_roles():
    controller = object.__new__(NorthRLController)
    controller._config = OmegaConf.create(
        {
            "experiment": {"seed": 17},
            "models": {"actor": {"tokenizer": "tokenizer"}},
            "data": {
                "train": {
                    "files": ["train.jsonl"],
                    "sources": None,
                    "total_samples": None,
                    "seed": 23,
                    "managed_variant_ready_timeout_seconds": 300.0,
                    "managed_variant_poll_initial_seconds": 1.0,
                    "managed_variant_poll_max_seconds": 10.0,
                    "batch_size": 4,
                    "rollout_repeat_n": None,
                    "prompt_key": "prompt",
                    "filter_prompts": False,
                    "shuffle": True,
                    "drop_last": True,
                    "render_chat": None,
                },
                "max_prompt_length": 128,
                "max_sequence_length": 1024,
            },
            "algorithm": {"type": "sft"},
            "rollout": {},
            "training": {"epochs": 3, "weight_sync": {"mode": "no-sync"}},
        }
    )

    dataloader_config = controller._derive_train_dataloader_config()
    pool_config = controller._derive_trajectory_pool_config()

    assert controller._get_module_class_for_role(NorthRLRole.DATA_LOADER) is SFTStreamingDataLoader
    assert controller._get_module_class_for_role(NorthRLRole.ROLLOUT_WORKER) is SFTRolloutWorker
    assert (
        controller._get_module_class_for_role(NorthRLRole.TRAINER) is RemoteApiCrossEntropyTrainer
    )
    assert dataloader_config.rollout_repeat_n == 1
    assert dataloader_config.keep_batch_order is True
    assert dataloader_config.max_sequence_length == 1024
    assert dataloader_config.epochs == 3
    assert dataloader_config.seed == 23
    assert dataloader_config.cycle is False
    assert pool_config.group_size == 1
    assert pool_config.key_list == []
    assert pool_config.need_sampling is False


def test_controller_derives_cycle_for_step_based_rl_training():
    controller = object.__new__(NorthRLController)
    controller._config = OmegaConf.create(
        {
            "experiment": {"seed": 17},
            "models": {"actor": {"tokenizer": "tokenizer"}},
            "data": {
                "train": {
                    "files": ["train.jsonl"],
                    "sources": None,
                    "total_samples": None,
                    "managed_variant_ready_timeout_seconds": 300.0,
                    "managed_variant_poll_initial_seconds": 1.0,
                    "managed_variant_poll_max_seconds": 10.0,
                    "batch_size": 32,
                    "rollout_repeat_n": 16,
                    "prompt_key": "prompt",
                    "filter_prompts": False,
                    "shuffle": True,
                    "drop_last": True,
                    "render_chat": None,
                    # This option belongs to deterministic SFT mixing. RL keeps
                    # main's experiment-seed behavior even if it is present.
                    "seed": 99,
                },
                "max_prompt_length": 4096,
                "max_sequence_length": 32768,
            },
            "algorithm": {"type": "grpo"},
            "training": {"total_steps": 3000, "weight_sync": {"mode": "sync"}},
        }
    )

    dataloader_config = controller._derive_train_dataloader_config()

    assert dataloader_config.cycle is True
    assert dataloader_config.seed == 17


def test_controller_selects_e2b_rollout_worker():
    controller = object.__new__(NorthRLController)
    controller._config = OmegaConf.create(
        {
            "algorithm": {"type": "grpo"},
            "rollout": {"type": "e2b", "execution": {}},
        }
    )

    assert controller._get_module_class_for_role(NorthRLRole.ROLLOUT_WORKER) is E2BRolloutWorker


def test_controller_stop_closes_shared_weaver_holder(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    close_awaitable = object()
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(
        runtime=SimpleNamespace(progress=SimpleNamespace(update_interval_s=1.0))
    )
    controller.use_weaver = True
    controller.activity_tracker = SimpleNamespace(
        update_progress=MagicMock(),
        get_running_status_summary=MagicMock(return_value="idle"),
        wait_quiescent=MagicMock(side_effect=lambda **_: events.append("wait") or True),
    )
    controller.trainer = SimpleNamespace(
        stop=MagicMock(side_effect=lambda **_: events.append("trainer-stop"))
    )
    controller.rollout_workers = [
        SimpleNamespace(
            request_stop=MagicMock(side_effect=lambda: events.append("worker-0-request")),
            wait_stopped=MagicMock(side_effect=lambda **_: events.append("worker-0-wait")),
        ),
        SimpleNamespace(
            request_stop=MagicMock(side_effect=lambda: events.append("worker-1-request")),
            wait_stopped=MagicMock(side_effect=lambda **_: events.append("worker-1-wait")),
        ),
    ]
    controller._weaver_service_holder = SimpleNamespace(
        close=MagicMock(side_effect=lambda: events.append("holder-close") or close_awaitable)
    )
    controller.dataloader = SimpleNamespace(
        close=MagicMock(side_effect=lambda: events.append("dataloader-close"))
    )

    def run_close(awaitable) -> None:
        assert awaitable is close_awaitable
        events.append("run-coro")

    monkeypatch.setattr(controller_module, "run_coro", run_close)

    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        controller._stop()

    assert events == [
        "worker-0-request",
        "worker-1-request",
        "trainer-stop",
        "worker-0-wait",
        "worker-1-wait",
        "wait",
        "holder-close",
        "run-coro",
        "dataloader-close",
    ]
    assert "Requesting shutdown from 2 rollout worker(s)" in caplog.messages
    assert "Stopping trainer; waiting for active training work to finish" in caplog.messages
    assert "Waiting for 2 rollout worker(s) to stop (shared timeout: 30.0s)" in caplog.messages
    assert "Closing dataloader" in caplog.messages
    assert "WeaverServiceHolder close coroutine completed during normal cleanup" in caplog.messages


def test_controller_interrupt_closes_weaver_before_stopping_trainer(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    close_awaitable = object()
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(
        runtime=SimpleNamespace(progress=SimpleNamespace(update_interval_s=1.0))
    )
    controller.use_weaver = True
    controller.activity_tracker = SimpleNamespace(
        update_progress=MagicMock(),
        get_running_status_summary=MagicMock(return_value="idle"),
        wait_quiescent=MagicMock(side_effect=lambda **_: events.append("wait") or True),
    )
    controller.trainer = SimpleNamespace(
        stop=MagicMock(side_effect=lambda **_: events.append("trainer-stop"))
    )
    controller.rollout_workers = [
        SimpleNamespace(
            request_stop=MagicMock(side_effect=lambda: events.append("worker-request")),
            wait_stopped=MagicMock(side_effect=lambda **_: events.append("worker-wait")),
        )
    ]
    controller._weaver_service_holder = SimpleNamespace(
        close=MagicMock(side_effect=lambda: events.append("holder-close") or close_awaitable)
    )
    controller.dataloader = SimpleNamespace(
        close=MagicMock(side_effect=lambda: events.append("dataloader-close"))
    )

    def run_close(awaitable) -> None:
        assert awaitable is close_awaitable
        events.append("run-coro")

    monkeypatch.setattr(controller_module, "run_coro", run_close)
    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        controller._stop(close_weaver_first=True)

    assert events == [
        "worker-request",
        "holder-close",
        "run-coro",
        "trainer-stop",
        "worker-wait",
        "wait",
        "dataloader-close",
    ]
    assert (
        "WeaverServiceHolder close coroutine completed during interrupt cleanup" in caplog.messages
    )


def test_controller_stop_uses_one_timeout_budget_for_all_rollout_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    wait_timeouts: list[float] = []

    class _Worker:
        def request_stop(self) -> None:
            pass

        def wait_stopped(self, timeout: float, **kwargs) -> None:
            wait_timeouts.append(timeout)
            clock[0] += timeout

    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(
        runtime=SimpleNamespace(progress=SimpleNamespace(update_interval_s=1.0))
    )
    controller.use_weaver = False
    controller.activity_tracker = SimpleNamespace(
        update_progress=MagicMock(),
        get_running_status_summary=MagicMock(return_value="idle"),
        wait_quiescent=MagicMock(return_value=True),
    )
    controller.trainer = SimpleNamespace(stop=MagicMock())
    controller.rollout_workers = [_Worker(), _Worker(), _Worker()]
    controller.dataloader = SimpleNamespace(close=MagicMock())
    monkeypatch.setattr(controller_module.time, "monotonic", lambda: clock[0])

    controller._stop()

    assert wait_timeouts == [30.0, 0.0, 0.0]


def test_controller_stop_closes_dataloader_when_weaver_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(
        runtime=SimpleNamespace(progress=SimpleNamespace(update_interval_s=1.0))
    )
    controller.use_weaver = True
    controller.activity_tracker = SimpleNamespace(
        update_progress=MagicMock(),
        get_running_status_summary=MagicMock(return_value="idle"),
        wait_quiescent=MagicMock(return_value=True),
    )
    controller.trainer = SimpleNamespace(stop=MagicMock())
    controller.rollout_workers = []
    controller._weaver_service_holder = SimpleNamespace(close=MagicMock(return_value=object()))
    controller.dataloader = SimpleNamespace(close=MagicMock())
    monkeypatch.setattr(
        controller_module,
        "run_coro",
        MagicMock(side_effect=RuntimeError("weaver close failed")),
    )

    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        with pytest.raises(RuntimeError, match="weaver close failed"):
            controller._stop()

    controller.dataloader.close.assert_called_once_with()
    assert "WeaverServiceHolder close coroutine failed during normal cleanup" in caplog.messages


def test_controller_interrupt_continues_cleanup_when_weaver_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = object.__new__(NorthRLController)
    controller._config = SimpleNamespace(
        runtime=SimpleNamespace(progress=SimpleNamespace(update_interval_s=1.0))
    )
    controller.use_weaver = True
    controller.activity_tracker = SimpleNamespace(
        update_progress=MagicMock(),
        get_running_status_summary=MagicMock(return_value="idle"),
        wait_quiescent=MagicMock(return_value=True),
    )
    controller.trainer = SimpleNamespace(stop=MagicMock())
    controller.rollout_workers = []
    controller._weaver_service_holder = SimpleNamespace(close=MagicMock(return_value=object()))
    controller.dataloader = SimpleNamespace(close=MagicMock())
    monkeypatch.setattr(
        controller_module,
        "run_coro",
        MagicMock(side_effect=RuntimeError("weaver close failed")),
    )

    with caplog.at_level(logging.INFO, logger=controller_module.__name__):
        controller._stop(close_weaver_first=True)

    controller.trainer.stop.assert_called_once_with(
        on_wait=controller.activity_tracker.update_progress, wait_interval=1.0
    )
    controller.dataloader.close.assert_called_once_with()
    assert "WeaverServiceHolder close coroutine failed during interrupt cleanup" in caplog.messages


@pytest.mark.parametrize("progress_interval", [0.25, 1.5])
def test_controller_progress_preserves_health_and_log_intervals(monkeypatch, progress_interval):
    controller = _make_controller()
    controller._config.runtime.progress.update_interval_s = progress_interval
    clock = [0.0]
    next_progress_at = [0.0]
    progress_times = []
    health_times = []

    def update_progress():
        if clock[0] >= next_progress_at[0]:
            progress_times.append(clock[0])
            next_progress_at[0] = clock[0] + progress_interval
        return next_progress_at[0]

    def sleep(seconds):
        assert seconds > 0
        clock[0] += seconds
        assert clock[0] <= 4.0, "Controller did not finish"

    monkeypatch.setattr(
        controller_module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    controller.activity_tracker.update_progress.side_effect = update_progress
    controller._check_module_exceptions = MagicMock(
        side_effect=lambda: health_times.append(clock[0]) or True
    )
    controller._check_finish = MagicMock(side_effect=lambda: clock[0] >= 3.0)

    controller.run()

    assert progress_times == pytest.approx(
        [index * progress_interval for index in range(int(3.0 / progress_interval) + 1)]
    )
    assert health_times == [0.0, 1.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize(
    ("exception_policy", "error_count"), [("stop_on_error", 0), ("continue", 1)]
)
def test_controller_refreshes_progress_while_waiting_for_validation(
    monkeypatch, exception_policy, error_count
):
    controller = _make_controller()
    controller._config.runtime.health.exception_policy = exception_policy
    controller.activity_tracker.get_error_health_status.return_value = {
        "error_level_count": error_count
    }
    clock = [0.0]
    progress_times = []
    events = []

    def update_progress():
        progress_times.append(clock[0])
        return clock[0] + 0.25

    def sleep(seconds):
        clock[0] += seconds
        assert clock[0] <= 1.0, "Validation did not finish"

    monkeypatch.setattr(
        controller_module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    controller.activity_tracker.update_progress.side_effect = update_progress
    controller.validator = SimpleNamespace(
        is_complete=lambda: clock[0] >= 0.75,
        compute_and_log_metrics=lambda: events.append("metrics"),
    )
    controller.rollout_workers = [
        SimpleNamespace(end_validate=lambda: events.append("end-worker-validation"))
    ]
    controller.weight_sync_controller.end_validate.side_effect = lambda _: events.append("release")

    controller._end_validate("actor")

    assert progress_times == [0.0, 0.25, 0.5]
    assert events == ["metrics", "release", "end-worker-validation"]


def test_controller_passes_progress_refresh_to_shutdown_waits():
    controller = _make_controller()
    controller._config.runtime.progress.update_interval_s = 0.25
    controller.activity_tracker.wait_quiescent.return_value = True
    events = []

    def wait_for_stop(*, on_wait, wait_interval, **kwargs):
        assert on_wait == controller.activity_tracker.update_progress
        assert wait_interval == 0.25
        on_wait()
        events.append("waited")

    controller.trainer.stop = MagicMock(side_effect=wait_for_stop)
    controller.rollout_workers = [
        SimpleNamespace(request_stop=MagicMock(), wait_stopped=MagicMock(side_effect=wait_for_stop))
    ]
    controller.dataloader = SimpleNamespace(close=MagicMock())

    NorthRLController._stop(controller)

    assert events == ["waited", "waited"]
    assert controller.activity_tracker.update_progress.call_count == 3
    controller.dataloader.close.assert_called_once_with()


@pytest.mark.parametrize("class_name", [None, "RecipeRolloutWorker"])
def test_custom_rollout_loads_from_flat_recipe_config(tmp_path, rollout_worker_config, class_name):
    worker_dir = tmp_path / "workers"
    worker_dir.mkdir()
    name = class_name or "CustomRolloutWorker"
    (worker_dir / "custom.py").write_text(
        "from northrl.rollout_worker import BaseRolloutWorker\n"
        f"class {name}(BaseRolloutWorker):\n"
        "    async def rollout(self, task):\n"
        "        return self._config.rollout.params.prefix + task['value']\n",
        encoding="utf-8",
    )
    rollout_worker_config._config_file_path = str(tmp_path / "recipe.yaml")
    rollout_worker_config.rollout.update(
        type="custom",
        custom_module_path="workers/custom.py",
        params={"prefix": "recipe:"},
    )
    if class_name is not None:
        rollout_worker_config.rollout.custom_class_name = class_name
    controller = _make_controller()
    controller._config = rollout_worker_config

    worker_class = controller._get_module_class_for_role(NorthRLRole.ROLLOUT_WORKER)
    worker = worker_class(rollout_worker_config)

    assert worker_class.__name__ == name
    assert asyncio.run(worker.rollout({"value": "task"})) == "recipe:task"


@pytest.mark.parametrize(
    "store", [{}, {"enabled": False}, {"enabled": True, "path": "/tmp/run-metrics"}]
)
def test_controller_wires_weaver_observer_and_storage(monkeypatch, rollout_worker_config, store):
    monkeypatch.setenv("WEAVER_API_KEY", "test-key")
    holder_factory = MagicMock()
    holder_factory.return_value.configure_sampling_flags = AsyncMock()
    monkeypatch.setattr("northrl.weaver.WeaverServiceHolder", holder_factory)
    controller = _make_controller()
    controller._config = rollout_worker_config
    controller._config.backend.metrics_store = store
    controller._init_weaver_service()
    kwargs = holder_factory.call_args.kwargs
    assert kwargs["metrics_store"] == store
    assert (
        kwargs["metric_observer"]
        == controller.activity_tracker.experiment_logger.queue_weaver_metrics
    )
