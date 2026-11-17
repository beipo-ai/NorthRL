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

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as remote_api_trainer_module
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.data_dumper import DataDumper
from northrl.utils.metric_catalog import prepare_wandb_metrics
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


class _DummyDataDumper:
    enabled = False

    def __init__(self):
        self.rollout_round_calls = []

    def should_dump(self, data_type, step):
        del data_type, step
        return False

    def dump_prepared_trajectories(self, **kwargs):
        del kwargs

    def dump_rollout_round(self, step, completed_step, records):
        self.rollout_round_calls.append((step, completed_step, records))

    def dump_datums(self, **kwargs):
        del kwargs

    def dump_training_metrics(self, **kwargs):
        del kwargs


@dataclass
class _FakeWeightSyncController:
    sync_mode: str = "sync"
    notifications: list[dict] = field(default_factory=list)

    def is_waiting_for_validation(self) -> bool:
        return False

    def train_worker_notify_weight_update(
        self,
        *,
        worker_name,
        model_tag,
        sampling_path,
        weight_version,
        weights_updated=True,
        continue_training=True,
        completed_step=None,
    ):
        self.notifications.append(
            {
                "worker_name": worker_name,
                "model_tag": model_tag,
                "sampling_path": sampling_path,
                "weight_version": weight_version,
                "weights_updated": weights_updated,
                "continue_training": continue_training,
                "completed_step": completed_step,
            }
        )


class _DummyRemoteApiTrainer(RemoteApiTrainer):
    def __init__(self, config):
        super().__init__(config)
        self._activity_tracker = _DummyActivityTracker()
        self._weight_sync_controller = _FakeWeightSyncController()

    async def _prepare_trajectories(self, trajectories, metrics):
        del metrics
        return trajectories

    def _convert_to_datums(self, trajectories):
        del trajectories
        count = getattr(self, "_datum_count", 1)
        return [
            {
                "input_tokens": [idx + 1, idx + 2],
                "loss_fn_inputs": {
                    "target_tokens": [idx + 2],
                    "logprobs": [0.0],
                    "advantages": [1.0],
                    "loss_mask": [1.0],
                },
            }
            for idx in range(count)
        ]


class _DummyActivityTracker:
    def __init__(self):
        self.posts: list[dict] = []
        self.training_steps: list[int] = []
        self.flushed_steps: list[int] = []

    def track(self, module, work):
        del module, work
        return nullcontext()

    def experiment_logger_post(self, **kwargs):
        self.posts.append(kwargs)

    def experiment_logger_flush(self, step):
        self.flushed_steps.append(step)

    def set_training_step(self, step):
        self.training_steps.append(step)


class _SingleBatchTrajectoryPool:
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.returned = False

    def get_trajectories(self):
        if self.returned:
            return None
        self.returned = True
        return self.trajectories

    def get_discarded_metrics(self):
        return {}


@dataclass
class _FakeWeaverHolder:
    custom_calls: list[dict] = field(default_factory=list)
    builtin_calls: list[dict] = field(default_factory=list)
    compute_logprobs_calls: list[dict] = field(default_factory=list)
    optim_calls: list[dict] = field(default_factory=list)
    saved_paths: list[str] = field(default_factory=list)

    async def compute_target_logprobs_batch(
        self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
    ):
        call = {
            "tokens_batch": [list(tokens) for tokens in tokens_batch],
            "sampling_masks": sampling_masks,
        }
        if loss_fn_config is not None:
            call["loss_fn_config"] = dict(loss_fn_config)
        self.compute_logprobs_calls.append(call)
        return [[-0.1 * idx for idx in range(1, len(tokens))] for tokens in tokens_batch]

    async def forward_backward_custom(
        self, *, datums_data, custom_loss_spec, mini_batch_sizes=None
    ):
        call = {"datums_data": datums_data, "custom_loss_spec": custom_loss_spec}
        if mini_batch_sizes is not None:
            call["mini_batch_sizes"] = mini_batch_sizes
        self.custom_calls.append(call)
        call_index = float(len(self.custom_calls))
        return {
            "loss": call_index,
            "metrics": {"loss": call_index, "custom_metric": call_index * 10.0},
        }

    async def forward_backward(self, **kwargs):
        self.builtin_calls.append(kwargs)
        return {"loss": 2.0, "metrics": {"loss": 2.0, "builtin_metric": 3.0}}

    async def optim_step(self, **kwargs):
        self.optim_calls.append(kwargs)
        return {
            "status": "success",
            "metrics": {"optim/step_time": float(len(self.optim_calls)) * 0.5},
        }

    async def save_weights_for_sampler(self, name):
        self.saved_paths.append(name)
        return f"/tmp/{name}"


def _make_config(loss=None, batch_size=None, mini_batch_size=None):
    return OmegaConf.create(
        {
            "training": {
                "total_steps": 4,
                "save": {"dir": "", "every_n_steps": 0},
            },
            "algorithm": {
                "type": "grpo",
                "loss": loss
                or {
                    "mode": "importance_sampling",
                    "params": {
                        "clip": {
                            "truncation_c": None,
                            "ratio": None,
                            "ratio_low": None,
                            "ratio_high": None,
                            "ratio_c": None,
                        }
                    },
                    "reduction": {"mode": "datum_mean"},
                },
                "entropy_coeff": 0.0,
                "sampling_logprob_source": "inference",
                "optim": {
                    "lr": 2e-6,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 1e-2,
                    "eps": 1e-8,
                    "grad_clip_norm": 1.0,
                    "batch_size": batch_size,
                    "mini_batch_size": mini_batch_size,
                },
            },
            "runtime": {"debug": {"enable_data_dump": False}},
        }
    )


def _make_trajectory():
    return Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=1.0, is_val=False)


def test_remote_api_trainer_routes_custom_loss_through_weaver(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(
        _make_config(
            loss={
                "mode": "custom",
                "params": {
                    "name": "direct_logprob",
                    "params": {"scale": 2.0},
                    "base_loss_fn": "importance_sampling",
                },
                "reduction": {"mode": "datum_mean"},
            }
        )
    )
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert len(holder.custom_calls) == 1
    assert trainer._learning_rate == 2e-6
    assert trainer._beta2 == 0.999
    assert holder.custom_calls[0]["custom_loss_spec"] == {
        "name": "direct_logprob",
        "params": {"scale": 2.0, "entropy_coeff": 0.0},
    }
    assert len(holder.optim_calls) == 1
    assert holder.builtin_calls == []
    assert holder.saved_paths == ["northrl_actor_v000001"]
    assert metrics["weaver/loss"] == 1.0
    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000001",
        "northrl_actor_v000001",
    )
    assert "training/custom_metric" not in metrics
    assert "weaver/custom_metric" not in metrics  # No declared service reducer.
    assert "training/optim/step_time" not in metrics
    assert "weaver/optim/step_time" not in metrics  # No declared service reducer.
    assert metrics["training/skipped_flag"] == 0


def test_remote_api_trainer_does_not_publish_sampler_weights_in_no_sync(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    trainer._weight_sync_controller = _FakeWeightSyncController(  # pylint: disable=protected-access
        sync_mode="no-sync"
    )
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert metrics["training/skipped_flag"] == 0
    assert (sampling_path, weight_version) == (None, None)
    assert not holder.saved_paths
    assert "timing_weights/weight_export_seconds" not in metrics
    assert "training/weight_export_success_flag" not in metrics


def test_remote_api_trainer_allows_null_base_loss_with_custom_callback(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(
        _make_config(
            loss={
                "mode": "custom",
                "params": {
                    "name": "direct_logprob",
                    "params": {"scale": 1.0},
                    "base_loss_fn": None,
                },
                "reduction": {"mode": "datum_mean"},
            },
        )
    )
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert trainer._loss_fn is None
    assert len(holder.custom_calls) == 1
    assert holder.builtin_calls == []
    assert metrics["weaver/loss"] == 1.0
    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000001",
        "northrl_actor_v000001",
    )


def test_remote_api_trainer_keeps_builtin_path_without_custom_loss(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert holder.custom_calls == []
    assert len(holder.builtin_calls) == 1
    assert len(holder.optim_calls) == 1
    assert metrics["weaver/loss"] == 2.0
    assert "training/builtin_metric" not in metrics
    assert "weaver/builtin_metric" not in metrics  # No declared service reducer.
    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000001",
        "northrl_actor_v000001",
    )


@pytest.mark.parametrize("submit_ahead", [0, 1])
def test_weaver_entropy_reaches_wandb_across_nested_tasks_and_optimizer_batches(
    monkeypatch, submit_ahead
):
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda *args, **kwargs: _DummyDataDumper(),
    )
    trainer = _DummyRemoteApiTrainer(_make_config(batch_size=2, mini_batch_size=1))
    trainer._datum_count = 4
    trainer._submit_ahead = submit_ahead
    # Each request contains two backend task summaries. Effective entropy token
    # counts differ from the generic token counts and the client's loss weights.
    summaries = iter(
        (
            ((0.0, 0), (4.0, 2)),
            ((5.0, 1), (9.0, 3)),
            ((4.0, 4), (0.0, 0)),
            ((8.0, 2), (10.0, 5)),
        )
    )

    class _EntropyWeaverHolder(_FakeWeaverHolder):
        collect_forward_backward_result = staticmethod(
            WeaverServiceHolder.collect_forward_backward_result
        )

        async def submit_forward_backward(self, **kwargs):
            self.builtin_calls.append(kwargs)
            assert kwargs["loss_fn_config"]["entropy_coeff"] == 0.0
            assert len(kwargs["datums_data"]) == 1
            return {
                "result": {
                    "forward_backward_task_results": {
                        str(index): {
                            "metrics": {
                                "loss": 1.0,
                                "tokens": 100,
                                "entropy": entropy_sum / tokens if tokens else 0.0,
                                "entropy:sum": entropy_sum,
                                "entropy:tokens": tokens,
                            }
                        }
                        for index, (entropy_sum, tokens) in enumerate(next(summaries))
                    }
                }
            }

        async def forward_backward(self, **kwargs):
            handle = await self.submit_forward_backward(**kwargs)
            return await self.collect_forward_backward_result(
                handle,
                datums_data=kwargs["datums_data"],
                loss_fn_config=kwargs["loss_fn_config"],
            )

    holder = _EntropyWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train([_make_trajectory()]))
    payload = prepare_wandb_metrics(metrics)

    assert len(holder.builtin_calls) == 4
    assert len(holder.optim_calls) == 2
    assert metrics["training/forward_backward_call_count"] == 4
    assert metrics["training/optimizer_step_count"] == 2
    assert payload["01_algorithm/56_weaver/entropy"] == pytest.approx(40.0 / 17)
    assert payload["01_algorithm/57_weaver/entropy:sum"] == 40.0
    assert payload["01_algorithm/58_weaver/entropy:tokens"] == 17


def test_remote_api_trainer_applies_batch_updates_for_custom_loss(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(
        _make_config(
            loss={
                "mode": "custom",
                "params": {
                    "name": "direct_logprob",
                    "params": {"scale": 2.0},
                    "base_loss_fn": "importance_sampling",
                },
                "reduction": {"mode": "datum_mean"},
            },
            batch_size=2,
        )
    )
    trainer._datum_count = 3
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert len(holder.custom_calls) == 2
    assert len(holder.optim_calls) == 2
    assert holder.builtin_calls == []
    assert [len(call["datums_data"]) for call in holder.custom_calls] == [2, 2]
    assert metrics["weaver/loss"] == 1.5
    assert "training/custom_metric" not in metrics
    assert "weaver/custom_metric" not in metrics  # No declared service reducer.
    assert "training/optim/step_time" not in metrics
    assert "weaver/optim/step_time" not in metrics  # No declared service reducer.
    assert "training/num_batches" not in metrics
    assert metrics["training/optimizer_step_count"] == 2
    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000001",
        "northrl_actor_v000001",
    )

    assert trainer._activity_tracker.posts == []


def test_batches_emit_one_completed_round_record(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(
        _make_config(
            loss={
                "mode": "custom",
                "params": {
                    "name": "direct_logprob",
                    "params": {"scale": 2.0},
                    "base_loss_fn": "importance_sampling",
                },
                "reduction": {"mode": "datum_mean"},
            },
            batch_size=2,
        )
    )
    trainer._total_train_steps = 1
    trainer._datum_count = 3
    trainer._trajectory_pool = _SingleBatchTrajectoryPool([_make_trajectory()])
    trainer.set_module_name("trainer")
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    async def on_training_complete():
        return None

    monkeypatch.setattr(trainer, "_on_training_complete", on_training_complete)
    run_coro(trainer._main_loop())

    assert trainer.get_train_step() == 1
    assert len(holder.custom_calls) == 2
    assert len(trainer._data_dumper.rollout_round_calls) == 1
    dump_step, completed_step, records = trainer._data_dumper.rollout_round_calls[0]
    assert (dump_step, completed_step) == (0, 1)
    assert len(records) == 1
    assert len(trainer._activity_tracker.posts) == 1
    post = trainer._activity_tracker.posts[0]
    assert post["step"] == 1
    assert post["data"]["training/train_step"] == 1
    assert "training/global_step" not in post["data"]
    assert post["data"]["training/optimizer_step_count"] == 2
    assert "training/update_step" not in post["data"]
    assert post["data"]["timing/trainer_wait_for_batch_seconds"] == 0.0
    assert trainer._weight_sync_controller.notifications == [
        {
            "worker_name": "trainer",
            "model_tag": "actor",
            "sampling_path": "/tmp/northrl_actor_v000001",
            "weight_version": "northrl_actor_v000001",
            "weights_updated": True,
            "continue_training": False,
            "completed_step": 1,
        }
    ]


def test_skipped_round_keeps_rollout_metrics_without_publishing_weights(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    trainer._datum_count = 0
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert metrics["training/skipped_flag"] == 1
    assert "timing_weights/weight_export_seconds" not in metrics
    assert "training/weight_export_success_flag" not in metrics
    assert (sampling_path, weight_version) == (None, None)
    assert metrics["trajectory/reward_mean"] == pytest.approx(1.0)
    assert metrics["batch/segment_count"] == 1
    assert not any(key.startswith("datum/") for key in metrics)
    assert metrics["batch/trainable_segment_count"] == 0
    assert holder.custom_calls == []
    assert holder.builtin_calls == []
    assert holder.optim_calls == []
    assert holder.saved_paths == []
    assert trainer._activity_tracker.posts == []

    # BaseTrainer records the skipped round as completed. The next successful
    # round is therefore published with step 2; step 1 records the skipped batch.
    trainer.set_train_step(1)
    trainer._datum_count = 1
    _, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))
    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000002",
        "northrl_actor_v000002",
    )
    assert holder.saved_paths == ["northrl_actor_v000002"]


def test_resume_uses_next_train_step_for_weights(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    trainer.set_train_step(10)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    _, sampling_path, weight_version = run_coro(trainer.train([_make_trajectory()]))

    assert (sampling_path, weight_version) == (
        "/tmp/northrl_actor_v000011",
        "northrl_actor_v000011",
    )
    assert trainer.get_train_step() == 10
    assert holder.saved_paths == ["northrl_actor_v000011"]


def test_recompute_logprobs_passes_target_aligned_sampling_mask(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    trajectory = Trajectory(
        tokens=[10, 11, 12, 13],
        loss_mask=[0, 0, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields={
            "old_log_probs": [0.0, -1.0, -1.0, -1.0],
            "sampling_mask": [[12, 99], [13, 98]],
        },
    )
    metrics = {}

    run_coro(trainer._recompute_logprobs([trajectory], metrics))

    assert holder.compute_logprobs_calls == [
        {
            "tokens_batch": [[10, 11, 12, 13]],
            "sampling_masks": [[[11], [12, 99], [13, 98]]],
        }
    ]
    assert trajectory["old_log_probs"] == pytest.approx([0.0, -0.1, -0.2, -0.3])
    assert not any(key.startswith("recompute_logprobs/") for key in metrics)
    assert metrics["timing/recompute_logprobs"] >= 0


def test_recompute_logprobs_batches_multiple_trajectories_even_with_batch_size(
    monkeypatch,
):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    config = _make_config(batch_size=1)
    trainer = _DummyRemoteApiTrainer(config)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    trajectories = [
        Trajectory(
            tokens=[10, 11, 12],
            loss_mask=[0, 0, 1],
            reward=1.0,
            is_val=False,
            extra_fields={"old_log_probs": [0.0, -1.0, -1.0]},
        ),
        Trajectory(
            tokens=[20, 21, 22, 23],
            loss_mask=[0, 0, 1, 1],
            reward=1.0,
            is_val=False,
            extra_fields={"old_log_probs": [0.0, -1.0, -1.0, -1.0]},
        ),
    ]
    metrics = {}

    run_coro(trainer._recompute_logprobs(trajectories, metrics))

    assert holder.compute_logprobs_calls == [
        {
            "tokens_batch": [[10, 11, 12], [20, 21, 22, 23]],
            "sampling_masks": None,
        }
    ]
    assert trajectories[0]["old_log_probs"] == pytest.approx([0.0, -0.1, -0.2])
    assert trajectories[1]["old_log_probs"] == pytest.approx([0.0, -0.1, -0.2, -0.3])
    assert not any(key.startswith("recompute_logprobs/") for key in metrics)
    assert metrics["timing/recompute_logprobs"] >= 0


def test_recompute_logprobs_rejects_mixed_sampling_mask_presence(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )

    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    trajectories = [
        Trajectory(
            tokens=[10, 11, 12],
            loss_mask=[0, 0, 1],
            reward=1.0,
            is_val=False,
            extra_fields={"old_log_probs": [0.0, -1.0, -1.0]},
        ),
        Trajectory(
            tokens=[20, 21, 22],
            loss_mask=[0, 0, 1],
            reward=1.0,
            is_val=False,
            extra_fields={
                "old_log_probs": [0.0, -1.0, -1.0],
                "sampling_mask": [[22, 99]],
            },
        ),
        Trajectory(
            tokens=[30, 31, 32],
            loss_mask=[0, 0, 1],
            reward=1.0,
            is_val=False,
            extra_fields={"old_log_probs": [0.0, -1.0, -1.0]},
        ),
    ]
    metrics = {}

    with pytest.raises(AssertionError, match="sampling_mask must be present for all sequences"):
        run_coro(trainer._recompute_logprobs(trajectories, metrics))

    assert holder.compute_logprobs_calls == []


def test_weaver_compute_target_logprobs_batch_uses_training_forward_backend():
    class _FakeTrainingClient:
        def __init__(self):
            self.forward_calls = []

        async def forward(self, data, loss_fn, wait=False):
            self.forward_calls.append({"data": data, "loss_fn": loss_fn, "wait": wait})
            return {
                "result": {
                    "loss_fn_outputs": [
                        {"logprobs": {"data": [-0.1, -0.2, -0.3]}},
                        {"logprobs": {"data": [-0.4, -0.5]}},
                    ],
                }
            }

    holder = object.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._training_client = _FakeTrainingClient()
    holder._sampling_client = object()
    holder._tokenizer = object()

    logprobs = run_coro(
        holder.compute_target_logprobs_batch(
            [[10, 11, 12, 13], [20, 21, 22]],
            sampling_masks=[
                [[11], [12, 99], [13, 98]],
                [[21], [22, 88]],
            ],
        )
    )

    assert logprobs == [[-0.1, -0.2, -0.3], [-0.4, -0.5]]
    assert len(holder._training_client.forward_calls) == 1
    call = holder._training_client.forward_calls[0]
    assert call["loss_fn"] == "forward_logprob"
    assert call["wait"] is False
    datums = call["data"]
    assert [datum.model_input.to_ints() for datum in datums] == [[10, 11, 12], [20, 21]]
    assert [datum.loss_fn_inputs["target_tokens"].tolist() for datum in datums] == [
        [11, 12, 13],
        [21, 22],
    ]
    assert [datum.loss_fn_inputs["sampling_mask"] for datum in datums] == [
        [[11], [12, 99], [13, 98]],
        [[21], [22, 88]],
    ]


@pytest.mark.parametrize("failure", [RuntimeError("retry"), [], [[-0.1, -0.2]]])
def test_recompute_retries_rpc_and_invalid_results_up_to_three_times(monkeypatch, failure):
    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    holder.compute_target_logprobs_batch = AsyncMock(side_effect=[failure] * 3 + [[[-0.4]]])
    trainer.set_service_holder(holder)
    trajectory = _make_trajectory()

    run_coro(trainer._recompute_logprobs([trajectory], {}))

    assert holder.compute_target_logprobs_batch.await_count == 4
    assert trajectory["old_log_probs"] == [0.0, -0.4]


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        (RuntimeError("failed"), 4),
        (asyncio.CancelledError(), 1),
        (FutureCancelledError(), 1),
    ],
)
@pytest.mark.parametrize("submit_ahead", [0, 1])
def test_recompute_failure_or_cancellation_aborts_training(
    monkeypatch, error, expected_calls, submit_ahead
):
    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    config = _make_config()
    config.algorithm.sampling_logprob_source = "recomputed"
    trainer = _DummyRemoteApiTrainer(config)
    trainer._submit_ahead = submit_ahead
    holder = _FakeWeaverHolder()
    holder.compute_target_logprobs_batch = AsyncMock(side_effect=error)
    trainer.set_service_holder(holder)

    async def train_and_capture():
        with pytest.raises(type(error)) as caught:
            await trainer.train([_make_trajectory()])
        return caught.value

    caught = run_coro(train_and_capture())
    # asyncio tasks may synthesize an equivalent cancellation at the pipeline boundary.
    if not isinstance(error, asyncio.CancelledError):
        assert caught is error
    assert holder.compute_target_logprobs_batch.await_count == expected_calls
    assert holder.builtin_calls == holder.custom_calls == holder.optim_calls == []


def test_custom_loss_keeps_whole_optimizer_batch_callback_with_multiple_fb_chunks(monkeypatch):
    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = _DummyRemoteApiTrainer(
        _make_config(
            loss={
                "mode": "custom",
                "params": {
                    "name": "direct_logprob",
                    "params": {"scale": 2.0},
                    "base_loss_fn": "importance_sampling",
                },
                "reduction": {"mode": "datum_mean"},
            },
            batch_size=4,
            mini_batch_size=3,
        )
    )
    trainer._datum_count = 8
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train([_make_trajectory()]))

    assert [len(call["datums_data"]) for call in holder.custom_calls] == [4, 4]
    assert [call["mini_batch_sizes"] for call in holder.custom_calls] == [[3, 1], [3, 1]]
    assert len(holder.optim_calls) == 2
    assert "training/num_batches" not in metrics
    assert metrics["training/optimizer_step_count"] == 2
    assert metrics["training/forward_backward_call_count"] == 4
    assert metrics["training/optimizer_step_count"] == 2


def test_completed_empty_group_advances_without_training_or_export(monkeypatch):
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    trajectory = Trajectory(
        segments=[],
        reward=0.5,
        extra_fields={
            "group_id": "empty",
            "run_id": 0,
            "rollout_id": "empty-0",
            "expected_rollouts": 1,
        },
    )
    metrics, sampling_path, weight_version = run_coro(trainer.train([trajectory]))
    assert metrics["training/skipped_flag"] == 1
    assert metrics["batch/segment_count"] == 0
    assert metrics["batch/trajectory_count"] == 1
    assert metrics["trajectory/reward_mean"] == 0.5
    assert sampling_path is weight_version is None
    assert not holder.optim_calls
    assert not holder.builtin_calls
    assert not holder.custom_calls
    assert not holder.saved_paths


def test_nested_trajectories_remain_complete_until_conversion_and_keep_update_boundaries(
    monkeypatch,
):
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums

    dumper = _DummyDataDumper()
    dumped = []
    monkeypatch.setattr(
        dumper, "should_dump", lambda data_type, step: data_type == "prepared_trajectories"
    )
    monkeypatch.setattr(
        dumper, "dump_prepared_trajectories", lambda **kwargs: dumped.append(kwargs)
    )
    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: dumper
    )
    trainer = _DummyRemoteApiTrainer(_make_config(batch_size=4, mini_batch_size=3))
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    nested_sequences = [
        {"tokens": [1, 2], "loss_mask": [0, 1], "old_log_probs": [0.0, -0.1]} for _ in range(3)
    ]
    trajectories = [
        Trajectory(
            segments=nested_sequences,
            reward=1.0,
            extra_fields={"rollout_id": "a", "advantage": 0.5},
        ),
        Trajectory(
            tokens=[3, 4],
            loss_mask=[0, 1],
            reward=0.0,
            extra_fields={"rollout_id": "b", "advantage": -0.5, "old_log_probs": [0.0, -0.2]},
        ),
    ]
    converted = []

    def convert(segments):
        converted.extend(segments)
        return convert_trajectories_to_datums(segments, drop_zero_advantage_datums=False)

    monkeypatch.setattr(trainer, "_convert_to_datums", convert)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert converted == trajectories
    assert all(converted[i] is trajectory for i, trajectory in enumerate(trajectories))
    assert dumped[0]["trajectories"] is trajectories
    assert [len(call["datums_data"]) for call in holder.builtin_calls] == [3, 1]
    assert len(holder.optim_calls) == 1
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["batch/segment_count"] == 4
    assert metrics["batch/trainable_segment_count"] == 4
    assert not any(key.startswith("datum/") for key in metrics)
    assert trajectories[1].segments is None
    assert trajectories[0].segments is nested_sequences
    assert all("reward" not in sequence for sequence in nested_sequences)
    assert metrics["algorithm/trajectory/reward_mean"] == 0.5


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_terminal_trajectory_without_trainable_segments_advances_round(monkeypatch, status):
    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    trainer._total_train_steps = 1
    trajectory = Trajectory(
        segments=[{"tokens": [1, 2], "loss_mask": [0, 1]}], reward=None, status=status
    )
    trainer._trajectory_pool = _SingleBatchTrajectoryPool([trajectory])
    trainer.set_module_name("trainer")
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    monkeypatch.setattr(trainer, "_on_training_complete", AsyncMock())

    run_coro(trainer._main_loop())

    assert trainer.get_train_step() == 1
    assert holder.optim_calls == holder.builtin_calls == holder.custom_calls == []
    assert holder.saved_paths == []
    assert len(trainer._data_dumper.rollout_round_calls) == 1
    dump_step, completed_step, records = trainer._data_dumper.rollout_round_calls[0]
    assert (dump_step, completed_step) == (0, 1)
    assert len(records) == 1
    assert records[0]["status"] == status
    assert records[0]["accepted_sequences"] == 0
    post = trainer._activity_tracker.posts[0]["data"]
    assert post["training/skipped_flag"] == 1
    assert post["batch/trajectory_count"] == 1
    assert post["batch/segment_count"] == 1
    assert post["batch/trainable_segment_count"] == 0
    assert trainer._weight_sync_controller.notifications[0]["weights_updated"] is False


def test_plain_trajectory_reaches_real_converter_without_wrapping(monkeypatch):
    from northrl.utils.finetune_service_utils import convert_trajectories_to_datums

    monkeypatch.setattr(
        remote_api_trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    tokens = [10, 11, 12]
    loss_mask = [0, 1, 1]
    trajectory = Trajectory(
        tokens=tokens,
        loss_mask=loss_mask,
        reward=1.0,
        extra_fields={"advantage": 2.0, "old_log_probs": [0.0, -0.1, -0.2]},
    )
    trajectories = [trajectory]
    converted = []

    def convert(parents):
        assert parents is trajectories
        assert parents[0] is trajectory
        converted.append(parents[0])
        return convert_trajectories_to_datums(parents)

    monkeypatch.setattr(trainer, "_convert_to_datums", convert)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert converted == [trajectory]
    assert trajectory.tokens is tokens
    assert trajectory.loss_mask is loss_mask
    assert trajectory.segments is None
    assert holder.builtin_calls[0]["datums_data"][0]["input_tokens"] == [10, 11]
    assert len(holder.optim_calls) == 1
    assert metrics["batch/trajectory_count"] == metrics["batch/segment_count"] == 1


@pytest.mark.parametrize("export_fails", [False, True])
def test_weight_export_timing_excludes_checkpoint_and_separates_failures(monkeypatch, export_fails):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    config = _make_config()
    config.training.save.every_n_steps = 1
    trainer = _DummyRemoteApiTrainer(config)
    trainer.set_train_step(10)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    now = [100.0]
    monkeypatch.setattr(
        remote_api_trainer_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    failure = RuntimeError("export failed")

    async def export(name):
        assert name == "northrl_actor_v000011"
        now[0] += 3.5
        if export_fails:
            raise failure
        return "/samplers/11"

    async def checkpoint(step):
        assert step == 11
        now[0] += 50.0
        return "/checkpoints/11"

    monkeypatch.setattr(holder, "save_weights_for_sampler", export)
    monkeypatch.setattr(trainer, "_save_full_checkpoint", checkpoint)
    if export_fails:
        with pytest.raises(RuntimeError) as caught:
            run_coro(trainer.train([_make_trajectory()]))
        assert caught.value is failure
        assert trainer._activity_tracker.posts == [
            {
                "backend": "wandb",
                "step": 11,
                "data": {
                    "timing_weights/weight_export_failed_seconds": 3.5,
                    "training/weight_export_success_flag": 0,
                },
            }
        ]
        assert now[0] == 103.5
    else:
        metrics, sampling_path, _ = run_coro(trainer.train([_make_trajectory()]))
        assert sampling_path == "/samplers/11"
        assert metrics["timing_weights/weight_export_seconds"] == 3.5
        assert metrics["timing_weights/checkpoint_seconds"] == 50.0
        assert metrics["timing_train/batch_training_seconds"] == 0.0
        assert metrics["training/weight_export_success_flag"] == 1
        assert "timing_weights/weight_export_failed_seconds" not in metrics
        assert metrics["training/checkpoint_path"] == "/checkpoints/11"
        assert now[0] == 153.5
        assert trainer._activity_tracker.posts == []


def test_training_timings_sum_fb_and_optimizer_separately_and_exclude_persistence(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    config = _make_config(batch_size=2, mini_batch_size=1)
    config.training.save.every_n_steps = 1
    trainer = _DummyRemoteApiTrainer(config)
    trainer._datum_count = 4
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    now = [100.0]
    monkeypatch.setattr(
        remote_api_trainer_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    fb_compute = iter((1000.0, 2000.0, 3000.0, 4000.0))
    fb_wall = iter((3.0, 4.0, 5.0, 6.0))
    opt_compute = iter((500.0, 600.0))
    original_convert = trainer._convert_to_datums

    async def prepare(trajectories, metrics):
        del metrics
        now[0] += 5.0
        return trajectories

    def convert(trajectories):
        now[0] += 2.0
        return original_convert(trajectories)

    async def fb(**kwargs):
        assert len(kwargs["datums_data"]) == 1
        now[0] += next(fb_wall)
        return {"metrics": {"loss": 1.0, "compute_ms": next(fb_compute)}}

    async def optimizer(**kwargs):
        del kwargs
        now[0] += 2.0
        return {"metrics": {"update_successful": True, "compute_ms": next(opt_compute)}}

    async def export(name):
        del name
        now[0] += 7.0
        return "/samplers/1"

    async def checkpoint(step):
        assert step == 1
        now[0] += 13.0
        return "/checkpoints/1"

    monkeypatch.setattr(trainer, "_prepare_trajectories", prepare)
    monkeypatch.setattr(trainer, "_convert_to_datums", convert)
    monkeypatch.setattr(holder, "forward_backward", fb)
    monkeypatch.setattr(holder, "optim_step", optimizer)
    monkeypatch.setattr(holder, "save_weights_for_sampler", export)
    monkeypatch.setattr(trainer, "_save_full_checkpoint", checkpoint)

    metrics, _, _ = run_coro(trainer.train([_make_trajectory()]))

    assert metrics["timing_train/batch_training_seconds"] == 29.0
    assert metrics["timing_train/forward_backward_wall_seconds"] == 18.0
    assert metrics["timing_train/optimizer_step_wall_seconds"] == 4.0
    assert metrics["timing_train/forward_backward_compute_seconds"] == 10.0
    assert metrics["timing_train/optimizer_step_compute_seconds"] == pytest.approx(1.1)
    assert metrics["timing_weights/weight_export_seconds"] == 7.0
    assert metrics["timing_weights/checkpoint_seconds"] == 13.0
    assert metrics["training/optimizer_step_count"] == 2
    assert metrics["training/forward_backward_call_count"] == 4
    assert now[0] == 149.0
    assert not any("_training_timing" in key for key in metrics)


def test_custom_loss_callback_compute_is_not_reported_as_native_fb_time(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    config = _make_config(
        loss={
            "mode": "custom",
            "params": {"name": "importance_sampling", "base_loss_fn": "importance_sampling"},
            "reduction": {"mode": "datum_mean"},
        }
    )
    trainer = _DummyRemoteApiTrainer(config)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    async def custom(**kwargs):
        del kwargs
        return {"metrics": {"loss": 1.0, "compute_ms": 123456.0}}

    monkeypatch.setattr(holder, "forward_backward_custom", custom)
    metrics, _, _ = run_coro(trainer.train([_make_trajectory()]))

    assert "timing_train/forward_backward_wall_seconds" in metrics
    assert "timing_train/optimizer_step_wall_seconds" in metrics
    assert "timing_train/forward_backward_compute_seconds" not in metrics


def test_mismatch_forward_wall_excludes_subsequent_statistics(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    now = [0.0]
    monkeypatch.setattr(
        remote_api_trainer_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )

    async def forward(tokens, sampling_masks=None):
        del sampling_masks
        now[0] += 2.0
        return [[-0.1] * (len(sequence) - 1) for sequence in tokens]

    def statistics(sequences, rollout, target, metrics):
        del sequences, rollout, target
        now[0] += 100.0
        metrics["mis/metrics_available"] = 1

    monkeypatch.setattr(trainer, "_read_mismatch_rollout_logprobs", lambda sequences: [[-0.1]])
    monkeypatch.setattr(holder, "compute_target_logprobs_batch", forward)
    monkeypatch.setattr(trainer, "_record_mismatch_metrics", statistics)
    metrics = {}

    run_coro(trainer._compute_inference_mismatch_metrics([_make_trajectory()], metrics))

    assert metrics["timing_train/mismatch_forward_seconds"] == 2.0
    assert metrics["mis/metrics_available"] == 1
    assert now[0] == 102.0
    assert "timing/mismatch_metrics" not in metrics
    assert "timing_train/mismatch_metrics_seconds" not in metrics


def test_export_metrics_failure_preserves_original_export_error(monkeypatch):
    DataDumper.reset_instances()
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    trainer = _DummyRemoteApiTrainer(_make_config())
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    failure = RuntimeError("export failed")
    monkeypatch.setattr(holder, "save_weights_for_sampler", AsyncMock(side_effect=failure))

    def fail_metrics(**kwargs):
        raise RuntimeError("logger unavailable")

    monkeypatch.setattr(trainer._activity_tracker, "experiment_logger_post", fail_metrics)
    with pytest.raises(RuntimeError) as caught:
        run_coro(trainer.train([_make_trajectory()]))
    assert caught.value is failure
