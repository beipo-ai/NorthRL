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
import logging
from contextlib import nullcontext

import pytest
from omegaconf import OmegaConf

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_cross_entropy_trainer as cross_entropy_trainer_module
from northrl.trainer import remote_api_trainer as remote_api_trainer_module
from northrl.trainer.remote_api_cross_entropy_trainer import RemoteApiCrossEntropyTrainer
from northrl.utils.async_utils import flush_sdk_logs, run_coro


class _DummyDataDumper:
    enabled = False

    def dump_rollout_round(self, step, completed_step, records):
        del step, completed_step, records

    def should_dump(self, data_type, step):
        del data_type, step
        return False


class _CapturingDataDumper(_DummyDataDumper):
    def __init__(self):
        self.datum_dumps = []

    def should_dump(self, data_type, step):
        del step
        return data_type == "datums"

    def dump_datums(self, **kwargs):
        self.datum_dumps.append(kwargs)


class _DummyActivityTracker:
    def experiment_logger_post(self, **kwargs):
        del kwargs


class _FailingActivityTracker:
    def track(self, module, work):
        del module, work
        return nullcontext()

    def experiment_logger_post(self, **kwargs):
        raise RuntimeError(f"metrics failed: {kwargs['step']}")

    def set_training_step(self, step):
        del step


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


class _NoSyncWeightSyncController:
    sync_mode = "no-sync"


class _CommitSpyDataLoader:
    def __init__(self, events, *, fail_release=False):
        self.events = events
        self.committed_batch_ids = []
        self.fail_release = fail_release

    def commit_batch(self, batch_id):
        self.events.append(("commit", batch_id))
        self.committed_batch_ids.append(batch_id)

    def release_next_batch(self):
        self.events.append(("release", None))
        if self.fail_release:
            raise RuntimeError("release failed")


class _FakeWeaverHolder:
    def __init__(
        self,
        events,
        *,
        fail_submit_at=None,
        fail_forward_at=None,
        fail_optim=False,
        update_successful=True,
        fb_metrics=None,
    ):
        self.events = events
        self.fail_submit_at = fail_submit_at
        self.fail_forward_at = fail_forward_at
        self.fail_optim = fail_optim
        self.update_successful = update_successful
        self.fb_metrics = fb_metrics
        self.forward_calls = []
        self.optim_calls = []

    async def submit_forward_backward(self, **kwargs):
        call_index = len(self.forward_calls) + 1
        self.events.append(("submit", call_index, len(kwargs["datums_data"])))
        self.forward_calls.append(kwargs)
        if self.fail_submit_at == call_index:
            raise RuntimeError(f"submit {call_index} failed")
        return call_index

    async def collect_forward_backward_result(self, handle, **kwargs):
        del kwargs
        call_index = handle
        self.events.append(("collect", call_index))
        if self.fail_forward_at == call_index:
            raise RuntimeError(f"forward {call_index} failed")
        metrics = (
            self.fb_metrics[call_index - 1]
            if self.fb_metrics is not None
            else {"loss": 1.0, "tokens": 6}
        )
        return {"loss": metrics.get("loss", 0.0), "metrics": metrics}

    async def optim_step(self, **kwargs):
        self.events.append(("optim", len(self.forward_calls)))
        self.optim_calls.append(kwargs)
        if self.fail_optim:
            raise RuntimeError("optimizer failed")
        return {
            "status": "success",
            "metrics": {"update_successful": int(self.update_successful)},
        }


def _make_config(*, max_sequence_length=32_768, batch_size=1):
    return OmegaConf.create(
        {
            "data": {
                "max_sequence_length": max_sequence_length,
                "train": {"batch_size": batch_size},
            },
            "training": {
                "total_steps": 4,
                "save": {"dir": "", "every_n_steps": 0},
            },
            "algorithm": {
                "type": "sft",
                "entropy_coeff": 0.0,
                "sampling_logprob_source": "recomputed",
                "optim": {
                    "lr": 2e-6,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 1e-2,
                    "eps": 1e-8,
                    "grad_clip_norm": 1.0,
                    "batch_size": 1,
                },
                "loss": {
                    "mode": "cross_entropy",
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
            },
            "rollout": {},
            "runtime": {"debug": {"enable_data_dump": False}},
        }
    )


def _make_trajectories(batch_id):
    return [
        Trajectory(
            tokens=[index, index + 1, index + 2],
            loss_mask=[0, 1, 1],
            reward=0.0,
            is_val=False,
            extra_fields={"sft_batch_id": batch_id},
        )
        for index in (10, 20, 30)
    ]


def _make_counted_trajectories(batch_id, counts):
    return [
        Trajectory(
            tokens=[index, index + 1],
            loss_mask=[0, 1],
            reward=0.0,
            is_val=False,
            extra_fields={
                "sft_batch_id": batch_id,
                "input_token_count": count,
            },
        )
        for index, count in enumerate(counts, start=10)
    ]


def _make_managed_trajectory(
    batch_id,
    *,
    content_visibility="public",
    loss_fn_inputs=None,
):
    if loss_fn_inputs is None:
        loss_fn_inputs = {}
    return Trajectory(
        tokens=None,
        loss_mask=None,
        reward=0.0,
        is_val=False,
        extra_fields={
            "model_tag": "actor",
            "group_id": "",
            "run_id": 0,
            "task_id": 0,
            "sft_batch_id": batch_id,
            "sample_ref": {
                "dataset": "hq-math",
                "version": "2026-08",
                "sample_idx": 5,
            },
            "datum_id": "northrl-managed-5",
            "input_token_count": 7,
            "content_visibility": content_visibility,
            "loss_fn_inputs": loss_fn_inputs,
        },
    )


def _make_trainer(monkeypatch, holder, dataloader, *, config=None):
    monkeypatch.setattr(
        remote_api_trainer_module,
        "get_data_dumper",
        lambda config, rank, key: _DummyDataDumper(),
    )
    trainer = RemoteApiCrossEntropyTrainer(config if config is not None else _make_config())
    aggregate_window_metrics = trainer._aggregate_optimizer_window_metrics

    def capture_window_metrics(forward_metrics, optimizer_metrics):
        result = aggregate_window_metrics(forward_metrics, optimizer_metrics)
        trainer._captured_optimizer_window_metrics = result
        return result

    monkeypatch.setattr(trainer, "_aggregate_optimizer_window_metrics", capture_window_metrics)
    trainer._activity_tracker = _DummyActivityTracker()  # pylint: disable=protected-access
    trainer.set_service_holder(holder)
    trainer.set_module_references(
        trajectory_pool=None,
        weight_sync_controller=_NoSyncWeightSyncController(),
        dataloader=dataloader,
    )
    return trainer


def test_sft_uses_one_optimizer_step_for_the_whole_token_budget_batch(monkeypatch, caplog):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    caplog.set_level(logging.INFO)
    metrics, sampling_path, weight_version = run_coro(trainer.train(_make_trajectories("batch-7")))
    flush_sdk_logs()
    assert (sampling_path, weight_version) == (None, None)
    assert (
        "[train] optimizer batch: train_step=1 batch=1/1 num_datums=3 forward_backward_calls=1"
        in caplog.messages
    )
    assert trainer._sampling_logprob_source == "inference"
    assert len(holder.forward_calls) == 1
    assert len(holder.forward_calls[0]["datums_data"]) == 3
    assert metrics["batch/segment_count"] == 3
    assert metrics["batch/trainable_segment_count"] == 3
    assert metrics["batch/trainable_trajectory_count"] == 3
    assert "batch/group_count" not in metrics
    assert not any(key.startswith("datum/") for key in metrics)
    assert holder.forward_calls[0]["loss_fn_config"] == trainer._build_loss_fn_config()
    assert "training/num_datums" not in metrics
    assert metrics["weaver/loss"] == 1.0
    assert metrics["training/window_shifted_tokens"] == 6
    window_metrics = trainer._captured_optimizer_window_metrics
    assert window_metrics["num_transport_batches"] == 1
    assert window_metrics["learning_rate"] == pytest.approx(2e-6)
    assert window_metrics["lr_schedule_step"] == 0
    assert len(holder.optim_calls) == 1
    assert events == [
        ("submit", 1, 3),
        ("collect", 1),
        ("optim", 1),
        ("commit", "batch-7"),
        ("release", None),
    ]
    assert dataloader.committed_batch_ids == ["batch-7"]


def test_sft_uses_resumed_completed_step_for_cosine_lr(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    config = _make_config()
    config.algorithm.optim.lr = 2.0e-5
    config.algorithm.optim.schedule = {
        "style": "cosine",
        "min_lr": 2.0e-6,
        "warmup_fraction": 0.03,
    }
    trainer = _make_trainer(monkeypatch, holder, dataloader, config=config)
    trainer.set_lr_schedule_total_steps(85)
    trainer.set_train_step(2)

    metrics, sampling_path, weight_version = run_coro(
        trainer.train(_make_trajectories("resumed-batch"))
    )
    assert (sampling_path, weight_version) == (None, None)

    assert holder.optim_calls[0]["learning_rate"] == pytest.approx(1.568627450980392e-5)
    window_metrics = trainer._captured_optimizer_window_metrics
    assert window_metrics["learning_rate"] == pytest.approx(holder.optim_calls[0]["learning_rate"])
    assert window_metrics["lr_schedule_step"] == 2
    assert window_metrics["lr_schedule_total_steps"] == 85


def test_sft_splits_optimizer_window_and_queues_all_forwards_before_collecting(
    monkeypatch,
):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(
        events,
        fb_metrics=[
            {"loss": 2.0, "loss:sum": 6.0, "loss_agg_denominator": 3.0},
            {"loss": 4.0, "loss:sum": 20.0, "loss_agg_denominator": 5.0},
            {"loss": 7.0, "loss:sum": 14.0, "loss_agg_denominator": 2.0},
        ],
    )
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trainer._loss_agg_mode = "token-mean"  # pylint: disable=protected-access
    trajectories = _make_counted_trajectories(
        "window-1",
        [32_767, 1, 2, 32_766, 48],
    )

    metrics, sampling_path, weight_version = run_coro(trainer.train(trajectories))
    assert (sampling_path, weight_version) == (None, None)

    assert [
        [datum["input_tokens"][0] for datum in call["datums_data"]] for call in holder.forward_calls
    ] == [
        [10, 11],
        [12, 13],
        [14],
    ]
    assert events == [
        ("submit", 1, 2),
        ("submit", 2, 2),
        ("submit", 3, 1),
        ("collect", 1),
        ("collect", 2),
        ("collect", 3),
        ("optim", 3),
        ("commit", "window-1"),
        ("release", None),
    ]
    for call in holder.forward_calls:
        assert call["loss_fn_config"] == trainer._build_loss_fn_config()
    assert len(holder.optim_calls) == 1
    assert metrics["training/window_shifted_tokens"] == 65_584
    window_metrics = trainer._captured_optimizer_window_metrics
    assert window_metrics["num_mini_batches"] == 3
    assert window_metrics["num_transport_batches"] == 3
    assert metrics["weaver/loss"] == 4.0
    assert window_metrics["loss_aggregation_exact"] == 1
    assert window_metrics["loss:sum"] == 40.0
    assert window_metrics["loss_agg_denominator"] == 10.0


@pytest.mark.parametrize("request_count", [1, 3])
def test_sft_times_complete_forward_window_separately_from_optimizer(monkeypatch, request_count):
    events = []
    holder = _FakeWeaverHolder(
        events,
        fb_metrics=[
            {"loss": 1.0, "compute_ms": 1000 * index} for index in range(1, request_count + 1)
        ],
    )
    trainer = _make_trainer(monkeypatch, holder, _CommitSpyDataLoader(events))
    original_collect = holder.collect_forward_backward_result
    original_optim = holder.optim_step
    last_request_finished = asyncio.Event()
    completions = []

    async def collect_out_of_order(handle, **kwargs):
        if handle != request_count:
            await last_request_finished.wait()
        result = await original_collect(handle, **kwargs)
        completions.append(handle)
        if handle == request_count:
            last_request_finished.set()
        return result

    async def optimizer_with_compute(**kwargs):
        assert len(completions) == request_count
        result = await original_optim(**kwargs)
        result["metrics"]["compute_ms"] = 50
        return result

    holder.collect_forward_backward_result = collect_out_of_order
    holder.optim_step = optimizer_with_compute
    clock_values = iter([10.0, 15.0, 100.0, 102.0])
    counter_events = []

    def perf_counter():
        counter_events.append(list(events))
        return next(clock_values)

    monkeypatch.setattr(cross_entropy_trainer_module.time, "perf_counter", perf_counter)
    mini_batches = [[{}] for _ in range(request_count)]
    service_metrics = run_coro(trainer._run_optimizer_batch([], mini_batches, None))

    assert completions[0] == request_count
    assert service_metrics["_training_timing"] == {
        "timing_train/forward_backward_wall_seconds": 5.0,
        "timing_train/optimizer_step_wall_seconds": 2.0,
        "timing_train/forward_backward_compute_seconds": sum(range(1, request_count + 1)),
        "timing_train/optimizer_step_compute_seconds": 0.05,
    }
    assert service_metrics["compute_ms"] == 50
    assert counter_events[0] == []
    assert len(counter_events[1]) == 2 * request_count
    assert counter_events[2] == counter_events[1]
    assert counter_events[3][-1] == ("optim", request_count)


def test_sft_partial_compute_preserves_complete_wall_time(monkeypatch):
    events = []
    holder = _FakeWeaverHolder(events, fb_metrics=[{"compute_ms": 1000}, {"loss": 1.0}])
    trainer = _make_trainer(monkeypatch, holder, _CommitSpyDataLoader(events))
    clock_values = iter([10.0, 15.0, 20.0, 22.0])
    monkeypatch.setattr(
        cross_entropy_trainer_module.time, "perf_counter", lambda: next(clock_values)
    )

    service_metrics = run_coro(trainer._run_optimizer_batch([], [[{}], [{}]], None))

    assert service_metrics["_training_timing"] == {
        "timing_train/forward_backward_wall_seconds": 5.0,
        "timing_train/optimizer_step_wall_seconds": 2.0,
    }


def test_sft_step_publishes_stage_times_without_private_service_metrics(monkeypatch):
    events = []
    holder = _FakeWeaverHolder(events, fb_metrics=[{"compute_ms": 1000}, {"compute_ms": 2000}])
    trainer = _make_trainer(monkeypatch, holder, _CommitSpyDataLoader(events))
    original_optim = holder.optim_step

    async def optimizer_with_compute(**kwargs):
        result = await original_optim(**kwargs)
        result["metrics"]["compute_ms"] = 50
        return result

    holder.optim_step = optimizer_with_compute
    metrics, _, _ = run_coro(
        trainer.train(_make_counted_trajectories("timed-window", [32_000, 1_000]))
    )

    assert metrics["timing_train/forward_backward_compute_seconds"] == 3.0
    assert metrics["timing_train/optimizer_step_compute_seconds"] == 0.05
    assert metrics["timing_train/forward_backward_wall_seconds"] >= 0.0
    assert metrics["timing_train/optimizer_step_wall_seconds"] >= 0.0
    assert not any("_training_timing" in key for key in metrics)


def test_sft_allows_whole_datum_at_transport_position_limit(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    metrics, sampling_path, weight_version = run_coro(
        trainer.train(_make_counted_trajectories("window-limit", [32_768]))
    )
    assert (sampling_path, weight_version) == (None, None)

    assert len(holder.forward_calls) == 1
    assert len(holder.forward_calls[0]["datums_data"]) == 1
    assert metrics["training/window_shifted_tokens"] == 32_768
    assert events == [
        ("submit", 1, 1),
        ("collect", 1),
        ("optim", 1),
        ("commit", "window-limit"),
        ("release", None),
    ]


def test_sft_sends_whole_datum_over_transport_batch_target_without_splitting(
    monkeypatch,
):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    metrics, sampling_path, weight_version = run_coro(
        trainer.train(_make_counted_trajectories("window-over-limit", [39_624, 1_000]))
    )
    assert (sampling_path, weight_version) == (None, None)

    assert [len(call["datums_data"]) for call in holder.forward_calls] == [1, 1]
    assert len(holder.optim_calls) == 1
    assert metrics["training/window_shifted_tokens"] == 40_624
    assert trainer._captured_optimizer_window_metrics["num_transport_batches"] == 2
    assert dataloader.committed_batch_ids == ["window-over-limit"]


def test_sft_reuses_loader_token_budget_for_one_long_context_transport(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(
        monkeypatch,
        holder,
        dataloader,
        config=_make_config(max_sequence_length=262_144, batch_size=1),
    )

    counts = [
        240,
        39_632,
        16_528,
        31_264,
        29_456,
        20_512,
        32_304,
        32_000,
        27_000,
        32_000,
    ]
    metrics, sampling_path, weight_version = run_coro(
        trainer.train(_make_counted_trajectories("window-256k", counts))
    )
    assert (sampling_path, weight_version) == (None, None)

    assert sum(counts) <= 262_144
    assert [len(call["datums_data"]) for call in holder.forward_calls] == [len(counts)]
    assert len(holder.optim_calls) == 1
    assert metrics["training/window_shifted_tokens"] == sum(counts)
    assert trainer._captured_optimizer_window_metrics["num_transport_batches"] == 1
    assert dataloader.committed_batch_ids == ["window-256k"]


def test_sft_multi_transport_does_not_report_inexact_mean_as_window_loss(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(
        events,
        fb_metrics=[{"loss": 2.0}, {"loss": 8.0}],
    )
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    metrics, sampling_path, weight_version = run_coro(
        trainer.train(_make_counted_trajectories("window-loss", [32_000, 1_000]))
    )
    assert (sampling_path, weight_version) == (None, None)

    assert "weaver/loss" not in metrics
    assert trainer._captured_optimizer_window_metrics["transport_batch_loss_mean"] == 5.0
    assert trainer._captured_optimizer_window_metrics["loss_aggregation_exact"] == 0


def test_sft_forward_failure_stops_window_before_optim_and_commit(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events, fail_forward_at=2)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    with pytest.raises(RuntimeError, match="forward 2 failed"):
        run_coro(trainer.train(_make_counted_trajectories("window-fail", [32_000, 1_000, 32_000])))

    assert events == [
        ("submit", 1, 1),
        ("submit", 2, 1),
        ("submit", 3, 1),
        ("collect", 1),
        ("collect", 2),
        ("collect", 3),
    ]
    assert holder.optim_calls == []
    assert dataloader.committed_batch_ids == []


def test_sft_submission_failure_drains_accepted_handles_before_propagating(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events, fail_submit_at=2)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    with pytest.raises(RuntimeError, match="submit 2 failed"):
        run_coro(
            trainer.train(_make_counted_trajectories("window-submit-fail", [32_000, 1_000, 32_000]))
        )

    assert events == [
        ("submit", 1, 1),
        ("submit", 2, 1),
        ("collect", 1),
    ]
    assert holder.optim_calls == []
    assert dataloader.committed_batch_ids == []


def test_sft_converts_mixed_inputs_without_materializing_managed_tokens(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trajectories = [
        _make_trajectories("batch-mixed")[0],
        _make_managed_trajectory(
            "batch-mixed",
            loss_fn_inputs={"auxiliary_scale": [0.25]},
        ),
    ]

    metrics, sampling_path, weight_version = run_coro(trainer.train(trajectories))
    assert (sampling_path, weight_version) == (None, None)

    sent = holder.forward_calls[0]["datums_data"]
    assert sent[0]["input_tokens"] == [10, 11]
    assert sent[1] == {
        "kind": "sample_ref",
        "datum_id": "northrl-managed-5",
        "dataset": "hq-math",
        "version": "2026-08",
        "sample_idx": 5,
        "content_visibility": "public",
        "loss_fn_inputs": {"auxiliary_scale": [0.25]},
    }
    assert "input_tokens" not in sent[1]
    assert holder.forward_calls[0]["loss_fn_config"] == trainer._build_loss_fn_config()
    assert metrics["batch/trainable_segment_count"] == 2
    assert not any(key.startswith("datum/") for key in metrics)
    assert events[-2:] == [("commit", "batch-mixed"), ("release", None)]


def test_mini_batch_split_preserves_managed_datums_without_padding(monkeypatch):
    events = []
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder(events),
        _CommitSpyDataLoader(events),
    )
    trainer._mini_batch_size = 2  # pylint: disable=protected-access
    datums = [
        {
            "kind": "sample_ref",
            "datum_id": f"managed-{sample_idx}",
            "dataset": "hq-math",
            "version": "2026-08",
            "sample_idx": sample_idx,
            "content_visibility": "protected",
            "loss_fn_inputs": {},
        }
        for sample_idx in range(3)
    ]

    split = getattr(remote_api_trainer_module.RemoteApiTrainer, "_split_into_mini_batches")
    first = split(trainer, datums)
    second = split(trainer, datums)

    assert [len(batch) for batch in first] == [2, 1]
    assert first == second
    assert [datum for batch in first for datum in batch] == datums
    assert all(datum["content_visibility"] == "protected" for batch in first for datum in batch)
    assert len({datum["datum_id"] for batch in first for datum in batch}) == 3


def test_managed_mixed_batch_uses_configured_loss_aggregation(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    dumper = _CapturingDataDumper()
    trainer._data_dumper = dumper  # pylint: disable=protected-access
    trajectories = [
        _make_trajectories("batch-protected")[0],
        _make_managed_trajectory(
            "batch-protected",
            content_visibility="protected",
            loss_fn_inputs={},
        ),
    ]

    run_coro(trainer.train(trajectories))

    call = holder.forward_calls[0]
    assert call["loss_fn"] == "cross_entropy"
    assert call["loss_fn_config"] == trainer._build_loss_fn_config()
    assert call["datums_data"][1]["content_visibility"] == "protected"
    assert call["datums_data"][1]["loss_fn_inputs"] == {}
    assert dumper.datum_dumps[0]["extra_info"]["loss_fn_config"] == trainer._build_loss_fn_config()


def test_public_only_managed_batch_uses_configured_loss_aggregation(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    run_coro(trainer.train([_make_managed_trajectory("batch-public")]))

    assert holder.forward_calls[0]["loss_fn_config"] == trainer._build_loss_fn_config()


@pytest.mark.parametrize("content_visibility", [None, "unexpected"])
def test_managed_visibility_does_not_change_loss_aggregation(
    monkeypatch,
    content_visibility,
):
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder([]),
        _CommitSpyDataLoader([]),
    )
    datum = {
        "kind": "sample_ref",
        "content_visibility": content_visibility,
        "loss_fn_inputs": {},
    }

    assert trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
        [datum],
        {"entropy_coeff": 0.0},
    ) == {"entropy_coeff": 0.0}


@pytest.mark.parametrize(
    "reduction,expected",
    [(None, "sample-mean"), ("token_mean", "token-mean"), ("datum_mean", "sample-mean")],
)
def test_sft_forwards_configured_loss_aggregation_to_weaver(monkeypatch, reduction, expected):
    trainer_config = _make_config()
    if reduction is not None:
        trainer_config.algorithm.loss = {
            "mode": "cross_entropy",
            "reduction": {"mode": reduction},
        }
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder([]),
        _CommitSpyDataLoader([]),
        config=trainer_config,
    )

    config = trainer._build_loss_fn_config()  # pylint: disable=protected-access

    # SFT datums carry raw masks, so Weaver must aggregate with the configured mode.
    assert config["loss_agg_mode"] == expected


def test_managed_microbatch_keeps_attached_loss_inputs_and_configuration(monkeypatch):
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder([]),
        _CommitSpyDataLoader([]),
    )
    datum = {
        "kind": "sample_ref",
        "content_visibility": "public",
        "loss_fn_inputs": {"auxiliary_scale": [0.25]},
    }
    configured = {"loss_agg_mode": "token-mean"}

    assert (
        trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
            [datum],
            configured,
        )
        is configured
    )


def test_managed_microbatch_rejects_non_builtin_ce(monkeypatch):
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder([]),
        _CommitSpyDataLoader([]),
    )
    datum = {
        "kind": "sample_ref",
        "content_visibility": "protected",
        "loss_fn_inputs": {},
    }
    trainer._loss_fn = "importance_sampling"  # pylint: disable=protected-access

    with pytest.raises(ValueError, match="built-in cross_entropy"):
        trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
            [datum],
            None,
        )

    trainer._loss_fn = "cross_entropy"  # pylint: disable=protected-access
    trainer._custom_loss_spec = object()  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="built-in cross_entropy"):
        trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
            [datum],
            None,
        )


def test_managed_batches_use_the_same_loss_config_for_all_visibilities(monkeypatch):
    trainer = _make_trainer(
        monkeypatch,
        _FakeWeaverHolder([]),
        _CommitSpyDataLoader([]),
    )
    configured = trainer._build_loss_fn_config()  # pylint: disable=protected-access
    protected_batch = [
        {
            "kind": "sample_ref",
            "content_visibility": "protected",
            "loss_fn_inputs": {},
        }
    ]
    public_and_local_batch = [
        {
            "kind": "sample_ref",
            "content_visibility": "public",
            "loss_fn_inputs": {},
        },
        {
            "input_tokens": [1],
            "loss_fn_inputs": {"target_tokens": [2], "loss_mask": [1]},
        },
    ]

    assert (
        trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
            protected_batch,
            configured,
        )
        is configured
    )
    assert (
        trainer._loss_fn_config_for_microbatch(  # pylint: disable=protected-access
            public_and_local_batch,
            configured,
        )
        is configured
    )


def test_sft_does_not_commit_cursor_when_optimizer_collection_fails(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events, fail_optim=True)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    with pytest.raises(RuntimeError, match="optimizer failed"):
        run_coro(trainer.train(_make_trajectories("batch-8")))

    assert events == [
        ("submit", 1, 3),
        ("collect", 1),
        ("optim", 1),
    ]
    assert dataloader.committed_batch_ids == []


def test_sft_does_not_commit_or_complete_step_when_optimizer_update_is_unsuccessful(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events, update_successful=False)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trainer._activity_tracker = _FailingActivityTracker()  # pylint: disable=protected-access
    trainer._trajectory_pool = _SingleBatchTrajectoryPool(  # pylint: disable=protected-access
        _make_trajectories("batch-overflow")
    )

    with pytest.raises(RuntimeError, match="optimizer update was not successful"):
        run_coro(trainer._main_loop())  # pylint: disable=protected-access

    assert events == [
        ("submit", 1, 3),
        ("collect", 1),
        ("optim", 1),
    ]
    assert dataloader.committed_batch_ids == []
    assert trainer.get_train_step() == 0


def test_sft_release_failure_propagates_after_commit(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events, fail_release=True)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)

    with pytest.raises(RuntimeError, match="release failed"):
        run_coro(trainer.train(_make_trajectories("batch-10")))

    assert events == [
        ("submit", 1, 3),
        ("collect", 1),
        ("optim", 1),
        ("commit", "batch-10"),
        ("release", None),
    ]


def test_sft_commits_cursor_before_observability_failure(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trainer._activity_tracker = _FailingActivityTracker()  # pylint: disable=protected-access
    trainer._trajectory_pool = _SingleBatchTrajectoryPool(  # pylint: disable=protected-access
        _make_trajectories("batch-9")
    )

    with pytest.raises(RuntimeError, match="metrics failed"):
        run_coro(trainer._main_loop())  # pylint: disable=protected-access

    assert events == [
        ("submit", 1, 3),
        ("collect", 1),
        ("optim", 1),
        ("commit", "batch-9"),
        ("release", None),
    ]
    assert dataloader.committed_batch_ids == ["batch-9"]


def test_sft_chunks_forward_backward_but_commits_only_after_single_optimizer_step(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trainer._sft_transport_position_budget = 4

    metrics, _, _ = run_coro(trainer.train(_make_trajectories("batch-chunked")))

    assert events == [
        ("submit", 1, 2),
        ("submit", 2, 1),
        ("collect", 1),
        ("collect", 2),
        ("optim", 2),
        ("commit", "batch-chunked"),
        ("release", None),
    ]
    assert all(
        call["loss_fn_config"] == trainer._build_loss_fn_config() for call in holder.forward_calls
    )
    assert "training/num_batches" not in metrics
    assert metrics["training/optimizer_step_count"] == 1
    assert metrics["training/forward_backward_call_count"] == 2


def test_sft_chunk_failure_leaves_cursor_uncommitted(monkeypatch):
    events = []
    dataloader = _CommitSpyDataLoader(events)
    holder = _FakeWeaverHolder(events, fail_forward_at=2)
    trainer = _make_trainer(monkeypatch, holder, dataloader)
    trainer._sft_transport_position_budget = 4
    with pytest.raises(RuntimeError, match="forward 2 failed"):
        run_coro(trainer.train(_make_trajectories("batch-failed")))

    assert events == [
        ("submit", 1, 2),
        ("submit", 2, 1),
        ("collect", 1),
        ("collect", 2),
    ]
    assert dataloader.committed_batch_ids == []
