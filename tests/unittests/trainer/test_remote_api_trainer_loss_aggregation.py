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

from copy import deepcopy

import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import KLConfig
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _DummyRemoteApiTrainer,
    _FakeWeaverHolder,
    _FakeWeightSyncController,
    _make_config,
    _make_trajectory,
)
from tests.unittests.trajectory_pool.test_complete_groups import _instance as _complete_group_pool


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    return _DummyRemoteApiTrainer(_make_config(batch_size=2))


def _datum(advantages, mask):
    return {
        "input_tokens": [1] * len(advantages),
        "loss_fn_inputs": {"advantages": advantages, "loss_mask": mask},
    }


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean"])
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
@pytest.mark.parametrize("reuse_counts", [False, True])
def test_sum_preserves_loss_and_gradient(trainer, mode, loss_fn, reuse_counts):
    trainer._algorithm_config.loss.reduction.mode, trainer._loss_fn = mode, loss_fn
    advantages = torch.tensor(
        [[2.0, -3.0, 0.0, 8.0], [-2.0, 4.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    datums = [_datum(a, m) for a, m in zip(advantages.tolist(), mask.tolist(), strict=True)]
    original = deepcopy(datums)
    normalized = trainer._normalize_advantages_for_sum(
        datums, token_counts=[3, 2, 0] if reuse_counts else None
    )
    scaled_advantages = torch.tensor(
        [d["loss_fn_inputs"]["advantages"] for d in normalized], dtype=torch.float64
    )
    log_ratios = (
        torch.tensor(
            [[0.6, 4.0, 1.1, 2.2], [1.05, 1.4, 1.0, 1.0], [0.9, 1.0, 1.0, 1.0]], dtype=torch.float64
        )
        .log()
        .requires_grad_()
    )

    def objective(adv):
        ratios = log_ratios.exp()
        losses = -adv * ratios
        if loss_fn == "ppo_clip":
            losses = torch.maximum(losses, -adv * ratios.clamp(0.8, 1.2))
            losses = torch.where(adv < 0, torch.minimum(losses, -adv * 3.0), losses)
        return losses * mask

    # Include a valid zero advantage, a fully masked sample and active dual clipping.
    raw_loss = objective(advantages)
    counts = mask.sum(dim=1)
    expected = (
        raw_loss.sum() / counts.sum()
        if mode == "token_mean"
        else (raw_loss.sum(dim=1) / counts.clamp_min(1)).sum() / (counts > 0).sum()
    )
    actual = objective(scaled_advantages).sum()
    expected_gradient = torch.autograd.grad(expected, log_ratios, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, log_ratios)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert expected_gradient.count_nonzero() > 0
    assert datums == original


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean"])
def test_train_normalizes_after_padding_without_mutation(trainer, monkeypatch, mode):
    trainer._algorithm_config.loss.reduction.mode = mode
    datums = [_datum([2.0], [1]), _datum([4.0, 0.0], [1, 1]), _datum([-6.0] * 3, [1] * 3)]
    original = deepcopy(datums)
    monkeypatch.setattr(trainer, "_convert_to_datums", lambda trajectories: datums)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)
    run_coro(trainer.train([_make_trajectory() for _ in datums]))

    assert len(holder.builtin_calls) == 2
    first, last = [call["datums_data"] for call in holder.builtin_calls]
    expected = [[2 / 3], [4 / 3, 0.0]] if mode == "token_mean" else [[1.0], [1.0, 0.0]]
    for datum, values in zip(first, expected, strict=True):
        assert datum["loss_fn_inputs"]["advantages"] == pytest.approx(values)
    for datum in last:
        assert datum["loss_fn_inputs"]["advantages"] == [-1.0] * 3
    assert last[0]["loss_fn_inputs"]["advantages"] is not last[1]["loss_fn_inputs"]["advantages"]
    assert all(call["loss_fn_config"]["loss_agg_mode"] == "sum" for call in holder.builtin_calls)
    assert datums == original


def test_train_reuses_counts_after_shuffle_and_refreshes_them_next_step(trainer, monkeypatch):
    trainer._algorithm_config.loss.reduction.mode = "token_mean"
    trainer._mini_batch_size = 1
    datums = [_datum([2.0] * 3, [1, 0, 0]), _datum([4.0] * 3, [1, 1, 0])]
    datums.append(_datum([6.0] * 3, [1, 1, 1]))
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda items: items.reverse())
    monkeypatch.setattr(trainer, "_convert_to_datums", lambda trajectories: datums)
    holder = _FakeWeaverHolder()
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train([_make_trajectory() for _ in datums]))
    submitted = [call["datums_data"][0] for call in holder.builtin_calls]
    # The reversed first optimizer batch has 3+2 targets. The last datum is
    # deep-copied for optimizer padding and contributes its single target twice.
    assert [d["loss_fn_inputs"]["advantages"][0] for d in submitted] == pytest.approx(
        [6 / 5, 4 / 5, 2 / 2, 2 / 2]
    )
    assert metrics["batch/trainable_response_tokens_total"] == 6
    assert [d["loss_fn_inputs"]["advantages"][0] for d in datums] == [2, 4, 6]

    holder.builtin_calls.clear()
    datums[0]["loss_fn_inputs"]["loss_mask"][:] = [1, 1, 1]
    metrics, _, _ = run_coro(trainer.train([_make_trajectory() for _ in datums]))
    submitted = [call["datums_data"][0] for call in holder.builtin_calls]
    assert [d["loss_fn_inputs"]["advantages"][0] for d in submitted] == pytest.approx(
        [6 / 5, 4 / 5, 2 / 6, 2 / 6]
    )
    assert metrics["batch/trainable_response_tokens_total"] == 8


class _AccumulatingHolder(_FakeWeaverHolder):
    """Check gradients across requests using a real scalar autograd graph."""

    def __init__(self, fail_fb=None):
        super().__init__()
        self.parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        self.events = []
        self.gradients = []
        self.fail_fb = fail_fb

    async def forward_backward(self, **kwargs):
        self.builtin_calls.append(kwargs)
        self.events.append(("forward_backward", len(kwargs["datums_data"])))
        if len(self.builtin_calls) == self.fail_fb:
            raise RuntimeError("forward_backward failed")
        assert kwargs["loss_fn_config"]["loss_agg_mode"] == "sum"
        loss = torch.zeros((), dtype=torch.float64)
        tokens = 0
        for datum in kwargs["datums_data"]:
            inputs = datum["loss_fn_inputs"]
            mask = torch.tensor(inputs["loss_mask"], dtype=torch.float64)
            advantages = torch.tensor(inputs["advantages"], dtype=torch.float64)
            loss = loss - (self.parameter.exp() * advantages * mask).sum()
            tokens += int(mask.sum())
        loss.backward()
        return {"loss": loss.item(), "metrics": {"loss": loss.item(), "tokens": tokens}}

    async def optim_step(self, **kwargs):
        self.events.append(("optim_step", None))
        self.gradients.append(self.parameter.grad.item())
        with torch.no_grad():
            self.parameter -= 0.1 * self.parameter.grad
            self.parameter.grad.zero_()
        return await super().optim_step(**kwargs)


def _grpo_trainer(
    monkeypatch,
    mode="token_mean",
    mini_batch_size=16,
    sync_mode="sync",
    *,
    sampling_logprob_source="inference",
    mis=None,
):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    config = _make_config(batch_size=None, mini_batch_size=mini_batch_size)
    config.algorithm.loss.reduction.mode = mode
    config.algorithm.sampling_logprob_source = sampling_logprob_source
    if mis is not None:
        config.algorithm.mis = mis.model_dump()
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    config.training.weight_sync = {"mode": sync_mode}
    trainer = RemoteApiGrpoTrainer(config)
    trainer._weight_sync_controller = _FakeWeightSyncController(sync_mode=sync_mode)
    return trainer


def _grpo_round():
    """Two complete groups, four rollouts, and 100 unequal-length sequences."""
    trajectories = []
    sequence_id = 1
    for rollout_index, sequence_count in enumerate((1, 34, 33, 32)):
        sequences = []
        for _ in range(sequence_count):
            response_length = 1 + sequence_id % 3
            sequences.append(
                {
                    "tokens": [sequence_id] + [101] * response_length,
                    "loss_mask": [0] + [1] * response_length,
                    "old_log_probs": [0.0] + [-0.1] * response_length,
                }
            )
            sequence_id += 1
        identity = {
            "group_id": str(rollout_index // 2),
            "run_id": rollout_index % 2,
            "rollout_id": str(rollout_index),
            "expected_rollouts": 2,
        }
        if sequence_count == 1:
            sequence = sequences[0]
            trajectory = Trajectory(
                tokens=sequence["tokens"],
                loss_mask=sequence["loss_mask"],
                reward=float(rollout_index % 2),
                extra_fields={**identity, "old_log_probs": sequence["old_log_probs"]},
            )
        else:
            trajectory = Trajectory(
                segments=sequences,
                reward=float(rollout_index % 2),
                extra_fields=identity,
            )
        trajectories.append(trajectory)
    return trajectories


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean"])
def test_full_grpo_round_chunking_preserves_sequences_loss_gradient_and_update(monkeypatch, mode):
    def unexpected_shuffle(_):
        pytest.fail("A full rollout batch must not shuffle or resample its Datums")

    monkeypatch.setattr(trainer_module.random, "shuffle", unexpected_shuffle)
    outcomes = []
    for mini_batch_size in (None, 16):
        trainer = _grpo_trainer(monkeypatch, mode, mini_batch_size)
        holder = _AccumulatingHolder()
        trainer.set_service_holder(holder)

        metrics, _, _ = run_coro(trainer.train(_grpo_round()))

        assert metrics["batch/trajectory_count"] == 4
        assert metrics["batch/segment_count"] == metrics["batch/trainable_segment_count"] == 100
        assert metrics["batch/trainable_segment_count"] == 100
        assert metrics["training/optimizer_step_count"] == 1
        outcomes.append((holder, metrics))

    baseline, split = outcomes
    assert split[0].events == [("forward_backward", 16)] * 6 + [
        ("forward_backward", 4),
        ("optim_step", None),
    ]
    assert split[1]["training/forward_backward_call_count"] == 7
    split_datums = [datum for call in split[0].builtin_calls for datum in call["datums_data"]]
    assert [datum["input_tokens"][0] for datum in split_datums] == list(range(1, 101))
    assert split_datums == baseline[0].builtin_calls[0]["datums_data"]
    assert len(split[0].optim_calls) == len(baseline[0].optim_calls) == 1
    assert abs(baseline[0].gradients[0]) > 0.1
    assert split[0].gradients == pytest.approx(baseline[0].gradients, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(split[0].parameter, baseline[0].parameter, rtol=1e-12, atol=1e-12)
    assert split[1]["weaver/loss"] == pytest.approx(baseline[1]["weaver/loss"])


@pytest.mark.parametrize("sync_mode", ["sync", "batch-async", "fully-async"])
def test_complete_pool_batch_trains_once_at_each_modes_boundary(monkeypatch, sync_mode):
    pool, state = _complete_group_pool(groups_per_batch=2)
    trajectories = _grpo_round()
    next_batch = deepcopy(trajectories)
    for trajectory in next_batch:
        trajectory["group_id"] = "next-" + trajectory["group_id"]
        trajectory["rollout_id"] = "next-" + trajectory["rollout_id"]
    # A pending group cannot block a full B-group delivery in any training mode.
    assert pool.put_trajectory(next_batch[0]) == "success"

    for trajectory in trajectories[:-1]:
        assert pool.put_trajectory(trajectory) == "success"
        assert pool.get_trajectories() is None
    assert pool.put_trajectory(trajectories[-1]) == "success"

    ready = pool.get_trajectories()
    assert ready is not None
    assert all(actual is expected for actual, expected in zip(ready, trajectories, strict=True))
    assert state.has_items and not state.idle
    assert not pool.is_empty()
    assert pool.get_trajectories() is None

    trainer = _grpo_trainer(monkeypatch, sync_mode=sync_mode)
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(ready))

    assert metrics["batch/trajectory_count"] == 4
    assert metrics["batch/segment_count"] == metrics["batch/trainable_segment_count"] == 100
    assert metrics["training/optimizer_step_count"] == 1
    assert metrics["training/forward_backward_call_count"] == 7
    assert holder.events == [("forward_backward", 16)] * 6 + [
        ("forward_backward", 4),
        ("optim_step", None),
    ]
    assert [
        datum["input_tokens"][0] for call in holder.builtin_calls for datum in call["datums_data"]
    ] == list(range(1, 101))

    # Completing the remaining groups returns the original pending trajectory next time.
    for trajectory in next_batch[1:]:
        assert pool.put_trajectory(trajectory) == "success"
    assert pool.get_trajectories() == next_batch
    assert pool.is_empty()


@pytest.mark.parametrize("failed", [False, True])
def test_full_round_without_training_sequences_skips_update_and_export(monkeypatch, failed):
    trainer = _grpo_trainer(monkeypatch)
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)
    trajectories = [
        Trajectory(
            segments=[],
            reward=None if failed and run_id == 0 else float(run_id),
            status="failed" if failed and run_id == 0 else "success",
            extra_fields={
                "group_id": "empty",
                "run_id": run_id,
                "rollout_id": str(run_id),
                "expected_rollouts": 2,
            },
        )
        for run_id in range(2)
    ]

    metrics, sampling_path, weight_version = run_coro(trainer.train(trajectories))

    assert metrics["batch/trajectory_count"] == 2
    assert metrics["batch/segment_count"] == metrics["training/optimizer_step_count"] == 0
    assert metrics["training/skipped_flag"] == 1
    assert (sampling_path, weight_version) == (None, None)
    assert holder.events == holder.saved_paths == []


def test_each_batch_finishes_all_fb_requests_before_optim_without_inner_padding(trainer):
    trainer._batch_size, trainer._mini_batch_size = 4, 3
    trainer._datum_count = 8
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)

    run_coro(trainer.train([_make_trajectory()]))

    expected = [("forward_backward", 3), ("forward_backward", 1), ("optim_step", None)]
    assert holder.events == expected * 2
    assert [
        datum["input_tokens"][0] for call in holder.builtin_calls for datum in call["datums_data"]
    ] == list(range(1, 9))


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean"])
@pytest.mark.parametrize("mini_batch_size", [1, 2])
def test_fb_chunking_preserves_batch_loss_gradient_and_update(
    trainer, monkeypatch, mode, mini_batch_size
):
    trainer._batch_size, trainer._algorithm_config.loss.reduction.mode = 0, mode
    datums = [_datum([2.0], [1]), _datum([4.0, -1.0], [1, 1]), _datum([-6.0] * 3, [1, 0, 1])]
    original = deepcopy(datums)
    monkeypatch.setattr(trainer, "_convert_to_datums", lambda trajectories: datums)
    outcomes = []
    for chunk_size in (None, mini_batch_size):
        trainer._mini_batch_size = chunk_size
        holder = _AccumulatingHolder()
        trainer.set_service_holder(holder)
        metrics, _, _ = run_coro(trainer.train([_make_trajectory()]))
        outcomes.append((holder, metrics))

    baseline, split = outcomes
    assert split[0].gradients == pytest.approx(baseline[0].gradients, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(split[0].parameter, baseline[0].parameter, rtol=1e-12, atol=1e-12)
    assert split[1]["weaver/loss"] == pytest.approx(baseline[1]["weaver/loss"])
    assert len(split[0].optim_calls) == len(baseline[0].optim_calls) == 1
    assert datums == original


def test_failed_fb_request_prevents_optimizer_step(trainer):
    trainer._batch_size, trainer._mini_batch_size = 4, 2
    trainer._datum_count = 8
    holder = _AccumulatingHolder(fail_fb=2)
    trainer.set_service_holder(holder)

    with pytest.raises(RuntimeError, match="forward_backward failed"):
        run_coro(trainer.train([_make_trajectory()]))

    assert holder.events == [("forward_backward", 2), ("forward_backward", 2)]
    assert holder.optim_calls == []
    assert holder.saved_paths == []
