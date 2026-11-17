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

"""Prompt-mean objective and gradient checks through complete GRPO rounds."""

import math
from collections import defaultdict
from copy import deepcopy

import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import MISConfig
from tests.unittests.trainer.test_remote_api_trainer_loss_aggregation import (
    _AccumulatingHolder,
    _grpo_round,
    _grpo_trainer,
)


def _sequence(tokens, mask):
    return {
        "tokens": tokens,
        "loss_mask": mask,
        "old_log_probs": [0.0] + [-token / 1000 for token in tokens[1:]],
        "sampling_mask": [[token, token + 1000] for token, keep in zip(tokens, mask) if keep],
    }


def _trajectory(group, run, response, *, split=False, reward=None, expected_rollouts=2):
    prefix = [10 + run]
    if split:
        segments = [
            _sequence(prefix + response[: index + 1], [0] * (index + 1) + [1])
            for index in range(len(response))
        ]
    else:
        segments = [_sequence(prefix + response, [0] + [1] * len(response))] if response else []
    return Trajectory(
        segments=segments,
        reward=float(run) if reward is None else reward,
        extra_fields={
            "group_id": group,
            "model_tag": "actor",
            "run_id": run,
            "rollout_id": f"{group}-{run}",
            "expected_rollouts": expected_rollouts,
        },
    )


def _round(*, split=False):
    return [
        _trajectory("a", 0, [101, 102, 103], split=split),
        _trajectory("a", 1, [104], split=split),
        _trajectory("b", 0, [105], split=split),
        _trajectory("b", 1, [106, 107, 108, 109, 110, 111], split=split),
    ]


def _ratios(parameter, target_tokens):
    # Ratio 4 exercises dual clipping for negative advantages; other ratios
    # exercise both ordinary clipping boundaries and the unclipped region.
    values = (0.6, 4.0, 1.1, 1.05, 0.9)
    bases = torch.tensor(
        [values[token % len(values)] for token in target_tokens], dtype=torch.float64
    )
    return bases * (parameter - 0.2).exp()


def _policy_losses(advantages, ratios, loss_fn):
    losses = -advantages * ratios
    if loss_fn == "ppo_clip":
        losses = torch.maximum(losses, -advantages * ratios.clamp(0.8, 1.2))
        losses = torch.where(advantages < 0, torch.minimum(losses, -advantages * 3.0), losses)
    return losses


class _PolicyHolder(_AccumulatingHolder):
    async def forward_backward(self, **kwargs):
        self.builtin_calls.append(kwargs)
        self.events.append(("forward_backward", len(kwargs["datums_data"])))
        assert kwargs["loss_fn_config"]["loss_agg_mode"] == "sum"
        loss = self.parameter * 0
        tokens = 0
        for datum in kwargs["datums_data"]:
            inputs = datum["loss_fn_inputs"]
            mask = torch.tensor(inputs["loss_mask"], dtype=torch.float64)
            advantages = torch.tensor(inputs["advantages"], dtype=torch.float64)
            ratios = _ratios(self.parameter, inputs["target_tokens"])
            loss = loss + (_policy_losses(advantages, ratios, kwargs["loss_fn"]) * mask).sum()
            tokens += int(mask.sum())
        loss.backward()
        return {"loss": loss.item(), "metrics": {"loss": loss.item(), "tokens": tokens}}


def _explicit_objective(trajectories, parameter, loss_fn):
    """Average complete groups directly, without relying on converted Datums."""
    losses = defaultdict(list)
    for trajectory in trajectories:
        group = (trajectory.get("model_tag"), trajectory["group_id"])
        losses[group]  # Successful groups with no sequences still count in the mean.
        sequences = trajectory.segments if trajectory.segments is not None else [trajectory]
        for sequence in sequences:
            mask = sequence["loss_mask"][1:]
            targets = [token for token, keep in zip(sequence["tokens"][1:], mask) if keep]
            advantages = torch.full(
                (len(targets),), float(trajectory["advantage"]), dtype=torch.float64
            )
            losses[group].extend(_policy_losses(advantages, _ratios(parameter, targets), loss_fn))
    return torch.stack(
        [torch.stack(values).mean() if values else parameter * 0 for values in losses.values()]
    ).mean()


def _train(monkeypatch, trajectories, *, loss_fn="importance_sampling", mini_batch_size=None):
    trainer = _grpo_trainer(monkeypatch, "prompt_mean", mini_batch_size)
    trainer._loss_fn = loss_fn
    trainer._clip_ratio_low = trainer._clip_ratio_high = 0.2
    trainer._clip_ratio_c = 3.0
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(trajectories))
    return holder, metrics


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
@pytest.mark.parametrize("mini_batch_size", [None, 1, 3])
def test_prompt_mean_matches_explicit_group_loss_gradient_and_update(
    monkeypatch, loss_fn, mini_batch_size
):
    trajectories = _round()
    holder, metrics = _train(
        monkeypatch, trajectories, loss_fn=loss_fn, mini_batch_size=mini_batch_size
    )
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    expected = _explicit_objective(trajectories, parameter, loss_fn)
    gradient = torch.autograd.grad(expected, parameter)[0]

    assert gradient.abs() > 0.01
    assert metrics["weaver/loss"] == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)
    assert holder.gradients == pytest.approx([gradient.item()], rel=1e-12, abs=1e-12)
    torch.testing.assert_close(holder.parameter, parameter - 0.1 * gradient, rtol=1e-12, atol=1e-12)
    assert len(holder.optim_calls) == metrics["training/optimizer_step_count"] == 1

    # Aggregation must preserve the behavior distribution and candidate sets.
    for call in holder.builtin_calls:
        for datum in call["datums_data"]:
            inputs = datum["loss_fn_inputs"]
            assert inputs["logprobs"] == [-token / 1000 for token in inputs["target_tokens"]]
            assert inputs["sampling_mask"] == [
                [token, token + 1000] for token in inputs["target_tokens"]
            ]


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
def test_segment_boundaries_and_microbatches_preserve_prompt_objective(monkeypatch, loss_fn):
    baseline, baseline_metrics = _train(monkeypatch, _round(), loss_fn=loss_fn)
    split, split_metrics = _train(
        monkeypatch, _round(split=True), loss_fn=loss_fn, mini_batch_size=3
    )

    assert baseline_metrics["batch/segment_count"] == 4
    assert split_metrics["batch/segment_count"] == 11
    assert split.events == [("forward_backward", 3)] * 3 + [
        ("forward_backward", 2),
        ("optim_step", None),
    ]
    assert split_metrics["weaver/loss"] == pytest.approx(baseline_metrics["weaver/loss"])
    assert split.gradients == pytest.approx(baseline.gradients, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(split.parameter, baseline.parameter, rtol=1e-12, atol=1e-12)


def test_full_round_keeps_each_sequence_once_and_accumulates_before_one_update(monkeypatch):
    outcomes = []
    for size in (None, 1, 16):
        trainer = _grpo_trainer(monkeypatch, "prompt_mean", size)
        holder = _AccumulatingHolder()
        trainer.set_service_holder(holder)
        metrics, _, _ = run_coro(trainer.train(_grpo_round()))
        datums = [datum for call in holder.builtin_calls for datum in call["datums_data"]]
        assert [datum["input_tokens"][0] for datum in datums] == list(range(1, 101))
        assert holder.events[-1] == ("optim_step", None)
        assert len(holder.optim_calls) == metrics["training/optimizer_step_count"] == 1
        outcomes.append((holder, metrics))
    baseline, baseline_metrics = outcomes[0]
    for holder, metrics in outcomes[1:]:
        assert holder.gradients == pytest.approx(baseline.gradients, rel=1e-12, abs=1e-12)
        assert metrics["weaver/loss"] == pytest.approx(baseline_metrics["weaver/loss"])
        torch.testing.assert_close(holder.parameter, baseline.parameter, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("empty", [False, True])
def test_zero_advantage_and_zero_sequence_groups_still_count_in_group_mean(monkeypatch, empty):
    baseline, baseline_metrics = _train(monkeypatch, _round())
    trajectories = _round() + [
        _trajectory("c", 0, [] if empty else [112, 113], reward=7.0),
        _trajectory("c", 1, [] if empty else [114, 115, 116, 117], reward=7.0),
    ]
    holder, metrics = _train(monkeypatch, trajectories, mini_batch_size=1)

    assert metrics["weaver/loss"] == pytest.approx(baseline_metrics["weaver/loss"] * 2 / 3)
    assert holder.gradients == pytest.approx([baseline.gradients[0] * 2 / 3])
    assert metrics["batch/trainable_segment_count"] == (4 if empty else 6)
    assert all(float(trajectory["advantage"]) == 0 for trajectory in trajectories[-2:])


def test_zero_advantage_tokens_within_nonconstant_group_remain_in_denominator(monkeypatch):
    trajectories = [
        _trajectory("a", 0, [101], expected_rollouts=3),
        _trajectory("a", 1, [102, 103, 104, 105, 106], expected_rollouts=3),
        _trajectory("a", 2, [107], expected_rollouts=3),
    ]
    holder, _ = _train(monkeypatch, trajectories)
    datums = holder.builtin_calls[0]["datums_data"]

    assert len(datums) == 3
    assert datums[1]["loss_fn_inputs"]["advantages"] == [0.0] * 5
    assert datums[0]["loss_fn_inputs"]["advantages"] == pytest.approx(
        [float(trajectories[0]["advantage"]) / 7]
    )


class _MismatchHolder(_PolicyHolder):
    kept_targets = {104, 106}

    async def compute_target_logprobs_batch(
        self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
    ):
        assert loss_fn_config == {"temperature": 1.0}
        # Reject most tokens while retaining one token in each original group.
        return [
            [
                -token / 1000 + math.log(1.0 if token in self.kept_targets else 4.0)
                for token in tokens[1:]
            ]
            for tokens in tokens_batch
        ]


@pytest.mark.parametrize("reject_entire_group", [False, True])
def test_mis_rejection_keeps_original_group_token_denominators(monkeypatch, reject_entire_group):
    trajectories = _round()
    for trajectory in trajectories:
        for sequence in trajectory.segments:
            sequence.pop("sampling_mask")
    trainer = _grpo_trainer(
        monkeypatch,
        "prompt_mean",
        1,
        sampling_logprob_source="recomputed",
        mis=MISConfig(enabled=True, tis_mode="mask", tis_upper_bound=2.0),
    )
    holder = _MismatchHolder()
    if reject_entire_group:
        holder.kept_targets = {104}
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert metrics["training/optimizer_step_count"] == 1
    kept = []
    for call in holder.builtin_calls:
        for datum in call["datums_data"]:
            inputs = datum["loss_fn_inputs"]
            for token, mask, advantage in zip(
                inputs["target_tokens"], inputs["loss_mask"], inputs["advantages"], strict=True
            ):
                if mask:
                    kept.append((token, advantage))
    expected = [(104, float(trajectories[1]["advantage"]) / (2 * 4))]
    if not reject_entire_group:
        expected.append((106, float(trajectories[3]["advantage"]) / (2 * 7)))
    assert [token for token, _ in kept] == [token for token, _ in expected]
    assert [advantage for _, advantage in kept] == pytest.approx(
        [advantage for _, advantage in expected]
    )


@pytest.mark.parametrize("missing_value", [None, ""])
def test_missing_group_identity_fails_before_forward_backward(monkeypatch, missing_value):
    trajectories = _round()
    for trajectory in trajectories[:2]:
        if missing_value is None:
            trajectory.extra_fields.pop("group_id")
        else:
            trajectory["group_id"] = missing_value
    trainer = _grpo_trainer(monkeypatch, "prompt_mean")
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)

    with pytest.raises(ValueError, match="group"):
        run_coro(trainer.train(trajectories))
    assert holder.events == holder.saved_paths == []


@pytest.mark.parametrize("empty_sequences", [False, True])
def test_no_trainable_tokens_skips_optimizer_and_weight_export(monkeypatch, empty_sequences):
    trajectories = _round()
    for trajectory in trajectories:
        if empty_sequences:
            trajectory.segments = []
        else:
            for sequence in trajectory.segments:
                sequence["loss_mask"] = [0] * len(sequence["tokens"])
                sequence["sampling_mask"] = []
    trainer = _grpo_trainer(monkeypatch, "prompt_mean")
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)

    metrics, sampling_path, weight_version = run_coro(trainer.train(trajectories))

    assert metrics["training/skipped_flag"] == 1
    assert metrics["training/optimizer_step_count"] == 0
    assert (sampling_path, weight_version) == (None, None)
    assert holder.events == holder.saved_paths == []


@pytest.mark.parametrize("failed", [False, True])
def test_repaired_group_counts_in_prompt_mean_while_fully_skipped_group_does_not(
    monkeypatch, failed
):
    baseline, baseline_metrics = _train(monkeypatch, _round())
    excluded = deepcopy(_round()[:2])
    for trajectory in excluded:
        trajectory["group_id"] = "excluded"
        trajectory["rollout_id"] = "excluded-" + str(trajectory["run_id"])
        if not failed:
            trajectory["group_skipped"] = True
    if failed:
        excluded[0].status = "failed"
        excluded[0].reward = None
    holder, metrics = _train(monkeypatch, _round() + excluded)

    # Repeating the sole valid member produces a zero-advantage third group.
    # It still contributes one prompt to the mean; a wholly skipped group does not.
    scale = 2 / 3 if failed else 1.0
    assert metrics["batch/trainable_segment_count"] == (6 if failed else 4)
    assert metrics["batch/resampled_group_count"] == int(failed)
    assert metrics["weaver/loss"] == pytest.approx(baseline_metrics["weaver/loss"] * scale)
    assert holder.gradients == pytest.approx([value * scale for value in baseline.gradients])


def test_same_group_id_for_distinct_models_keeps_separate_prompt_weights(monkeypatch):
    trajectories = _round()
    for trajectory in trajectories[2:]:
        trajectory["group_id"] = "a"
        trajectory["model_tag"] = "other-actor"
    holder, metrics = _train(monkeypatch, trajectories)
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    expected = _explicit_objective(trajectories, parameter, "importance_sampling")
    gradient = torch.autograd.grad(expected, parameter)[0]

    assert metrics["weaver/loss"] == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)
    assert holder.gradients == pytest.approx([gradient.item()], rel=1e-12, abs=1e-12)


def test_float_binary_masks_preserve_prompt_mean_loss_and_gradient(monkeypatch):
    baseline, baseline_metrics = _train(monkeypatch, _round())
    trajectories = _round()
    for trajectory in trajectories:
        for sequence in trajectory.segments:
            sequence["loss_mask"] = [float(value) for value in sequence["loss_mask"]]
    holder, metrics = _train(monkeypatch, trajectories, mini_batch_size=1)

    assert metrics["weaver/loss"] == pytest.approx(baseline_metrics["weaver/loss"])
    assert holder.gradients == pytest.approx(baseline.gradients, rel=1e-12, abs=1e-12)
    torch.testing.assert_close(holder.parameter, baseline.parameter, rtol=1e-12, atol=1e-12)
