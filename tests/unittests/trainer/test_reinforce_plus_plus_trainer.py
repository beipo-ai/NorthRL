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

"""REINFORCE++ estimator integration through complete remote training rounds."""

from array import array
from collections import defaultdict
from unittest.mock import AsyncMock

import pytest
import torch

from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import KLConfig, MISConfig
from northrl.utils.trajectory_sequences import iter_training_sequences
from tests.unittests.trainer.test_prompt_mean import (
    _MismatchHolder,
    _policy_losses,
    _PolicyHolder,
    _ratios,
    _trajectory,
)
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _FakeWeightSyncController,
    _make_config,
)

ESTIMATORS = [
    ("reinforce_plus_plus", 1.0),
    ("reinforce_plus_plus", 0.6),
    ("reinforce_plus_plus_baseline", 1.0),
]


def _trainer(
    monkeypatch,
    estimator,
    *,
    gamma=1.0,
    loss_fn="importance_sampling",
    batch_size=None,
    mini_batch_size=None,
    mode="token_mean",
    kl_mode=None,
    source="inference",
    mis=None,
):
    monkeypatch.setattr(
        trainer_module, "get_data_dumper", lambda *args, **kwargs: _DummyDataDumper()
    )
    config = _make_config(batch_size=batch_size, mini_batch_size=mini_batch_size)
    config.algorithm.advantage = {"estimator": {"mode": estimator, "params": {"gamma": gamma}}}
    config.algorithm.sampling_logprob_source = source
    if mis is not None:
        config.algorithm.mis = mis.model_dump()
    config.algorithm.loss.mode = loss_fn
    config.algorithm.loss.reduction.mode = mode
    config.algorithm.loss.params.clip.ratio_low = config.algorithm.loss.params.clip.ratio_high = 0.2
    config.algorithm.loss.params.clip.ratio_c = 3.0
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    config.training.weight_sync = {"mode": "sync"}
    config.data = {"train": {"rollout_repeat_n": 2}}
    if kl_mode is not None:
        config.algorithm.kl.mode = kl_mode
        config.algorithm.kl.params = {"coef": 0.3, "estimator": "k1"}
        config.models.reference = {"name": "reference", "checkpoint": None}
    trainer = RemoteApiGrpoTrainer(config)
    trainer._weight_sync_controller = _FakeWeightSyncController()
    return trainer


def _round(*, split=False):
    return [
        _trajectory("a", 0, [101, 102, 103], split=split, reward=1.0),
        _trajectory("a", 1, [104], split=split, reward=4.0),
        _trajectory("b", 0, [105], split=split, reward=10.0),
        _trajectory("b", 1, [106, 107, 108, 109, 110, 111], split=split, reward=14.0),
    ]


def _response_tokens(trajectory):
    return [
        token
        for sequence in trajectory.segments
        for token, keep in zip(sequence["tokens"][1:], sequence["loss_mask"][1:], strict=True)
        if keep
    ]


def _expected_advantages(trajectories, estimator, gamma):
    """Independent scalar-reward return calculation over complete response streams."""
    scores = {trajectory["rollout_id"]: trajectory.reward for trajectory in trajectories}
    groups = defaultdict(list)
    for trajectory in trajectories:
        groups[(trajectory["model_tag"], trajectory["group_id"])].append(
            scores[trajectory["rollout_id"]]
        )
    values = {}
    for trajectory in trajectories:
        tokens = _response_tokens(trajectory)
        if estimator == "reinforce_plus_plus_baseline":
            group = groups[(trajectory["model_tag"], trajectory["group_id"])]
            centered = scores[trajectory["rollout_id"]] - sum(group) / len(group)
            values.update(dict.fromkeys(tokens, centered))
        else:
            for index, token in enumerate(tokens):
                values[token] = gamma ** (len(tokens) - 1 - index) * trajectory.reward
    raw = torch.tensor(list(values.values()), dtype=torch.float64)
    whitened = (raw - raw.mean()) / (raw.var(unbiased=True) + 1e-8).sqrt()
    return dict(zip(values, whitened.tolist(), strict=True))


def _submitted_advantages(holder):
    return {
        token: advantage
        for call in holder.builtin_calls
        for datum in call["datums_data"]
        for token, advantage, mask in zip(
            datum["loss_fn_inputs"]["target_tokens"],
            datum["loss_fn_inputs"]["advantages"],
            datum["loss_fn_inputs"]["loss_mask"],
            strict=True,
        )
        if mask
    }


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
@pytest.mark.parametrize("mini_batch_size", [None, 1, 3])
def test_native_loss_matches_global_token_objective(
    monkeypatch, estimator, gamma, loss_fn, mini_batch_size
):
    trajectories = _round()
    expected = _expected_advantages(trajectories, estimator, gamma)
    trainer = _trainer(
        monkeypatch,
        estimator,
        gamma=gamma,
        loss_fn=loss_fn,
        mini_batch_size=mini_batch_size,
    )
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(trajectories))

    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor(list(expected.values()), dtype=torch.float64)
    objective = _policy_losses(advantages, _ratios(parameter, list(expected)), loss_fn).mean()
    gradient = torch.autograd.grad(objective, parameter)[0]
    assert gradient.abs() > 0.01
    assert _submitted_advantages(holder) == pytest.approx(
        {token: value / len(expected) for token, value in expected.items()}, abs=2e-7
    )
    assert holder.gradients == pytest.approx([gradient.item()], rel=2e-6, abs=2e-7)
    assert metrics["weaver/loss"] == pytest.approx(objective.item(), rel=2e-6, abs=2e-7)
    torch.testing.assert_close(holder.parameter, parameter - 0.1 * gradient, rtol=2e-6, atol=2e-7)
    assert metrics["training/optimizer_step_count"] == len(holder.optim_calls) == 1
    assert holder.custom_calls == []
    assert all(call["loss_fn"] == loss_fn for call in holder.builtin_calls)
    assert metrics["algorithm/trajectory/advantage_token_weighted_mean"] == pytest.approx(
        0.0, abs=2e-7
    )
    means = [
        sum(expected[token] for token in _response_tokens(t)) / len(_response_tokens(t))
        for t in trajectories
    ]
    assert [t["advantage"] for t in trajectories] == pytest.approx(means, abs=2e-7)
    assert metrics["algorithm/trajectory/advantage_mean"] == pytest.approx(
        sum(means) / len(means), abs=2e-7
    )


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
def test_segment_boundaries_and_microbatches_preserve_returns_and_update(
    monkeypatch, estimator, gamma
):
    outcomes = []
    for split in (False, True):
        trainer = _trainer(
            monkeypatch, estimator, gamma=gamma, mini_batch_size=3 if split else None
        )
        holder = _PolicyHolder()
        trainer.set_service_holder(holder)
        metrics, _, _ = run_coro(trainer.train(_round(split=split)))
        outcomes.append((holder, metrics))
    (whole, whole_metrics), (split, split_metrics) = outcomes
    assert whole_metrics["batch/segment_count"] == 4
    assert split_metrics["batch/segment_count"] == 11
    assert _submitted_advantages(split) == pytest.approx(_submitted_advantages(whole), abs=2e-7)
    assert split.gradients == pytest.approx(whole.gradients, rel=2e-6, abs=2e-7)
    assert split_metrics["weaver/loss"] == pytest.approx(whole_metrics["weaver/loss"], abs=2e-7)
    assert split.events == [("forward_backward", 3)] * 3 + [
        ("forward_backward", 2),
        ("optim_step", None),
    ]


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_repeated_training_replaces_prior_token_advantages(monkeypatch, estimator):
    trajectories = _round(split=True)
    trainer = _trainer(monkeypatch, estimator)
    trainer.set_service_holder(_PolicyHolder())
    run_coro(trainer.train(trajectories))

    for trajectory, reward in zip(trajectories, [7.0, 2.0, 3.0, 12.0], strict=True):
        trajectory.reward = reward
    expected = _expected_advantages(trajectories, estimator, 1.0)
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    run_coro(trainer.train(trajectories))
    assert _submitted_advantages(holder) == pytest.approx(
        {token: value / len(expected) for token, value in expected.items()}, abs=2e-7
    )


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_constant_returns_keep_zero_advantage_response_tokens(monkeypatch, estimator):
    trajectories = _round()
    for trajectory in trajectories:
        trajectory.reward = 7.0
    trainer = _trainer(monkeypatch, estimator, mini_batch_size=1)
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(trajectories))
    assert _submitted_advantages(holder) == dict.fromkeys(range(101, 112), 0.0)
    assert metrics["batch/trainable_response_tokens_total"] == 11
    assert metrics["training/skipped_flag"] == 0
    assert len(holder.optim_calls) == 1


def test_all_masked_round_skips_optimizer_and_weight_export(monkeypatch):
    trajectories = _round()
    for sequence in iter_training_sequences(trajectories):
        sequence["loss_mask"] = [0] * len(sequence["tokens"])
        sequence["sampling_mask"] = []
    trainer = _trainer(monkeypatch, "reinforce_plus_plus")
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, sampling_path, version = run_coro(trainer.train(trajectories))
    assert metrics["training/skipped_flag"] == 1
    assert holder.builtin_calls == holder.optim_calls == holder.saved_paths == []
    assert sampling_path is version is None


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_empty_parent_contributes_to_baseline_without_an_empty_optimizer_step(
    monkeypatch, estimator
):
    monkeypatch.setattr(trainer_module.random, "shuffle", lambda datums: None)
    trajectories = _round()
    # A successfully scored parent has only a masked observation. Its reward
    # still participates in group a's baseline even though it has no targets.
    trajectories[0].segments = [
        {
            "tokens": [10, 11],
            "loss_mask": [0, 0],
            "old_log_probs": [0.0, -0.2],
            "sampling_mask": [],
        }
    ]
    trajectories[1].segments.insert(
        0,
        {
            "tokens": [20, 21],
            "loss_mask": [0, 0],
            "old_log_probs": [0.0, -0.3],
            "sampling_mask": [],
        },
    )
    expected = _expected_advantages(trajectories, estimator, 1.0)
    trainer = _trainer(monkeypatch, estimator, batch_size=1)
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train(trajectories))

    # Each remaining sequence is its own optimizer batch and normalizes by its
    # own active token count. Empty datums must not cause weight-decay updates.
    assert _submitted_advantages(holder) == pytest.approx(
        {
            token: expected[token] / len(_response_tokens(trajectory))
            for trajectory in trajectories[1:]
            for token in _response_tokens(trajectory)
        },
        abs=2e-7,
    )
    assert holder.events == [("forward_backward", 1), ("optim_step", None)] * 3
    assert len(holder.optim_calls) == metrics["training/optimizer_step_count"] == 3
    assert metrics["batch/trainable_trajectory_count"] == 3
    assert metrics["batch/trainable_segment_count"] == 3
    assert metrics["batch/trainable_response_tokens_total"] == 8
    assert trajectories[0]["advantage"] == 0.0
    assert trajectories[0].segments[0]["token_advantages"] == []


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
def test_mis_filters_after_full_round_whitening(monkeypatch, estimator, gamma):
    trajectories = _round()
    expected = _expected_advantages(trajectories, estimator, gamma)
    for sequence in iter_training_sequences(trajectories):
        sequence.pop("sampling_mask")
    trainer = _trainer(
        monkeypatch,
        estimator,
        gamma=gamma,
        mini_batch_size=1,
        source="recomputed",
        mis=MISConfig(enabled=True, tis_mode="mask", tis_upper_bound=2.0),
    )
    holder = _MismatchHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(trajectories))
    assert _submitted_advantages(holder) == pytest.approx(
        {token: expected[token] / 2 for token in holder.kept_targets}, abs=2e-7
    )
    assert metrics["batch/trainable_response_tokens_total"] == 2
    assert metrics["training/optimizer_step_count"] == 1


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
@pytest.mark.parametrize("mask_transport", ["inline", "reference"])
def test_actor_recomputation_precedes_estimation_without_extra_inputs(
    monkeypatch, estimator, gamma, mask_transport
):
    events = []

    def sampling_mask_ref(token):
        return {
            "sampling_mask_ref": {
                "schema": "weaver.sampling_mask.v1",
                "uri": f"weaver://offline/mask-{token}",
                "token_count": 1,
                "empty_row": "full-vocabulary",
                "future_field": 7,
            }
        }

    def expected_support(targets):
        token = int(targets[-1])
        if mask_transport == "reference":
            return {
                **sampling_mask_ref(token),
                "sampling_mask_slice": {"start": 0, "stop": 1},
                "sampling_mask_positions": [len(targets) - 1],
            }
        # Masked context targets use a harmless singleton support during forwards.
        return [[int(target)] for target in targets[:-1]] + [[token, token + 1000]]

    class RecomputedHolder(_PolicyHolder):
        async def compute_target_logprobs_batch(
            self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
        ):
            events.append("actor")
            assert loss_fn_config == {"temperature": 0.7}
            assert sampling_masks == [expected_support(tokens[1:]) for tokens in tokens_batch]
            return [[-token / 100 for token in tokens[1:]] for tokens in tokens_batch]

        async def forward_backward(self, **kwargs):
            events.append("backward")
            assert "kl_coeff" not in kwargs["loss_fn_config"]
            return await super().forward_backward(**kwargs)

    trainer = _trainer(monkeypatch, estimator, gamma=gamma, source="recomputed")
    trainer._training_temperature = 0.7
    trajectories = _round(split=True)
    if mask_transport == "reference":
        for sequence in iter_training_sequences(trajectories):
            sequence["sampling_mask"] = sampling_mask_ref(sequence["tokens"][-1])
    expected = _expected_advantages(trajectories, estimator, gamma)

    original_estimator = trainer._advantage_estimator

    class RecordingEstimator:
        def estimate(self, prepared, inputs, metrics, *, logger=None):
            events.append("estimate")
            assert inputs == {}
            for sequence in iter_training_sequences(prepared):
                assert list(sequence["old_log_probs"]) == [0.0] + [
                    -token / 100 for token in sequence["tokens"][1:]
                ]
                assert "ref_logprobs" not in sequence
            assert all("advantage" not in parent for parent in prepared)
            result = original_estimator.estimate(prepared, inputs, metrics, logger=logger)
            assert result is prepared
            return result

    class RecordingPostprocessor:
        def process(self, prepared, metrics, *, logger=None):
            events.append("postprocess")
            values = [
                value
                for sequence in iter_training_sequences(prepared)
                for value in sequence["token_advantages"]
            ]
            assert values == pytest.approx(list(expected.values()), abs=2e-7)
            return prepared

    trainer._advantage_estimator = RecordingEstimator()
    trainer._advantage_postprocessor = RecordingPostprocessor()

    holder = RecomputedHolder()
    trainer.set_service_holder(holder)
    run_coro(trainer.train(trajectories))
    assert events == ["actor", "estimate", "postprocess", "backward"]
    assert _submitted_advantages(holder) == pytest.approx(
        {token: value / len(expected) for token, value in expected.items()}, abs=2e-7
    )
    for call in holder.builtin_calls:
        for datum in call["datums_data"]:
            inputs = datum["loss_fn_inputs"]
            assert inputs["logprobs"] == [
                -token / 100 if mask else 0.0
                for token, mask in zip(inputs["target_tokens"], inputs["loss_mask"], strict=True)
            ]


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_disabled_kl_prepares_no_extra_inputs_or_reference_forward(monkeypatch, estimator):
    trainer = _trainer(monkeypatch, estimator)
    reference_forward = AsyncMock()
    monkeypatch.setattr(trainer, "_compute_and_attach_ref_logprobs", reference_forward)
    metrics = {}

    assert run_coro(trainer._prepare_advantage_inputs(_round(), metrics)) == {}
    reference_forward.assert_not_awaited()
    assert "kl/coefficient" not in metrics


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_reward_kl_remains_unsupported(monkeypatch, estimator):
    with pytest.raises(NotImplementedError, match="in_reward"):
        _trainer(monkeypatch, estimator, kl_mode="in_reward")


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_zero_or_one_token_segments_do_not_submit_empty_datums(monkeypatch, estimator):
    trainer = _trainer(monkeypatch, estimator)
    trajectories = _round()
    empty_sequences = [
        {"tokens": [], "loss_mask": [], "old_log_probs": [], "sampling_mask": []},
        {"tokens": [42], "loss_mask": [0], "old_log_probs": [0.0], "sampling_mask": []},
    ]
    trajectories[0].segments[:0] = empty_sequences
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)

    metrics, _, _ = run_coro(trainer.train(trajectories))

    assert metrics["batch/trainable_segment_count"] == 4
    assert metrics["training/optimizer_step_count"] == 1
    assert len(holder.builtin_calls[0]["datums_data"]) == 4
    for sequence in empty_sequences:
        assert sequence["token_advantages"] == []


def test_loss_side_kl_remains_in_native_loss_config(monkeypatch):
    trainer = _trainer(monkeypatch, "reinforce_plus_plus", kl_mode="in_loss")
    config = trainer._build_loss_fn_config()
    assert config["kl_coeff"] == 0.3
    assert config["kl_estimator"] == "k1"


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
def test_compact_response_advantages_keep_exact_service_payload_and_gradient(
    monkeypatch, estimator, gamma
):
    parents = _round(split=True)
    expected = _expected_advantages(parents, estimator, gamma)
    sequences = list(iter_training_sequences(parents))
    for sequence in sequences:
        for field, typecode in [("tokens", "q"), ("loss_mask", "b"), ("old_log_probs", "d")]:
            sequence[field] = array(typecode, sequence[field])
    buffers = [
        (sequence["tokens"], sequence["loss_mask"], sequence["old_log_probs"])
        for sequence in sequences
    ]
    trainer = _trainer(monkeypatch, estimator, gamma=gamma, loss_fn="ppo_clip", mini_batch_size=1)
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(parents))
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    objective = _policy_losses(
        torch.tensor(list(expected.values()), dtype=torch.float64),
        _ratios(parameter, list(expected)),
        "ppo_clip",
    ).mean()
    gradient = torch.autograd.grad(objective, parameter)[0]
    assert _submitted_advantages(holder) == pytest.approx(
        {token: value / len(expected) for token, value in expected.items()}, abs=2e-7
    )
    assert holder.gradients == pytest.approx([gradient.item()], abs=2e-7)
    assert metrics["weaver/loss"] == pytest.approx(objective.item(), abs=2e-7)
    for sequence, original in zip(sequences, buffers, strict=True):
        assert all(
            sequence[field] is value
            for field, value in zip(("tokens", "loss_mask", "old_log_probs"), original, strict=True)
        )
        assert isinstance(sequence["token_advantages"], array)
        assert sequence["token_advantages"].typecode == "d"
