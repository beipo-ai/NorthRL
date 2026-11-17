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

"""Tool-error rebalancing uses final advantages over all parents in a train step."""

import math
from array import array

import pytest

from northrl.algorithm.advantage import build_estimator, build_postprocessor
from northrl.algorithm.advantage.tool_error_rebalance import ToolErrorRebalancePostprocessor
from northrl.utils.algorithm_config import EstimatorConfig, PostprocessingConfig
from northrl.utils.async_utils import run_coro
from northrl.utils.trajectory_sequences import iter_training_sequences
from tests.unittests.algorithm.test_tool_error_rebalance import PREFIX, _annotation
from tests.unittests.trainer.test_prompt_mean import _PolicyHolder
from tests.unittests.trainer.test_reinforce_plus_plus_trainer import (
    ESTIMATORS,
    _expected_advantages,
    _round,
    _submitted_advantages,
)
from tests.unittests.trainer.test_reinforce_plus_plus_trainer import _trainer as _reinforce_trainer
from tests.unittests.trainer.test_remote_api_trainer_loss_aggregation import _AccumulatingHolder
from tests.unittests.trainer.test_step_advantage_execution import _parents, _trainer


@pytest.mark.parametrize("batch_size,mini_batch_size", [(None, None), (1, 1), (3, 2), (8, 1)])
def test_tool_error_rebalance_runs_once_before_mis_dump_and_batch_splitting(
    monkeypatch, batch_size, mini_batch_size
):
    events = []
    trainer, dumper = _trainer(
        monkeypatch, events, batch_size=batch_size, mini_batch_size=mini_batch_size
    )
    postprocessor = build_postprocessor(PostprocessingConfig(mode="tool_error_rebalance"))
    parents = _parents()
    sequences = list(iter_training_sequences(parents))
    buffers = [sequence["tokens"] for sequence in sequences]
    # Negative error in group 0; negative clean in group 1. These must share a pool.
    for index, sequence in enumerate(sequences):
        sequence["turn_annotations"] = [
            _annotation(reasons=["missing_tool_name"] if index == 0 else [])
        ]
    calls = []

    class RecordingPostprocessor:
        def process(self, trajectories, metrics, *, logger=None):
            calls.append(trajectories)
            assert all(sequence["old_log_probs"][1] == -0.1 for sequence in sequences)
            assert all("correction_mask" not in sequence for sequence in sequences)
            shaped = postprocessor.process(trajectories, metrics, logger=logger)
            assert shaped is trajectories
            return shaped

    trainer._advantage_postprocessor = RecordingPostprocessor()
    metrics, _, _ = run_coro(trainer.train(parents))
    assert len(calls) == 1 and len(calls[0]) == 4
    assert dumper.prepared[-1].segments == []
    assert metrics[PREFIX + "negative_scale"] == 0.5
    assert metrics[PREFIX + "negative_scale_clamped"] == 1
    assert metrics[PREFIX + "negative_error_tokens"] == 1
    for index, (sequence, tokens) in enumerate(zip(sequences, buffers, strict=True)):
        assert sequence["tokens"] is tokens
        assert sequence["token_advantages"][0] == pytest.approx(
            sequence["base_token_advantages"][0] * (2 if index == 0 else 0.5 if index == 3 else 1)
        )


@pytest.mark.parametrize("flow", ["grpo", "opd"])
@pytest.mark.parametrize("second_postprocessing", [True, False])
def test_fresh_training_estimation_replaces_prior_tool_error_rebalance_cache(
    monkeypatch, flow, second_postprocessing
):
    trainer, _ = _trainer(monkeypatch, [], flow=flow, mis=False)
    trainer._advantage_postprocessor = ToolErrorRebalancePostprocessor()
    if flow == "grpo":
        trainer._advantage_estimator = build_estimator(
            EstimatorConfig(mode="grpo", params={"norm_adv_by_std": False})
        )
    parents = _parents()[:2]
    sequences = list(iter_training_sequences(parents))
    buffers = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]
    for index, sequence in enumerate(sequences):
        sequence["turn_annotations"] = [
            _annotation(reasons=["missing_tool_name"] if index == 0 else [])
        ]
    teacher_score = -2.0

    async def teacher(prepared_sequences):
        return [[0.0, teacher_score] for _ in prepared_sequences]

    if flow == "opd":
        monkeypatch.setattr(trainer, "_compute_teacher_logprobs_parallel", teacher)
    run_coro(trainer.train(parents))
    first_base = [list(sequence["base_token_advantages"]) for sequence in sequences]
    # Replay is idempotent; a new estimate must instead replace both cached arrays.
    shaped = [list(sequence["token_advantages"]) for sequence in sequences]
    trainer._advantage_postprocessor.process(parents, {})
    assert [sequence["token_advantages"] for sequence in sequences] == shaped
    parents[1].reward = 4.0
    teacher_score = -4.0
    if not second_postprocessing:
        trainer._advantage_postprocessor = None
    for sequence in sequences:
        sequence["old_log_probs"] = [0.0, -0.2]
    run_coro(trainer.train(parents))
    expected = [-2.0, 2.0, 2.0] if flow == "grpo" else [-1.9, -1.9, -1.9]
    if second_postprocessing:
        assert [sequence["base_token_advantages"][0] for sequence in sequences] == pytest.approx(
            expected
        )
        assert [sequence["base_token_advantages"] for sequence in sequences] != first_base
    else:
        assert all("base_token_advantages" not in sequence for sequence in sequences)
        if flow == "grpo":
            assert all("token_advantages" not in sequence for sequence in sequences)
            assert [float(parent["advantage"]) for parent in parents] == [-2, 2]
        else:
            assert [sequence["token_advantages"][0] for sequence in sequences] == pytest.approx(
                expected
            )
    for sequence, (tokens, mask) in zip(sequences, buffers, strict=True):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask


@pytest.mark.parametrize("mini_batch_size", [None, 1])
def test_tool_error_rebalance_datum_mean_keeps_zero_error_datum_in_denominator(
    monkeypatch, mini_batch_size
):
    trainer, dumper = _trainer(monkeypatch, [], mini_batch_size=mini_batch_size, mis=False)
    trainer._algorithm_config.loss.reduction.mode = "datum_mean"
    trainer._advantage_estimator = build_estimator(
        EstimatorConfig(mode="grpo", params={"norm_adv_by_std": False})
    )
    trainer._advantage_postprocessor = ToolErrorRebalancePostprocessor()
    holder = _AccumulatingHolder()
    trainer.set_service_holder(holder)
    parents = _parents()[:2]
    negative = parents[0].segments[0]
    negative.update(
        tokens=[10, 20, 30, 40],
        loss_mask=[0, 1, 0, 1],
        old_log_probs=[0.0, -0.2, 0.0, -0.2],
        turn_annotations=[_annotation(1, 2), _annotation(3, 4)],
    )
    parents[1].segments = parents[1].segments[:1]
    parents[1].segments[0]["turn_annotations"] = [_annotation(reasons=["missing_tool_name"])]
    run_coro(trainer.train(parents))
    datums = [datum for call in holder.builtin_calls for datum in call["datums_data"]]
    assert len(datums) == 2
    by_length = {len(datum["input_tokens"]): datum["loss_fn_inputs"] for datum in datums}
    assert by_length[3]["loss_mask"] == [1, 0, 1]
    # Two nonempty datums, with two and one responses respectively. The all-zero
    # positive error still counts, so each negative token has denominator 2 * 2.
    assert by_length[3]["advantages"] == [-0.25, 0, -0.25]
    assert by_length[1]["advantages"] == [0]
    assert by_length[1]["loss_mask"] == [1]
    assert holder.gradients == pytest.approx([0.5 * math.exp(0.2)])
    assert negative["token_advantages"] == [-1, -1]
    assert dumper.prepared[1].segments[0]["token_advantages"] == [0]


def _reinforce_round(*, split, compact):
    parents = _round(split=split)
    for sequence in iter_training_sequences(parents):
        sequence["turn_annotations"] = [
            _annotation(
                index,
                index + 1,
                reasons=["missing_tool_name"] if token in {101, 111} else [],
            )
            for index, (token, keep) in enumerate(
                zip(sequence["tokens"], sequence["loss_mask"], strict=True)
            )
            if keep
        ]
        if compact:
            for field, typecode in [("tokens", "q"), ("loss_mask", "b"), ("old_log_probs", "d")]:
                sequence[field] = array(typecode, sequence[field])
    return parents


def _token_values(parents, field="token_advantages"):
    return {
        token: value
        for sequence in iter_training_sequences(parents)
        for token, value in zip(
            (
                token
                for token, keep in zip(sequence["tokens"], sequence["loss_mask"], strict=True)
                if keep
            ),
            sequence[field],
            strict=True,
        )
    }


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
@pytest.mark.parametrize("split,compact", [(False, False), (True, False), (True, True)])
def test_reinforce_rebalance_uses_complete_step_before_real_datum_conversion(
    monkeypatch, estimator, gamma, split, compact
):
    outcomes = []
    for batch_size in (None, 1):
        parents = _reinforce_round(split=split, compact=compact)
        expected = _expected_advantages(parents, estimator, gamma)
        trainer = _reinforce_trainer(
            monkeypatch, estimator, gamma=gamma, batch_size=batch_size, mini_batch_size=1
        )
        holder = _PolicyHolder()
        trainer.set_service_holder(holder)
        calls = []

        class RecordingPostprocessor:
            def process(self, trajectories, metrics, *, logger=None):
                calls.append(trajectories)
                assert len(trajectories) == 4
                assert holder.builtin_calls == []
                return ToolErrorRebalancePostprocessor().process(
                    trajectories, metrics, logger=logger
                )

        trainer._advantage_postprocessor = RecordingPostprocessor()
        metrics, _, _ = run_coro(trainer.train(parents))
        assert len(calls) == 1
        assert _token_values(parents, "base_token_advantages") == pytest.approx(expected, abs=2e-7)
        shaped = _token_values(parents)
        assert shaped[101] == pytest.approx(2 * expected[101], abs=2e-7)
        assert shaped[111] == 0
        assert 1 < metrics[PREFIX + "positive_scale"] <= 2
        assert 0.5 <= metrics[PREFIX + "negative_scale"] < 1
        assert metrics[PREFIX + "response_tokens"] == 11
        assert metrics[PREFIX + "error_tokens"] == 2
        assert metrics[PREFIX + "clean_tokens"] == 9
        # Loss normalization uses each optimizer batch, but shaping and whitening
        # must still use the complete step even when every datum gets its own update.
        denominators = {
            token: sum(sequence["loss_mask"]) if batch_size else len(expected)
            for sequence in iter_training_sequences(parents)
            for token, keep in zip(sequence["tokens"], sequence["loss_mask"], strict=True)
            if keep
        }
        assert _submitted_advantages(holder) == pytest.approx(
            {token: value / denominators[token] for token, value in shaped.items()}, abs=2e-7
        )
        sequence_count = sum(1 for _ in iter_training_sequences(parents))
        assert len(holder.optim_calls) == (sequence_count if batch_size else 1)
        outcomes.append(
            (shaped, {key: value for key, value in metrics.items() if key.startswith(PREFIX)})
        )
    assert outcomes[1][0] == pytest.approx(outcomes[0][0], abs=2e-7)
    assert outcomes[1][1] == pytest.approx(outcomes[0][1], abs=2e-7)


@pytest.mark.parametrize("estimator,gamma", ESTIMATORS)
@pytest.mark.parametrize("split,compact", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("second_postprocessing", [True, False])
def test_reinforce_reestimation_refreshes_rebalance_cache_and_submitted_token_values(
    monkeypatch, estimator, gamma, split, compact, second_postprocessing
):
    parents = _reinforce_round(split=split, compact=compact)
    sequences = list(iter_training_sequences(parents))
    buffers = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]
    trainer = _reinforce_trainer(monkeypatch, estimator, gamma=gamma, mini_batch_size=1)
    trainer._advantage_postprocessor = ToolErrorRebalancePostprocessor()
    trainer.set_service_holder(_PolicyHolder())
    run_coro(trainer.train(parents))
    first_base = _token_values(parents, "base_token_advantages")
    first_shaped = _token_values(parents)
    trainer._advantage_postprocessor.process(parents, {})
    assert _token_values(parents) == first_shaped

    for parent, reward in zip(parents, [8.0, 1.0, 3.0, 12.0], strict=True):
        parent.reward = reward
    expected = _expected_advantages(parents, estimator, gamma)
    if not second_postprocessing:
        trainer._advantage_postprocessor = None
    holder = _PolicyHolder()
    trainer.set_service_holder(holder)
    metrics, _, _ = run_coro(trainer.train(parents))
    if second_postprocessing:
        assert _token_values(parents, "base_token_advantages") == pytest.approx(expected, abs=2e-7)
        assert _token_values(parents, "base_token_advantages") != first_base
        assert metrics[PREFIX + "response_tokens"] == 11
    else:
        assert all("base_token_advantages" not in sequence for sequence in sequences)
        assert _token_values(parents) == pytest.approx(expected, abs=2e-7)
        assert not any(key.startswith(PREFIX) for key in metrics)
    assert _submitted_advantages(holder) == pytest.approx(
        {token: value / len(expected) for token, value in _token_values(parents).items()}, abs=2e-7
    )
    for sequence, (tokens, mask) in zip(sequences, buffers, strict=True):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask
        assert isinstance(sequence["token_advantages"], array if compact else list)
        if second_postprocessing:
            assert isinstance(sequence["base_token_advantages"], array if compact else list)
