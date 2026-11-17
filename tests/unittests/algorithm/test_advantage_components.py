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

"""Numerical and ownership contracts at the independent advantage boundary."""

import math
from array import array

import numpy as np
import pytest
import torch

from northrl.algorithm.advantage import (
    build_estimator,
    build_postprocessor,
    prepare_grpo_trajectories,
)
from northrl.algorithm.advantage.grpo import GRPOEstimator
from northrl.algorithm.advantage.opd import OPDEstimator
from northrl.algorithm.groups import validate_trajectory_groups
from northrl.northrl_types import Trajectory
from northrl.utils.algorithm_config import EstimatorConfig, PostprocessingConfig
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.logging_utils import (
    log_advantage_distribution_metrics,
    log_grpo_metrics,
)
from northrl.utils.trajectory_sequences import iter_training_sequences, iter_trajectory_sequences


def _sequence(token=1):
    return {
        "tokens": [token, 20, 21, 22],
        "loss_mask": [0, 1, 0, 1],
        "old_log_probs": [0.0, -1.0, -7.0, -2.0],
    }


def _parents():
    return [
        Trajectory(
            segments=[_sequence(2), _sequence(3)],
            reward=2.0,
            extra_fields={
                "task_id": np.int64(4),
                "group_id": "a",
                "run_id": 1,
                "rollout_id": "winner",
                "expected_rollouts": 2,
            },
        ),
        Trajectory(
            tokens=[1, 20, 21, 22],
            loss_mask=[0, 1, 0, 1],
            reward=0.0,
            extra_fields={
                "task_id": np.int64(4),
                "group_id": "a",
                "run_id": 0,
                "rollout_id": "loser",
                "expected_rollouts": 2,
                "old_log_probs": [0.0, -1.0, -7.0, -2.0],
            },
        ),
        Trajectory(
            segments=[],
            reward=10.0,
            extra_fields={"index": 2, "group_id": "b", "run_id": 0, "rollout_id": "empty"},
        ),
    ]


@pytest.mark.parametrize("normalize", [True, False])
def test_grpo_preserves_parent_math_metadata_metrics_and_original_buffers(normalize):
    parents = _parents()
    sequences = [sequence for parent in parents for sequence in iter_trajectory_sequences(parent)]
    buffers = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]
    parents[0].segments[0].update(group_id="stale", run_id=99, advantage=99.0)
    metrics = {}
    estimator = build_estimator(EstimatorConfig(mode="grpo", params={"norm_adv_by_std": normalize}))
    assert isinstance(estimator, GRPOEstimator)
    result = estimator.estimate(parents, {}, metrics)
    assert [parent["rollout_id"] for parent in result] == ["empty", "loser", "winner"]
    expected = 1.0 / (math.sqrt(2) + 1e-6) if normalize else 1.0
    assert [float(parent["advantage"]) for parent in result] == pytest.approx(
        [0.0, -expected, expected]
    )
    assert all(parent["advantage"].dtype == torch.float32 for parent in result)
    assert [float(seq["advantage"]) for seq in iter_training_sequences(result)] == (
        pytest.approx([-expected, expected, expected])
    )
    assert parents[0].segments[0]["group_id"] == "a"
    assert parents[0].segments[0]["run_id"] == 1
    assert result[2] is parents[0]
    assert [parent["rollout_id"] for parent in parents] == ["winner", "loser", "empty"]
    for sequence, (tokens, mask) in zip(sequences, buffers, strict=True):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask
        assert "token_advantages" not in sequence
    expected_metrics = {}
    log_grpo_metrics(result, expected_metrics)
    log_advantage_distribution_metrics(result, expected_metrics)
    assert metrics == expected_metrics


def test_grpo_baseline_counts_empty_parent_once_without_segment_weighting():
    parents = _parents()[:2]
    parents[0].segments = []
    result = GRPOEstimator(norm_adv_by_std=False).estimate(parents, {}, {})
    assert [float(parent["advantage"]) for parent in result] == [-1.0, 1.0]
    assert len(list(iter_training_sequences(result))) == 1


def _set_token_advantages(parents, location, value):
    target = {
        "flat": parents[1],
        "segment": parents[0].segments[0],
        "parent": parents[0],
    }[location]
    target["token_advantages"] = value


@pytest.mark.parametrize("mode", ["grpo", "rloo"])
@pytest.mark.parametrize("location", ["flat", "segment", "parent"])
@pytest.mark.parametrize(
    "value",
    [[], [0.2, 0.2], array("d", [0.2, 0.2])],
    ids=["empty", "constant-list", "constant-array"],
)
def test_scalar_estimators_reject_token_advantages_that_would_override_scalar(
    mode, location, value
):
    parents = _parents()
    _set_token_advantages(parents, location, value)
    estimator = build_estimator(EstimatorConfig(mode=mode))
    assert estimator is not None

    with pytest.raises(ValueError, match="token_advantages"):
        estimator.estimate(parents, {}, {})


@pytest.mark.parametrize("mode", ["grpo", "rloo"])
@pytest.mark.parametrize("location", ["flat", "segment", "parent"])
def test_scalar_estimators_allow_none_token_advantages_and_converter_uses_scalar(mode, location):
    parents = _parents()
    _set_token_advantages(parents, location, None)
    estimator = build_estimator(EstimatorConfig(mode=mode))
    assert estimator is not None
    result = estimator.estimate(parents, {}, {})

    datums = convert_trajectories_to_datums(result, drop_zero_advantage_datums=False)
    sequences = list(iter_training_sequences(result))
    assert len(datums) == len(sequences) == 3
    for datum, sequence in zip(datums, sequences, strict=True):
        advantage = float(sequence["advantage"])
        assert datum["loss_fn_inputs"]["advantages"] == pytest.approx([advantage, 0.0, advantage])


def test_grpo_preparation_validates_before_estimation_without_writing_advantages():
    parents = _parents()
    prepared = prepare_grpo_trajectories(parents)
    assert [parent["rollout_id"] for parent in prepared] == ["empty", "loser", "winner"]
    assert all("advantage" not in parent for parent in prepared)
    parents[0].status = "failed"
    groups = validate_trajectory_groups(parents)
    assert groups[(None, "a")] == parents[:2]
    assert all(parent.get("group_skipped") for parent in parents[:2])
    assert not parents[2].get("group_skipped", False)
    assert all("advantage" not in parent for parent in parents)


@pytest.mark.parametrize(
    "change,message",
    [
        ("duplicate", "Duplicate complete trajectory"),
        ("incomplete", "Incomplete trajectory group"),
        ("invalid_reward", "finite"),
        ("mixed_skip", "Inconsistent group_skipped"),
    ],
)
def test_group_validation_is_available_before_any_remote_work(change, message):
    parents = _parents()
    if change == "duplicate":
        parents[1]["rollout_id"] = parents[0]["rollout_id"]
    elif change == "incomplete":
        parents.pop(1)
    elif change == "invalid_reward":
        parents[1].reward = math.nan
    else:
        parents[1]["group_skipped"] = True
    with pytest.raises(ValueError, match=message):
        prepare_grpo_trajectories(parents)
    assert all("advantage" not in parent for parent in parents)


def _opd_step():
    first = _sequence()
    second = {"tokens": [4, 5], "loss_mask": [0, 1], "old_log_probs": [0.0, -3.0]}
    parents = [
        Trajectory(
            segments=[first, second],
            reward=0.0,
            # OPD must not acquire GRPO's complete-comparison-group validation.
            extra_fields={"group_id": "a", "run_id": 0, "expected_rollouts": 2},
        )
    ]
    inputs = {
        "sequences": [first, second],
        "student_logprobs": [first["old_log_probs"], second["old_log_probs"]],
        "teacher_logprobs": [[0.0, -2.0, -1.0, -4.0], [0.0, -4.0]],
    }
    return parents, inputs


@pytest.mark.parametrize("discount,expected", [(0.0, [-0.5, -1.0]), (0.5, [-0.75, -1.0])])
def test_opd_discount_counts_context_and_resets_at_segment_boundaries(discount, expected):
    parents, inputs = _opd_step()
    metrics = {}
    estimator = build_estimator(
        EstimatorConfig(mode="opd", params={"kl_penalty_coef": 0.5, "kl_discount_factor": discount})
    )
    assert isinstance(estimator, OPDEstimator)
    result = estimator.estimate(parents, inputs, metrics)
    assert result is parents
    assert parents[0].segments[0]["token_advantages"] == pytest.approx(expected)
    assert parents[0].segments[1]["token_advantages"] == [-0.5]
    assert "advantage" not in parents[0]
    assert metrics == {"opd/teacher_kl": 4.0 / 3.0, "opd/valid_tokens": 3}
    datums = convert_trajectories_to_datums(parents, drop_zero_advantage_datums=False)
    assert datums[0]["loss_fn_inputs"]["advantages"] == [expected[0], 0.0, expected[1]]


def test_opd_uses_captured_student_scores_after_recompute_replaces_old_logprobs():
    parents, inputs = _opd_step()
    buffers = [(seq["tokens"], seq["loss_mask"]) for seq in inputs["sequences"]]
    captured = inputs["student_logprobs"]
    for sequence in inputs["sequences"]:
        sequence["old_log_probs"] = [0.0] + [-99.0] * (len(sequence["tokens"]) - 1)
    OPDEstimator().estimate(parents, inputs, {})
    assert parents[0].segments[0]["token_advantages"] == [-1.0, -2.0]
    assert parents[0].segments[1]["token_advantages"] == [-1.0]
    assert captured[0] == [0.0, -1.0, -7.0, -2.0]
    for sequence, (tokens, mask) in zip(inputs["sequences"], buffers, strict=True):
        assert sequence["tokens"] is tokens
        assert sequence["loss_mask"] is mask
        assert sequence["old_log_probs"][1] == -99.0


@pytest.mark.parametrize("field", ["student_logprobs", "teacher_logprobs"])
@pytest.mark.parametrize("invalid", [None, [], [0.0], [1.0, -1.0, 0.0, -2.0], [0, None, 0, -2]])
def test_opd_missing_or_misaligned_rows_fall_back_to_zero_without_polluting_metrics(field, invalid):
    parents, inputs = _opd_step()
    inputs[field][0] = invalid
    metrics = {}
    OPDEstimator().estimate(parents, inputs, metrics)
    assert parents[0].segments[0]["token_advantages"] == [0.0, 0.0]
    assert parents[0].segments[1]["token_advantages"] == [-1.0]
    assert metrics == {"opd/teacher_kl": 1.0, "opd/valid_tokens": 1}


def test_zero_opd_coefficient_needs_no_teacher_or_student_inputs():
    parents, _ = _opd_step()
    metrics = {}
    result = OPDEstimator(kl_penalty_coef=0.0).estimate(parents, {}, metrics)
    assert result is parents
    assert [seq["token_advantages"] for seq in parents[0].segments] == [[0.0, 0.0], [0.0]]
    assert metrics == {}


@pytest.mark.parametrize("mismatch", ["missing", "count", "order"])
def test_opd_requires_an_explicit_sequence_aligned_input_contract(mismatch):
    parents, inputs = _opd_step()
    if mismatch == "missing":
        del inputs["student_logprobs"]
    elif mismatch == "count":
        inputs["teacher_logprobs"].pop()
    else:
        inputs["sequences"].reverse()
    with pytest.raises(ValueError, match="OPD"):
        OPDEstimator().estimate(parents, inputs, {})
    assert all("token_advantages" not in sequence for sequence in parents[0].segments)


def test_opd_keeps_packed_buffers_and_untrainable_parents_untouched():
    parents, inputs = _opd_step()
    for sequence in inputs["sequences"]:
        sequence["tokens"] = array("q", sequence["tokens"])
        sequence["loss_mask"] = array("b", sequence["loss_mask"])
    original = parents[0].segments[0]["tokens"]
    parents.append(Trajectory(segments=[], reward=None, status="failed"))
    OPDEstimator().estimate(parents, inputs, {})
    assert parents[0].segments[0]["tokens"] is original
    assert parents[0].segments[0]["token_advantages"] == [-1.0, -2.0]
    assert "token_advantages" not in parents[1]


@pytest.mark.parametrize("config", [{}, {"mode": None}])
def test_disabled_components_need_no_runtime_object(config):
    assert build_estimator(EstimatorConfig.model_validate(config)) is None
    assert build_postprocessor(PostprocessingConfig.model_validate(config)) is None


@pytest.mark.parametrize("config_class", [EstimatorConfig, PostprocessingConfig])
def test_empty_string_is_not_a_disabled_component_mode(config_class):
    with pytest.raises(ValueError):
        config_class(mode="")


def test_factories_fail_explicitly_for_unimplemented_modes():
    # Defend against constructed or future schema variants without dispatching
    # an unrelated estimator. Normal configuration validation rejects these too.
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        build_estimator(EstimatorConfig.model_construct(mode="unknown_estimator", params={}))
    with pytest.raises(ValueError, match="Unknown advantage postprocessor"):
        build_postprocessor(PostprocessingConfig.model_construct(mode="unknown", params={}))


def test_estimator_instances_keep_their_own_parameters_across_steps():
    config = EstimatorConfig(mode="opd", params={"kl_penalty_coef": 0.5})
    first = build_estimator(config)
    config.params["kl_penalty_coef"] = 2.0
    second = build_estimator(config)
    assert isinstance(first, OPDEstimator)
    assert isinstance(second, OPDEstimator)
    for estimator, expected in [
        (first, [-0.5, -1.0]),
        (second, [-2.0, -4.0]),
        (first, [-0.5, -1.0]),
    ]:
        parents, inputs = _opd_step()
        estimator.estimate(parents, inputs, {})
        assert parents[0].segments[0]["token_advantages"] == expected
