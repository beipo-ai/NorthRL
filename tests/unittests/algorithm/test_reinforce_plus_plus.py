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

"""Independent formula checks for complete-rollout REINFORCE++ preparation."""

import math
from array import array
from copy import deepcopy

import numpy as np
import pytest

from northrl.algorithm.advantage.reinforce_plus_plus import (
    ReinforcePlusPlusEstimator,
    compute_reinforce_plus_plus_advantages,
)
from northrl.northrl_types import Trajectory
from northrl.utils.trajectory_sequences import iter_training_sequences, iter_trajectory_sequences


def _sequence(mask):
    return {"tokens": list(range(len(mask))), "loss_mask": mask}


def _parent(reward, masks, *, group="group", model="actor", plain=False):
    metadata = {"group_id": group, "model_tag": model}
    if plain:
        return Trajectory(
            tokens=list(range(len(masks[0]))),
            loss_mask=masks[0],
            reward=reward,
            extra_fields=metadata,
        )
    return Trajectory(
        segments=[_sequence(mask) for mask in masks], reward=reward, extra_fields=metadata
    )


def _flatten(parents):
    return [
        value
        for parent in parents
        for sequence in iter_trajectory_sequences(parent)
        for value in sequence["token_advantages"]
    ]


def _whiten(values):
    if len(values) <= 1:
        return [0.0] * len(values)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return [(value - mean) / math.sqrt(variance + 1e-8) for value in values]


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_vanilla_terminal_returns_and_unbiased_global_whitening(gamma):
    parents = [_parent(2, [[0, 1, 1, 1]], plain=True), _parent(4, [[0, 1]])]
    compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=gamma)
    expected = _whiten([2 * gamma**2, 2 * gamma, 2, 4])
    assert _flatten(parents) == pytest.approx(expected, abs=1e-12)
    assert parents[0]["advantage"] == pytest.approx(sum(expected[:3]) / 3)
    assert parents[1]["advantage"] == pytest.approx(expected[3])
    assert math.fsum(_flatten(parents)) == pytest.approx(0, abs=1e-12)


def test_discount_does_not_reset_at_segments_or_advance_on_masked_context():
    merged = [_parent(3, [[0, 1, 1, 1, 1]]), _parent(1, [[0, 1]])]
    segmented = [
        _parent(3, [[0, 0, 1, 0], [0, 1, 0, 1, 1, 0]]),
        _parent(1, [[0, 0, 1, 0]]),
    ]
    for parents in (merged, segmented):
        compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=0.5)
    assert _flatten(segmented) == pytest.approx(_flatten(merged), abs=1e-12)
    assert _flatten(segmented) == pytest.approx(_whiten([0.375, 0.75, 1.5, 3, 1]))
    assert [len(s["token_advantages"]) for s in segmented[0].segments] == [1, 3]


def test_whitening_counts_tokens_globally_instead_of_sequences_or_groups():
    parents = [_parent(1, [[0, 1, 1, 1]], group="a"), _parent(5, [[0, 1]], group="b")]
    compute_reinforce_plus_plus_advantages(parents, baseline=False)
    values = _flatten(parents)
    assert values == pytest.approx(_whiten([1, 1, 1, 5]))
    assert values[0] == pytest.approx(-0.5)
    assert values[-1] == pytest.approx(1.5)
    assert parents[0]["advantage"] + parents[1]["advantage"] != pytest.approx(0)


def test_baseline_is_inclusive_centering_without_rloo_or_group_std_scaling():
    rewards = [0, 2, 0, 2, 7]
    parents = [
        _parent(reward, [[0, 1]], group="two" if index < 2 else "three")
        for index, reward in enumerate(rewards)
    ]
    compute_reinforce_plus_plus_advantages(parents, baseline=True)
    actual = _flatten(parents)
    assert actual == pytest.approx(_whiten([-1, 1, -3, -1, 4]), abs=1e-12)
    assert actual != pytest.approx(_whiten([-2, 2, -4.5, -1.5, 6]))
    std_a = math.sqrt(2)
    std_b = math.sqrt(13)
    assert actual != pytest.approx(
        _whiten([-1 / std_a, 1 / std_a, -3 / std_b, -1 / std_b, 4 / std_b])
    )


def test_baseline_grouping_keeps_models_separate():
    parents = [
        _parent(0, [[0, 1]], model="a"),
        _parent(2, [[0, 1]], model="a"),
        _parent(100, [[0, 1]], model="b"),
        _parent(110, [[0, 1]], model="b"),
    ]
    compute_reinforce_plus_plus_advantages(parents, baseline=True)
    assert _flatten(parents) == pytest.approx(_whiten([-1, 1, -5, 5]))


def test_successful_empty_parent_counts_once_in_outcome_baseline():
    parents = [
        _parent(2, [[0, 1], [0, 1]], group="a"),
        _parent(6, [], group="a"),
        _parent(0, [[0, 1]], group="b"),
        _parent(2, [[0, 1]], group="b"),
    ]
    compute_reinforce_plus_plus_advantages(parents, baseline=True)
    assert _flatten(parents) == pytest.approx(_whiten([-2, -2, -1, 1]))
    assert parents[1]["advantage"] == 0


def test_failed_and_skipped_parents_do_not_enter_baseline_or_whitening():
    parents = [_parent(1, [[0, 1]]), _parent(3, [[0, 1]])]
    failed = _parent(math.nan, [[1, 0]], group="failed")
    failed.status = "failed"
    skipped = _parent(math.nan, [[0, 1]], group="skipped")
    skipped["group_skipped"] = True
    failed["advantage"] = skipped["advantage"] = 42.0
    compute_reinforce_plus_plus_advantages(parents + [failed, skipped], baseline=True)
    assert _flatten(parents) == pytest.approx(_whiten([-1, 1]))
    assert failed["advantage"] == skipped["advantage"] == 42.0


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("masks", [[], [[0]], [[0, 0]], [[0, 1]]])
def test_empty_or_single_token_batch_has_finite_zero_advantages(baseline, masks):
    parents = [_parent(10, masks)]
    compute_reinforce_plus_plus_advantages(parents, baseline=baseline)
    assert _flatten(parents) == [0.0] * sum(sum(mask) for mask in masks)
    assert parents[0]["advantage"] == 0


@pytest.mark.parametrize("baseline", [False, True])
def test_constant_zero_returns_and_long_constant_returns_are_zero(baseline):
    for reward in (0, 0.1):
        parents = [_parent(reward, [[0] + [1] * 100]), _parent(reward, [[0, 1]])]
        compute_reinforce_plus_plus_advantages(parents, baseline=baseline)
        assert _flatten(parents) == [0.0] * 101


def test_singletons_and_ungrouped_parents_have_zero_baseline_score():
    parents = [_parent(7, [[0, 1, 1]], group="singleton"), _parent(3, [[0, 1]])]
    parents[1].extra_fields.pop("group_id")
    parents.extend([_parent(1, [[0, 1]], group="pair"), _parent(3, [[0, 1]], group="pair")])
    compute_reinforce_plus_plus_advantages(parents, baseline=True)
    assert _flatten(parents) == pytest.approx(_whiten([0, 0, 0, -1, 1]))


def test_repeated_preparation_replaces_vectors_and_clears_stale_parent_vector():
    parents = [_parent(3, [[0, 1], [0, 1, 1]]), _parent(1, [[0, 1]], plain=True)]
    parents[0]["token_advantages"] = [999] * 50
    for sequence in parents[0].segments:
        sequence["token_advantages"] = [888] * 10
    compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=0.5)
    assert "token_advantages" not in parents[0].extra_fields
    first = deepcopy(_flatten(parents))
    # The trainer's metadata broadcast must not overwrite the segment vectors.
    list(iter_training_sequences(parents))
    assert _flatten(parents) == first
    parents[0].reward = 6
    parents[0].segments[1]["loss_mask"] = [0, 0, 1]
    compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=0.5)
    assert _flatten(parents) == pytest.approx(_whiten([3, 6, 1]))
    assert [len(sequence["token_advantages"]) for sequence in parents[0].segments] == [1, 1]


@pytest.mark.parametrize("mask", [[1, 0], [0, 0.5], [0, -1], [0, math.nan], [0, "1"]])
def test_invalid_masks_rejected(mask):
    parent = _parent(1, [mask])
    with pytest.raises(ValueError, match="loss_mask"):
        compute_reinforce_plus_plus_advantages([parent], baseline=False)


def test_mask_length_mismatch_rejected_even_for_empty_tokens():
    parent = Trajectory(tokens=[], loss_mask=[0], reward=1)
    with pytest.raises(ValueError, match="token-aligned"):
        compute_reinforce_plus_plus_advantages([parent], baseline=False)


@pytest.mark.parametrize("reward", [math.inf, math.nan, True, "1"])
def test_invalid_rewards_rejected_even_for_successful_empty_parent(reward):
    parent = _parent(reward, [])
    with pytest.raises(ValueError, match="reward"):
        compute_reinforce_plus_plus_advantages([parent], baseline=True)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"gamma": -0.1}, "gamma"),
        ({"gamma": 1.1}, "gamma"),
        ({"gamma": math.nan}, "gamma"),
        ({"baseline": True, "gamma": 0.5}, "requires gamma=1"),
    ],
)
def test_invalid_options_rejected(kwargs, match):
    options = {"baseline": False, **kwargs}
    with pytest.raises(ValueError, match=match):
        compute_reinforce_plus_plus_advantages([], **options)


def _scalar_discount_reference(reward, count, gamma):
    values = [0.0] * count
    running = reward
    for index in reversed(range(count)):
        if index != count - 1:
            running *= gamma
        values[index] = running
    return values


@pytest.mark.parametrize("gamma", [0.0, 1e-300, 0.73, 1 - 1e-12, 1.0])
def test_packed_and_list_returns_match_terminal_reward_discount(gamma):
    source = [
        _parent(3.125, [[0, 0, 1, 0, 1] * 7, [0, 1, 1, 0] * 19]),
        _parent(-2.375, [[0, 1, 0, 1] * 23], plain=True),
    ]
    expected_returns = []
    for parent in source:
        count = sum(sum(sequence["loss_mask"]) for sequence in iter_trajectory_sequences(parent))
        expected_returns.extend(
            parent.reward * gamma**distance for distance in reversed(range(count))
        )
    expected = _whiten(expected_returns)
    for compact in (False, True):
        parents = deepcopy(source)
        sequences = list(iter_training_sequences(parents))
        if compact:
            for sequence in sequences:
                for field, typecode in [("tokens", "q"), ("loss_mask", "b")]:
                    sequence[field] = array(typecode, sequence[field])
        original = [(sequence["tokens"], sequence["loss_mask"]) for sequence in sequences]
        compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=gamma)
        assert _flatten(parents) == pytest.approx(expected, rel=2e-13, abs=2e-13)
        for sequence, (tokens, mask) in zip(sequences, original, strict=True):
            assert sequence["tokens"] is tokens and sequence["loss_mask"] is mask
            assert isinstance(sequence["token_advantages"], array if compact else list)
            if compact:
                assert sequence["token_advantages"].typecode == "d"


@pytest.mark.parametrize("gamma", [1e-300, 0.99, 1 - 1e-12])
def test_long_discounted_stream_is_stable_across_segment_boundaries(gamma):
    parents = [_parent(1.25, [[0] + [1] * 12000, [0] * 37 + [1] * 12000]), _parent(3, [[0, 1]])]
    for sequence in iter_training_sequences(parents):
        sequence["tokens"] = array("q", sequence["tokens"])
        sequence["loss_mask"] = array("b", sequence["loss_mask"])
    raw = _scalar_discount_reference(1.25, 24000, gamma) + [3.0]
    compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=gamma)
    assert _flatten(parents) == pytest.approx(_whiten(raw), abs=2e-12, rel=2e-12)


def test_token_free_successful_parent_keeps_empty_advantage_vector():
    parent = Trajectory(reward=1)
    compute_reinforce_plus_plus_advantages([parent], baseline=False)
    assert parent["advantage"] == 0
    assert parent["token_advantages"] == []


@pytest.mark.parametrize("gamma", [0.0, -0.0])
def test_zero_discount_preserves_signed_zero_after_whitening(gamma):
    parents = [_parent(-1, [[0, 1, 1]]), _parent(1, [[0, 1, 1]])]
    for parent in parents:
        sequence = parent.segments[0]
        sequence.update(
            tokens=array("q", [0, 1, 2]),
            loss_mask=array("b", [0, 1, 1]),
        )
    compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=gamma)
    values = _flatten(parents)
    expected_signs = [-1, -1, 1, 1] if math.copysign(1, gamma) > 0 else [1, -1, -1, 1]
    assert [math.copysign(1, value) for value in values] == expected_signs
    assert values == pytest.approx([0, -1.2247448622060026, 0, 1.2247448622060026])


@pytest.mark.parametrize("container", [list, tuple])
def test_legacy_mask_scalar_item_validation_precedes_numpy_coercion(container):
    class MaskScalar(float):
        def item(self):
            return 2

    parent = _parent(1, [container([0, MaskScalar(1)])])
    with pytest.raises(ValueError, match="loss_mask"):
        compute_reinforce_plus_plus_advantages([parent], baseline=False)


@pytest.mark.parametrize("baseline", [False, True])
def test_estimator_needs_no_probability_inputs(baseline):
    parents = [_parent(1, [[0, 1, 1]]), _parent(3, [[0, 1]])]
    metrics = {}
    result = ReinforcePlusPlusEstimator(baseline=baseline).estimate(parents, {}, metrics)
    assert result is parents
    expected_returns = [-1, -1, 1] if baseline else [1, 1, 3]
    assert _flatten(parents) == pytest.approx(_whiten(expected_returns), abs=1e-12)
    assert metrics


@pytest.mark.parametrize("baseline", [False, True])
def test_formula_does_not_read_or_change_probability_fields(baseline):
    parents = [_parent(1, [[0, 1, 1]]), _parent(3, [[0, 1]])]
    sequences = list(iter_training_sequences(parents))
    # Probability validation belongs to the loss path; these fields are not
    # inputs to the reward-only advantage formula, even when they are present.
    for sequence in sequences:
        sequence["old_log_probs"] = object()
        sequence["ref_logprobs"] = object()
    original = [(sequence["old_log_probs"], sequence["ref_logprobs"]) for sequence in sequences]
    compute_reinforce_plus_plus_advantages(parents, baseline=baseline)
    expected_returns = [-1, -1, 1] if baseline else [1, 1, 3]
    assert _flatten(parents) == pytest.approx(_whiten(expected_returns), abs=1e-12)
    for sequence, (old, reference) in zip(sequences, original, strict=True):
        assert sequence["old_log_probs"] is old
        assert sequence["ref_logprobs"] is reference


@pytest.mark.parametrize("baseline", [False, True])
def test_nonfinite_whitening_statistics_fail_before_writing_advantages(baseline):
    parents = [_parent(1e308, [[0, 1]]), _parent(-1e308, [[0, 1]])]
    with pytest.raises(ValueError, match="whitening statistics must remain finite"):
        compute_reinforce_plus_plus_advantages(parents, baseline=baseline)
    assert all("token_advantages" not in sequence for sequence in iter_training_sequences(parents))


def test_overflowing_centered_reward_is_rejected():
    parents = [_parent(reward, [[0, 1]]) for reward in [1.7e308, -1.7e308, -1.7e308]]
    with pytest.raises(ValueError, match="centered reward must be a finite real scalar"):
        compute_reinforce_plus_plus_advantages(parents, baseline=True)


def test_tiny_gamma_remains_valid_with_strict_numpy_errors():
    parents = [_parent(1, [[0, 1, 1, 1]]), _parent(2, [[0, 1]])]
    with np.errstate(all="raise"):
        compute_reinforce_plus_plus_advantages(parents, baseline=False, gamma=1e-300)
    assert _flatten(parents) == pytest.approx(_whiten([0, 1e-300, 1, 2]), abs=1e-12)
