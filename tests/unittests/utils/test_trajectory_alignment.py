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

from northrl.northrl_types import Trajectory
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.trajectory_alignment import align_response_sampling_mask_to_target


def _make_traj(
    *,
    tokens=None,
    loss_mask=None,
    old_log_probs=None,
    sampling_mask=None,
) -> Trajectory:
    if tokens is None:
        tokens = [10, 20, 30, 40]
    if loss_mask is None:
        loss_mask = [0, 0, 1, 1]
    if old_log_probs is None:
        old_log_probs = [0.0, -0.1, -0.2, -0.3]
    extra_fields = {
        "advantage": 1.0,
        "old_log_probs": old_log_probs,
    }
    if sampling_mask is not None:
        extra_fields["sampling_mask"] = sampling_mask
    return Trajectory(
        tokens=tokens,
        loss_mask=loss_mask,
        reward=1.0,
        is_val=False,
        extra_fields=extra_fields,
    )


def test_response_sampling_mask_expands_to_target_alignment():
    target_mask = align_response_sampling_mask_to_target(
        tokens=[10, 20, 30, 40],
        loss_mask=[0, 0, 1, 1],
        sampling_mask=[[30, 31], [40, 41]],
    )

    assert target_mask == [[20], [30, 31], [40, 41]]


def test_convert_trajectories_uses_explicit_token_and_target_alignment():
    traj = _make_traj(sampling_mask=[[30, 31], [40, 41]])

    [datum] = convert_trajectories_to_datums([traj])

    assert datum["input_tokens"] == [10, 20, 30]
    assert datum["loss_fn_inputs"]["target_tokens"] == [20, 30, 40]
    assert datum["loss_fn_inputs"]["loss_mask"] == [0.0, 1.0, 1.0]
    assert datum["loss_fn_inputs"]["logprobs"] == [0.0, -0.2, -0.3]
    assert datum["loss_fn_inputs"]["advantages"] == [0.0, 1.0, 1.0]
    assert datum["loss_fn_inputs"]["sampling_mask"] == [[20], [30, 31], [40, 41]]


def test_convert_trajectories_rejects_target_length_old_logprobs():
    traj = _make_traj(old_log_probs=[-0.1, -0.2, -0.3])

    with pytest.raises(ValueError, match="old_log_probs must be token-aligned"):
        convert_trajectories_to_datums([traj])


def test_convert_trajectories_rejects_target_aligned_sampling_mask():
    traj = _make_traj(sampling_mask=[[20], [30, 31], [40, 41]])

    with pytest.raises(ValueError, match="sampling_mask must be response-aligned"):
        convert_trajectories_to_datums([traj])


def test_response_sampling_mask_must_contain_sampled_token():
    with pytest.raises(ValueError, match="must contain the sampled response token"):
        align_response_sampling_mask_to_target(
            tokens=[10, 20, 30],
            loss_mask=[0, 0, 1],
            sampling_mask=[[31, 32]],
        )


def _make_multiturn_correction_traj() -> Trajectory:
    trajectory = _make_traj(
        tokens=[10, 11, 20, 21, 30, 31, 40, 2],
        loss_mask=[0, 0, 1, 1, 0, 0, 1, 1],
        old_log_probs=[0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7],
        sampling_mask=[[20, 22], [21, 23], [40, 42], [2, 3]],
    )
    trajectory.extra_fields.update(
        advantage=2.0,
        ref_logprobs=[0.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7],
        correction_weights=[0.0, 9.0, 0.5, 2.0, 7.0, 8.0, 3.0, 0.25],
        correction_mask=[0, 1, 1, 0, 1, 1, 0, 1],
    )
    return trajectory


def test_corrections_preserve_multiturn_alignment_and_eos_candidates():
    trajectory = _make_multiturn_correction_traj()
    original = deepcopy(trajectory)
    uncorrected = deepcopy(trajectory)
    del uncorrected.extra_fields["correction_weights"]
    del uncorrected.extra_fields["correction_mask"]
    [baseline] = convert_trajectories_to_datums([uncorrected])

    [datum] = convert_trajectories_to_datums([trajectory], drop_zero_advantage_datums=False)
    fields = datum["loss_fn_inputs"]

    assert fields["advantages"] == [0.0, 1.0, 4.0, 0.0, 0.0, 6.0, 0.5]
    assert fields["loss_mask"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert fields["sampling_mask"] == [[11], [20, 22], [21, 23], [30], [31], [40, 42], [2, 3]]
    assert datum["input_tokens"] == baseline["input_tokens"]
    for name in ("target_tokens", "logprobs", "ref_logprobs", "sampling_mask"):
        assert fields[name] == baseline["loss_fn_inputs"][name]
    assert fields["logprobs"] == [0.0, -0.2, -0.3, 0.0, 0.0, -0.6, -0.7]
    assert fields["ref_logprobs"] == [0.0, -1.2, -1.3, 0.0, 0.0, -1.6, -1.7]
    assert "correction_weights" not in fields
    assert "correction_mask" not in fields
    assert trajectory == original


def test_token_advantages_align_before_correction_mask_rejects_tokens():
    trajectory = _make_multiturn_correction_traj()
    trajectory.extra_fields["token_advantages"] = [1.0, -2.0, 3.0, 4.0]

    [datum] = convert_trajectories_to_datums([trajectory], drop_zero_advantage_datums=False)

    assert datum["loss_fn_inputs"]["advantages"] == [0.0, 0.5, -4.0, 0.0, 0.0, 9.0, 1.0]
    assert datum["loss_fn_inputs"]["loss_mask"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("advantage", "weights", "correction_mask", "expected_advantages", "expected_mask"),
    [
        pytest.param(
            0.0,
            [0.0, 1.0, 2.0, 3.0],
            [0, 1, 1, 1],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            id="zero-advantage",
        ),
        pytest.param(
            1.0,
            [0.0, 1.0, 2.0, 3.0],
            [0, 0, 0, 0],
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            id="all-rejected",
        ),
        pytest.param(
            1.0,
            [0.0, 0.0, 0.0, 0.0],
            [0, 1, 1, 1],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            id="zero-weights",
        ),
    ],
)
def test_grpo_retains_zero_advantages_only_with_active_loss_mask(
    advantage, weights, correction_mask, expected_advantages, expected_mask
):
    trajectory = _make_traj(sampling_mask=[[30, 31], [40, 41]])
    trajectory.extra_fields.update(
        advantage=advantage,
        correction_weights=weights,
        correction_mask=correction_mask,
    )

    datums = convert_trajectories_to_datums([trajectory], drop_zero_advantage_datums=False)
    if not any(expected_mask):
        assert datums == []
        return
    [datum] = datums

    assert datum["loss_fn_inputs"]["advantages"] == expected_advantages
    assert datum["loss_fn_inputs"]["loss_mask"] == expected_mask
    assert datum["loss_fn_inputs"]["sampling_mask"] == [[20], [30, 31], [40, 41]]


def test_identity_correction_preserves_uncorrected_datum():
    trajectory = _make_traj(sampling_mask=[[30, 31], [40, 41]])
    baseline = convert_trajectories_to_datums([trajectory])
    trajectory.extra_fields.update(
        correction_weights=[0.0, 1.0, 1.0, 1.0],
        correction_mask=[0, 1, 1, 1],
    )

    assert convert_trajectories_to_datums([trajectory]) == baseline


@pytest.mark.parametrize("missing_field", ["correction_weights", "correction_mask"])
def test_correction_fields_must_be_paired(missing_field):
    trajectory = _make_traj()
    trajectory.extra_fields.update(
        correction_weights=[0.0, 1.0, 1.0, 1.0],
        correction_mask=[0, 1, 1, 1],
    )
    del trajectory.extra_fields[missing_field]

    with pytest.raises(ValueError, match="must be provided together"):
        convert_trajectories_to_datums([trajectory])


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        pytest.param(
            "correction_weights", [1.0, 1.0, 1.0], "must be token-aligned", id="short-weights"
        ),
        pytest.param("correction_mask", [1, 1, 1], "must be token-aligned", id="short-mask"),
        pytest.param("correction_weights", [0.0] * 5, "must be token-aligned", id="long-weights"),
        pytest.param("correction_mask", [0] * 5, "must be token-aligned", id="long-mask"),
        pytest.param("correction_weights", [1.0] * 4, "first token", id="weights-placeholder"),
        pytest.param("correction_mask", [1] * 4, "first token", id="mask-placeholder"),
        pytest.param(
            "correction_weights", [0.0, 1.0, -0.1, 1.0], "finite nonnegative", id="negative-weight"
        ),
        pytest.param(
            "correction_weights",
            [0.0, 1.0, float("nan"), 1.0],
            "finite nonnegative",
            id="nan-weight",
        ),
        pytest.param(
            "correction_weights",
            [0.0, 1.0, float("inf"), 1.0],
            "finite nonnegative",
            id="infinite-weight",
        ),
        pytest.param(
            "correction_weights", [0.0, 1.0, "1", 1.0], "finite nonnegative", id="string-weight"
        ),
        pytest.param(
            "correction_weights", [0.0, 1.0, True, 1.0], "finite nonnegative", id="boolean-weight"
        ),
        pytest.param("correction_mask", [0, 1, 2, 1], "only 0 or 1", id="large-mask"),
        pytest.param("correction_mask", [0, 1, -1, 1], "only 0 or 1", id="negative-mask"),
        pytest.param("correction_mask", [0, 1, 0.5, 1], "only 0 or 1", id="fractional-mask"),
        pytest.param("correction_mask", [0, 1, float("nan"), 1], "only 0 or 1", id="nan-mask"),
        pytest.param("correction_mask", [0, 1, "1", 1], "only 0 or 1", id="string-mask"),
        pytest.param(
            "correction_weights", None, "token-aligned sequence", id="missing-weight-values"
        ),
        pytest.param("correction_mask", None, "token-aligned sequence", id="missing-mask-values"),
        pytest.param("correction_weights", 1.0, "token-aligned sequence", id="scalar-weights"),
        pytest.param("correction_mask", "0111", "token-aligned sequence", id="scalar-mask"),
    ],
)
def test_invalid_correction_fields_are_rejected_before_datum_filtering(field, values, message):
    trajectory = _make_traj()
    trajectory.extra_fields.update(
        advantage=0.0,
        correction_weights=[0.0, 1.0, 1.0, 1.0],
        correction_mask=[0, 1, 1, 1],
    )
    trajectory.extra_fields[field] = values

    with pytest.raises(ValueError, match=message):
        convert_trajectories_to_datums([trajectory])


def test_masked_nonfinite_logprobs_are_neutralized_before_correction():
    trajectory = _make_multiturn_correction_traj()
    trajectory.extra_fields["old_log_probs"] = [
        0.0,
        float("nan"),
        -0.2,
        -0.3,
        float("inf"),
        float("-inf"),
        -0.6,
        -0.7,
    ]
    trajectory.extra_fields["ref_logprobs"] = [
        0.0,
        float("inf"),
        -1.2,
        -1.3,
        float("nan"),
        float("-inf"),
        -1.6,
        -1.7,
    ]

    [datum] = convert_trajectories_to_datums([trajectory], drop_zero_advantage_datums=False)

    assert datum["loss_fn_inputs"]["logprobs"] == [0.0, -0.2, -0.3, 0.0, 0.0, -0.6, -0.7]
    assert datum["loss_fn_inputs"]["ref_logprobs"] == [0.0, -1.2, -1.3, 0.0, 0.0, -1.6, -1.7]
    assert datum["loss_fn_inputs"]["advantages"] == [0.0, 1.0, 4.0, 0.0, 0.0, 6.0, 0.5]
    assert datum["loss_fn_inputs"]["loss_mask"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
