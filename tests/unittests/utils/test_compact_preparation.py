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

"""Compact preparation keeps the legacy list path's values and ownership."""

from array import array

import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.logging_utils import log_advantage_distribution_metrics, log_logprob_metrics
from northrl.utils.trajectory_alignment import align_logprobs_to_target, align_loss_mask_to_target


def _trajectory(compact, typecode="d", advantage=-0.1, extra=None, masks=None):
    masks = masks or [0, 0, 1, -2, 2, 1]
    lp = array(typecode, [0.0, float("nan"), -0.0, -0.7, -1.1, float("inf")])
    tokens = array("q", range(len(masks)))
    return Trajectory(
        tokens=tokens if compact else list(tokens),
        loss_mask=array("b", masks) if compact else masks,
        reward=1.0,
        extra_fields={
            "advantage": advantage,
            "old_log_probs": lp if compact else list(lp),
            **(extra or {}),
        },
    )


@pytest.mark.parametrize("typecode", ["f", "d"])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
@pytest.mark.parametrize("value", [-0.1, -0.0, 0.0, 1e-38])
def test_compact_conversion_bitwise_and_owned(typecode, dtype, value):
    advantage = value if dtype is None else torch.tensor(value, dtype=dtype)
    original = _trajectory(True, typecode, advantage)
    legacy = _trajectory(False, typecode, advantage)
    before = [x.tobytes() for x in (original.tokens, original.loss_mask, original["old_log_probs"])]
    kwargs = {"drop_zero_advantage_datums": False}
    actual = convert_trajectories_to_datums([original], **kwargs)[0]
    expected = convert_trajectories_to_datums([legacy], **kwargs)[0]
    assert list(actual["input_tokens"]) == expected["input_tokens"]
    for key, values in actual["loss_fn_inputs"].items():
        assert array("d", values).tobytes() == array("d", expected["loss_fn_inputs"][key]).tobytes()
    for buf, snapshot in zip(
        (original.tokens, original.loss_mask, original["old_log_probs"]), before
    ):
        assert buf.tobytes() == snapshot
        buf.append(0)  # no retained buffer export can prevent resizing
    for values in actual["loss_fn_inputs"].values():
        assert isinstance(values, array)
        values.append(0)


@pytest.mark.parametrize("optional", ["ref", "opd", "mis", "sampling", "zero"])
def test_optional_fields_and_zero_filter_preserve_legacy(optional):
    compact = _trajectory(True, masks=[0, 0, 1, 1, 0, 1])
    legacy = _trajectory(False, masks=[0, 0, 1, 1, 0, 1])
    fields = {
        "ref": {"ref_logprobs": array("d", [0, 0, -0.3, -0.4, -0.1, -0.8])},
        "opd": {"token_advantages": [0.1, 0.2, -0.3]},
        "mis": {"correction_weights": [0, 0, 0.5, 2, 1, 1], "correction_mask": [0, 0, 1, 0, 1, 1]},
        "sampling": {"sampling_mask": [[2, 7], [3, 8], [5, 9]]},
        "zero": {"advantage": 0.0},
    }[optional]
    for key, value in fields.items():
        compact[key] = value
        legacy[key] = list(value) if isinstance(value, array) else value
    outputs = []
    for trajectory in (legacy, compact):
        sources = []
        datums = convert_trajectories_to_datums(
            [trajectory], datum_sources=sources, metadata_builder=lambda _, idx: {"index": idx}
        )
        for datum in datums:
            datum["input_tokens"] = list(datum["input_tokens"])
            datum["loss_fn_inputs"] = {
                key: (value if key == "sampling_mask" else array("d", value).tobytes())
                for key, value in datum["loss_fn_inputs"].items()
            }
        outputs.append((datums, sources))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("typecode", ["f", "d"])
@pytest.mark.parametrize("mask_length", [3, 6, 7])
def test_compact_statistics_exact(typecode, mask_length):
    results = []
    for compact in (False, True):
        trajectory = _trajectory(compact, typecode)
        mask = [0, 0, 1, -2, 2, 1, 1][:mask_length]
        trajectory.loss_mask = array("b", mask) if compact else mask
        metrics = {}
        log_advantage_distribution_metrics([trajectory], metrics)
        log_logprob_metrics([trajectory], metrics)
        results.append(metrics)
    assert results[0] == results[1]


@pytest.mark.parametrize("compact", [False, True])
def test_alignment_checks_and_default_return_contract(compact):
    tokens = array("q", [1, 2, 3])
    for lp in (array("d", [0, 1]), array("d", [1, 2, 3]), [0, None, -1]):
        with pytest.raises(ValueError):
            align_logprobs_to_target(tokens=tokens, logprobs=lp, compact=compact)
    for mask in (array("b", [0, 1]), array("b", [1, 0, 1])):
        with pytest.raises(ValueError):
            align_loss_mask_to_target(tokens=tokens, loss_mask=mask, compact=compact)
    result = align_logprobs_to_target(
        tokens=tokens, logprobs=array("d", [0, -1, -2]), compact=compact
    )
    assert isinstance(result, array if compact else list)
