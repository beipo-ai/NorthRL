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

"""Exact differential checks for the isolated aligned-mask candidate."""

import copy
import random
import struct

import pytest
import torch

from northrl.northrl_types import Trajectory
from northrl.utils import finetune_service_utils as conversion
from tests.unittests.utils.preparation_reference import use_legacy_preparation

build_aligned_masked_arrays = conversion._build_aligned_masked_arrays


def assert_exact(actual, expected):
    if isinstance(expected, float):
        assert type(actual) is float
        assert struct.pack("!d", actual) == struct.pack("!d", expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for x, y in zip(actual, expected, strict=True):
            assert_exact(x, y)
    else:
        assert actual == expected


@pytest.mark.parametrize("size", [0, 1, 1023, 1024, 4096])
@pytest.mark.parametrize("layout", ["prefix", "zeros", "ones", "holes", "weighted"])
def test_ieee_bit_patterns_and_mask_layouts_are_identical(size, layout):
    rng = random.Random(20261019)
    special = [
        0x0000000000000000,  # +0
        0x8000000000000000,  # -0
        0x0000000000000001,  # smallest positive subnormal
        0x7FEFFFFFFFFFFFFF,  # largest finite value
        0x7FF0000000000000,  # +infinity
        0xFFF0000000000000,  # -infinity
        0x7FF8000000000001,  # quiet NaN with payload
        0xFFF8000000000002,  # negative quiet NaN
        0x7FF0000000000001,  # signaling NaN; multiplication must still quiet it
    ]
    bits = (special * (size // len(special) + 1))[:size]
    if size > 32:
        bits[32:] = [rng.getrandbits(64) for _ in bits[32:]]
    values = [struct.unpack("!d", struct.pack("!Q", value))[0] for value in bits]
    if layout == "prefix":
        mask = [0] * (size // 3) + [1] * (size - size // 3)
    elif layout == "zeros":
        mask = [0] * size
    elif layout == "ones":
        mask = [1] * size
    elif layout == "holes":
        mask = [index % 2 for index in range(size)]
    else:
        mask = [rng.choice([0, 1, -1, 2, 3, 2**53 - 1, 2**53 + 1]) for _ in values]
    expected = conversion._mask_target_values(values, mask), [float(m) for m in mask]
    original_values = list(values)
    original_mask = list(mask)
    assert_exact(build_aligned_masked_arrays(values, mask), expected)
    assert_exact(values, original_values)
    assert mask == original_mask


def trajectory():
    length = 1537
    return Trajectory(
        tokens=list(range(length)),
        loss_mask=[0] * 12 + [1] * (length - 12),
        reward=1.0,
        extra_fields={
            "advantage": 0.75,
            "old_log_probs": [0.0] + [-index / 100 for index in range(1, length)],
            "group_id": "comparison",
        },
    )


@pytest.mark.parametrize(
    "mode",
    ["grpo", "zero_advantage", "opd", "reference", "mis", "prompt_mean", "holes", "weighted"],
)
def test_complete_converter_outputs_are_identical(mode):
    item = trajectory()
    kwargs = {"drop_zero_advantage_datums": False}
    if mode == "zero_advantage":
        item["advantage"] = 0.0
    elif mode == "opd":
        item["token_advantages"] = [0.5] * sum(item.loss_mask)
    elif mode == "reference":
        item["ref_logprobs"] = [None] + [-0.5] * (len(item.tokens) - 1)
    elif mode == "mis":
        item["correction_weights"] = [0.0] + [0.25] * (len(item.tokens) - 1)
        item["correction_mask"] = [0] + [
            int(index % 3 != 0) for index in range(1, len(item.tokens))
        ]
    elif mode == "prompt_mean":
        kwargs["prompt_mean"] = True
    elif mode == "holes":
        item.loss_mask[100] = 0
    elif mode == "weighted":
        item.loss_mask[100] = 2
    kwargs["metadata_builder"] = lambda traj, index: {"group_id": traj["group_id"], "index": index}
    original_sources = []
    with use_legacy_preparation():
        expected = conversion.convert_trajectories_to_datums(
            [copy.deepcopy(item)], datum_sources=original_sources, **kwargs
        )
    candidate_sources = []
    actual = conversion.convert_trajectories_to_datums(
        [copy.deepcopy(item)], datum_sources=candidate_sources, **kwargs
    )
    assert_exact(actual, expected)
    assert candidate_sources == original_sources


def test_256k_converter_matches_main_array_path_bit_for_bit():
    length = 262145
    item = Trajectory(
        tokens=list(range(length)),
        loss_mask=[0] * 4096 + [1] * (length - 4096),
        reward=1.0,
        extra_fields={
            "advantage": torch.tensor(-1 / 3, dtype=torch.float32),
            "old_log_probs": [0.0] + [-0.5] * (length - 1),
        },
    )
    with use_legacy_preparation():
        expected = conversion.convert_trajectories_to_datums([item])
    actual = conversion.convert_trajectories_to_datums([item])
    assert_exact(actual, expected)


@pytest.mark.parametrize(
    "kind",
    [
        "length_lp",
        "head_lp",
        "masked_none",
        "masked_invalid",
        "length_mask",
        "head_mask",
        "huge_mask",
        "bad_reference",
    ],
)
def test_malformed_input_exception_type_and_message_are_unchanged(kind):
    item = trajectory()
    if kind == "length_lp":
        item["old_log_probs"] = [0.0]
    elif kind == "head_lp":
        item["old_log_probs"][0] = 1.0
    elif kind == "masked_none":
        item["old_log_probs"][1] = None
    elif kind == "masked_invalid":
        item["old_log_probs"][1] = "not-a-number"
    elif kind == "length_mask":
        item.loss_mask.pop()
    elif kind == "head_mask":
        item.loss_mask[0] = 1
    elif kind == "huge_mask":
        item.loss_mask[-1] = 1 << 2048
    else:
        item["ref_logprobs"] = [None, None] + [-0.5] * (len(item.tokens) - 2)
    with use_legacy_preparation():
        with pytest.raises(Exception) as original:
            conversion.convert_trajectories_to_datums([copy.deepcopy(item)])
    with pytest.raises(type(original.value)) as candidate:
        conversion.convert_trajectories_to_datums([copy.deepcopy(item)])
    assert str(candidate.value) == str(original.value)


def test_patch_restored_after_exception():
    original = conversion._build_aligned_masked_arrays
    with pytest.raises(RuntimeError, match="sentinel"):
        with use_legacy_preparation():
            assert conversion._build_aligned_masked_arrays is not original
            raise RuntimeError("sentinel")
    assert conversion._build_aligned_masked_arrays is original


def test_verified_prefix_uses_fast_path(monkeypatch):
    def unexpected_fallback(*args):
        raise AssertionError("A verified long prefix mask should use the candidate")

    monkeypatch.setattr(conversion, "_mask_target_values", unexpected_fallback)
    masked, float_mask = build_aligned_masked_arrays([-0.5] * 2048, [0] * 128 + [1] * 1920)
    assert masked == [0.0] * 128 + [-0.5] * 1920
    assert float_mask == [0.0] * 128 + [1.0] * 1920


@pytest.mark.parametrize("layout", ["short", "holes", "weighted"])
def test_noncanonical_layouts_call_original_fallback(monkeypatch, layout):
    calls = []
    original = conversion._mask_target_values

    def observed(values, mask):
        calls.append(True)
        return original(values, mask)

    monkeypatch.setattr(conversion, "_mask_target_values", observed)
    mask = [0] * 128 + [1] * 1920
    if layout == "short":
        mask = mask[:100]
    elif layout == "holes":
        mask[1024] = 0
    else:
        mask[1024] = 2
    build_aligned_masked_arrays([-0.5] * len(mask), mask)
    assert calls == [True]
