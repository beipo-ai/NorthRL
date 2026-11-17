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

"""Bitwise parity and ownership for compact normalization and mask scans."""

import math
import struct
from array import array
from types import SimpleNamespace

import numpy as np
import pytest

from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.finetune_service_utils import _mask_target_values
from northrl.utils.step_metrics import (
    datum_response_token_count,
    log_step_metrics,
    log_trainable_metrics,
)


def normalize(datums, mode, counts=None):
    trainer = SimpleNamespace(
        _loss_fn="ppo_clip",
        _train_step=0,
        _algorithm_config=AlgorithmConfig(
            type="grpo",
            optim={"lr": 1e-6},
            entropy_coeff=0,
            loss={"mode": "ppo_clip", "reduction": {"mode": mode}},
        ),
    )
    return RemoteApiTrainer._normalize_advantages_for_sum(trainer, datums, token_counts=counts)


def datum(values, mask, denominator=7):
    return {
        "loss_fn_inputs": {"advantages": values, "loss_mask": mask},
        "_prompt_mean_denominator": denominator,
    }


VALUES = [
    0.0,
    -0.0,
    5e-324,
    -5e-324,
    1e-300,
    -1e-300,
    1e308,
    -1e308,
    1.0,
    -0.1,
    float("inf"),
    -float("inf"),
    float("nan"),
]
# Include noncanonical quiet/signaling NaN payloads.
VALUES += [
    struct.unpack("=d", struct.pack("=Q", bits))[0]
    for bits in (0x7FF8000000000001, 0xFFF8000000000003, 0x7FF0000000000001)
]


@pytest.mark.parametrize("mode", ["token_mean", "datum_mean", "prompt_mean"])
@pytest.mark.parametrize("reuse_counts", [False, True])
def test_normalization_exact_owned_duplicates(mode, reuse_counts):
    values = array("d", VALUES)
    mask = array("b", [1] * len(values))
    item = datum(values, mask)
    inputs = [item, item, datum(array("d", [1.0, -0.0]), array("b", [0, 0]), 3)]
    counts = [len(values), len(values), 0]
    denominators = (
        [7, 7, 3]
        if mode == "prompt_mean"
        else (
            [2 * len(values), 2 * len(values), 1] if mode == "datum_mean" else [2 * len(values)] * 3
        )
    )
    before = [d["loss_fn_inputs"]["advantages"].tobytes() for d in inputs]
    with np.errstate(all="raise"):
        output = normalize(inputs, mode, counts if reuse_counts else None)
    for i, (out, den) in enumerate(zip(output, denominators)):
        actual = out["loss_fn_inputs"]["advantages"]
        expected = array("d", (v / den for v in inputs[i]["loss_fn_inputs"]["advantages"]))
        assert type(actual) is array and actual.typecode == "d"
        assert actual.tobytes() == expected.tobytes()
        assert len(actual) == len(expected)
        assert actual is not inputs[i]["loss_fn_inputs"]["advantages"]
        actual.append(0)  # no retained NumPy export
    output[0]["loss_fn_inputs"]["advantages"][0] = 99
    assert output[1]["loss_fn_inputs"]["advantages"][0] != 99
    assert [d["loss_fn_inputs"]["advantages"].tobytes() for d in inputs] == before


@pytest.mark.parametrize("factory", [list, lambda x: array("f", x), lambda x: array("d", x)])
def test_empty_and_fallback(factory):
    for values in ([], [0.0, -0.0, 1e-30, -0.1]):
        original = factory(values)
        result = normalize([datum(original, [1] * len(values))], "token_mean")[0]
        expected = [v / max(len(values), 1) for v in original]
        actual = result["loss_fn_inputs"]["advantages"]
        assert type(actual) is (list if type(original) is list else array)
        assert array("d", actual).tobytes() == array("d", expected).tobytes()
    assert normalize([], "token_mean") == []


@pytest.mark.parametrize("denominator", [2**53 + 1, 2**100, 10**400])
def test_large_denominator_fallback(denominator):
    item = datum(array("d", [1.0]), [1], denominator)
    if denominator > 10**308:
        with pytest.raises(OverflowError):
            normalize([item], "prompt_mean")
    else:
        actual = normalize([item], "prompt_mean")[0]["loss_fn_inputs"]["advantages"]
        assert actual.tobytes() == array("d", [1.0 / denominator]).tobytes()


def test_validation_unchanged():
    item = datum(array("d", [1, 2]), [1])  # normalization does not impose mask length
    assert list(normalize([item], "token_mean")[0]["loss_fn_inputs"]["advantages"]) == [1, 2]
    with pytest.raises(ValueError, match="final response-token"):
        normalize([item], "token_mean", [])
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match="positive complete-group"):
            normalize([datum(array("d", [1]), [1], bad)], "prompt_mean")
    with pytest.raises(TypeError):
        normalize([datum([None], [1])], "token_mean")
    assert normalize([item], "sum")[0] is item


@pytest.mark.parametrize("typecode", list("bBhHiIlLqQfd"))
def test_counts_and_duplicate_effective_masks(typecode):
    a = array(typecode, [0, 1, 2, 0])
    b = array(typecode, [0, 3, 1, 0])
    parent = Trajectory(tokens=[0, 1, 2, 3], loss_mask=[0, 1, 2, 0], reward=1)
    items = [datum([], a), datum([], b)]
    metrics = {}
    assert log_trainable_metrics([parent], items, metrics, datum_sources=[(0, 0)] * 2) == [2, 2]
    assert metrics["batch/trainable_response_tokens_total"] == 2
    b[0] = 1
    with pytest.raises(ValueError, match="inconsistent effective masks"):
        log_trainable_metrics([parent], items, {}, datum_sources=[(0, 0)] * 2)
    b.pop()
    with pytest.raises(ValueError, match="inconsistent effective masks"):
        log_trainable_metrics([parent], items, {}, datum_sources=[(0, 0)] * 2)


def test_nonbinary_nan_masks_metrics_and_no_cached_counts():
    mask = array("d", [0.0, -0.0, -2, float("nan"), float("inf"), -float("inf"), 2])
    item = datum([], mask)
    parent = Trajectory(tokens=list(range(len(mask))), loss_mask=mask, reward=1)
    compact_metrics, list_metrics = {}, {}
    log_step_metrics([parent], compact_metrics)
    parent.loss_mask = list(mask)
    log_step_metrics([parent], list_metrics)
    assert compact_metrics == list_metrics
    assert datum_response_token_count(item) == 2
    mask[-1] = -1
    assert datum_response_token_count(item) == 1
    other = array("d", [0, 0, -4, -5, 9, 0, -1])
    assert log_trainable_metrics(
        [parent], [item, datum([], other)], {}, datum_sources=[(0, 0)] * 2
    ) == [1, 1]
    with pytest.raises(TypeError):
        datum_response_token_count(datum([], [None]))
    assert datum_response_token_count({"kind": "sample_ref"}) == 0
    with pytest.raises(ValueError, match="Unknown Datum source"):
        log_trainable_metrics([parent], [item], {}, datum_sources=[(1, 0)])
    with pytest.raises(ValueError, match="original parent"):
        log_trainable_metrics([parent], [item], {}, datum_sources=[])
    parent.status = "failed"
    with pytest.raises(ValueError, match="failed/skipped"):
        log_trainable_metrics([parent], [item], {}, datum_sources=[(0, 0)])


@pytest.mark.parametrize("length", [0, 3, len(VALUES), len(VALUES) + 2])
def test_mask_output_exact_owned_and_truncated(length):
    values = array("d", VALUES)
    mask = array("b", ([0, 1, -2, 2] * len(VALUES))[:length])
    before = values.tobytes(), mask.tobytes()
    with np.errstate(all="ignore"):
        result = _mask_target_values(values, mask)
    expected = _mask_target_values(list(values), list(mask))
    assert type(result) is array and result.typecode == "d"
    assert result.tobytes() == array("d", expected).tobytes()
    assert (values.tobytes(), mask.tobytes()) == before
    for value, active in zip(result, mask):
        if not active:
            assert value == 0 and math.copysign(1, value) == 1
    result.append(0)
    values.append(0)
    mask.append(0)


def test_random_finite_division_is_bitwise_equal():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2**64 - 1, 4096, dtype=np.uint64)
    values = array("d", bits.view(np.float64)[np.isfinite(bits.view(np.float64))])
    for denominator in (1, 3, 7, 46733328, 69111736, 2**53):
        result = normalize([datum(values, [], denominator)], "prompt_mean", [0])[0]
        expected = array("d", (v / denominator for v in values))
        assert result["loss_fn_inputs"]["advantages"].tobytes() == expected.tobytes()


def test_array_subclasses_keep_iteration_fallback():
    class CustomArray(array):
        def __iter__(self):
            yield from (2.0, 4.0)

    values = CustomArray("d", [10, 20])
    item = datum(values, CustomArray("d", [-1, -1]))
    assert datum_response_token_count(item) == 2
    result = normalize([item], "token_mean")[0]["loss_fn_inputs"]["advantages"]
    assert result == array("d", [1, 2])
