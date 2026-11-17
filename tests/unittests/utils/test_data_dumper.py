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

"""Datum dump compatibility for dense lists and heterogeneous metadata."""

import json
import math
import struct
from array import array

import numpy as np
import pytest
import torch

from northrl.utils.data_dumper import DataDumper


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
def test_prepared_dump_preserves_turn_annotations(tmp_path, segmented, dump_format):
    from northrl.mock.mock_rollout_worker import MockRolloutWorker
    from northrl.northrl_types import Trajectory

    annotation = {
        "version": 1,
        "turn_id": "turn-0",
        "response_span": [1, 3],
        "diagnostics": {
            "checked": True,
            "malformed": True,
            "detected": None,
            "invalid": None,
            "reasons": ["invalid_tool_call_structure"],
        },
        "finish_reason": "stop",
        "truncated": False,
        "tools_declared": True,
        "source": "northgate",
    }
    evidence = {
        "turn_annotations": [annotation],
        "base_token_advantages": [1.0, -2.0],
        "token_advantages": [0.0, -4.0],
        "old_log_probs": [0.0, -0.2, -0.3],
    }
    sequence = {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1]}
    trajectory = (
        Trajectory(segments=[{**sequence, **evidence}], reward=1.0)
        if segmented
        else Trajectory(**sequence, extra_fields=evidence, reward=1.0)
    )
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": dump_format,
                    "remote_api_dump_options": {"prepared_trajectories": True},
                }
            }
        }
    )
    dumper.dump_prepared_trajectories(0, [trajectory])
    path = tmp_path / "prepared_trajectories" / f"step_000000.{dump_format}"
    saved = (
        torch.load(path, weights_only=True) if dump_format == "pt" else json.loads(path.read_text())
    )
    worker = MockRolloutWorker.__new__(MockRolloutWorker)
    [restored] = worker._restore_dump_rows(saved["trajectories"], saved["format_version"])
    actual = restored.segments[0] if segmented else restored.extra_fields
    for key, value in evidence.items():
        assert actual[key] == value


class _FloatWithToList(float):
    def tolist(self):
        return [np.int64(7), torch.tensor(-0.0)]


class _IntWithItem(int):
    def item(self):
        return -2.5


class _ListWithToList(list):
    def tolist(self):
        return [np.float32(0.125)]


def _assert_same_value(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_same_value(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_same_value(actual_item, expected_item)
    elif isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            # Equality alone does not distinguish negative zero.
            assert struct.pack("!d", actual) == struct.pack("!d", expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
def test_datums_roundtrip_preserves_schema_and_scalar_conversion(tmp_path, dump_format):
    tensor = torch.tensor([0.25, -0.0], requires_grad=True)
    array = np.array([[3, 4], [5, 6]], dtype=np.int32)
    scalar_values = [0, 2**60, -0.0, True, "label", None, float("inf"), float("nan")]
    datum = {
        "input_tokens": [1, 2, 3],
        "loss_fn_inputs": {
            "target_tokens": (2, 3, 4),
            "logprobs": tensor,
            "advantages": [np.float32(0.1), np.float64(-0.0), torch.tensor(0.5)],
            "loss_mask": [0.0, 1.0, 1.0],
        },
        "metadata": {
            "array": array,
            "scalar_values": scalar_values,
            "nested": [{"value": np.int64(9)}, (np.bool_(True), np.float32(0.125))],
            "custom_float": _FloatWithToList(99),
            "custom_int": _IntWithItem(99),
            # Direct Datum sequences have always been iterated before conversion;
            # nested sequences use _to_serializable's tolist/item precedence.
            "direct_list_subclass": _ListWithToList([1, 2]),
            "nested_list_subclass": [_ListWithToList([1, 2])],
        },
    }
    expected_datum = {
        "input_tokens": [1, 2, 3],
        "loss_fn_inputs": {
            "target_tokens": [2, 3, 4],
            "logprobs": [0.25, -0.0],
            "advantages": [float(np.float32(0.1)), -0.0, 0.5],
            "loss_mask": [0.0, 1.0, 1.0],
        },
        "metadata": {
            "array": [[3, 4], [5, 6]],
            "scalar_values": list(scalar_values),
            "nested": [{"value": 9}, [True, 0.125]],
            "custom_float": [7, -0.0],
            "custom_int": -2.5,
            "direct_list_subclass": [1, 2],
            "nested_list_subclass": [[0.125]],
        },
    }
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": dump_format,
                    "remote_api_dump_options": {"datums": True},
                }
            }
        }
    )
    dumper.dump_datums(3, [datum], extra_info={"loss_fn": "ppo_clip"})
    path = tmp_path / "datums" / f"step_000003.{dump_format}"
    actual = (
        torch.load(path, weights_only=True) if dump_format == "pt" else json.loads(path.read_text())
    )
    _assert_same_value(
        actual,
        {
            "step": 3,
            "num_datums": 1,
            "datums": [expected_datum],
            "extra_info": {"loss_fn": "ppo_clip"},
        },
    )
    assert datum["loss_fn_inputs"]["logprobs"] is tensor
    assert tensor.requires_grad
    assert torch.equal(tensor, torch.tensor([0.25, -0.0]))
    assert datum["metadata"]["array"] is array
    np.testing.assert_array_equal(array, [[3, 4], [5, 6]])
    assert datum["metadata"]["scalar_values"] is scalar_values
    assert isinstance(datum["loss_fn_inputs"]["target_tokens"], tuple)
    assert isinstance(datum["metadata"]["custom_float"], _FloatWithToList)


def test_dense_datum_lists_remain_independent_copies():
    tokens = list(range(128))
    advantages = [0.25, -0.0] * 64
    datum = {
        "input_tokens": tokens,
        "loss_fn_inputs": {"advantages": advantages},
        "metadata": {"nested": [tokens, {"values": advantages}]},
    }

    serialized = DataDumper()._serialize_datum(datum)

    _assert_same_value(serialized, datum)
    assert serialized["input_tokens"] is not tokens
    assert serialized["loss_fn_inputs"]["advantages"] is not advantages
    assert serialized["metadata"]["nested"][0] is not tokens
    assert serialized["metadata"]["nested"][1]["values"] is not advantages
    serialized["input_tokens"][0] = 999
    serialized["loss_fn_inputs"]["advantages"][0] = 999.0
    serialized["metadata"]["nested"][0].append(999)
    assert tokens == list(range(128))
    assert advantages == [0.25, -0.0] * 64


@pytest.mark.parametrize("dump_format", ["pt", "jsonl"])
def test_packed_fields_stay_compact_until_written_and_roundtrip_exactly(tmp_path, dump_format):
    fields = {
        "tokens": array("q", [2**60 + 1, 12]),
        "mask": array("b", [0, 1]),
        "logprobs": array("d", [-0.0, -0.123456789012345]),
        "empty": array("d"),
    }
    dumper = DataDumper(
        {
            "runtime": {
                "debug": {
                    "enable_data_dump": True,
                    "dump_dir": str(tmp_path),
                    "dump_format": dump_format,
                    "remote_api_dump_options": {"datums": True, "prepared_trajectories": True},
                }
            }
        }
    )
    serialized = dumper._serialize_datum(fields)
    for key, value in fields.items():
        if dump_format == "pt":
            assert isinstance(serialized[key], torch.Tensor)
            if value:
                assert serialized[key].data_ptr() == value.buffer_info()[0]
        else:
            assert serialized[key] is value
    dumper.dump_datums(1, [fields])
    path = tmp_path / "datums" / f"step_000001.{dump_format}"
    if dump_format == "pt":
        loaded = torch.load(path, weights_only=True)["datums"][0]
        actual = {key: value.tolist() for key, value in loaded.items()}
    else:
        actual = json.loads(path.read_text())["datums"][0]
    _assert_same_value(actual, {key: value.tolist() for key, value in fields.items()})
    # The round index is JSON even when training dumps use pt.
    dumper.dump_rollout_round(1, 2, [fields])
    index = json.loads((tmp_path / "rollout_rounds" / "step_000001.json").read_text())
    _assert_same_value(index["rollouts"][0], actual)
