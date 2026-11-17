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

"""Prepare dense loss arrays without the SDK's repeated Python shape scan."""

from array import array
from collections.abc import Mapping
from typing import Any

import torch

_DENSE_LOSS_INPUT_KEYS = (
    "target_tokens",
    "logprobs",
    "advantages",
    "loss_mask",
    "ref_logprobs",
)
_BUFFER_DTYPES = {"q": (8, torch.int64), "d": (8, torch.float64), "b": (1, torch.int8)}


def _copy_dense_array(value: array, key: str, dtype: torch.dtype) -> torch.Tensor:
    source = _BUFFER_DTYPES.get(value.typecode) if type(value) is array else None
    if (
        source is not None
        and value.itemsize == source[0]
        and len(value) > 0
        and (
            (key == "target_tokens" and value.typecode == "q")
            or (
                key != "target_tokens"
                and value.typecode in ("d", "b")
                and dtype in (torch.float32, torch.float64)
            )
        )
    ):
        # get_default_device() does not reflect every torch.device(...) context.
        device = torch.empty(0).device
        if device.type == "cpu":
            view = torch.frombuffer(value, dtype=source[1])
            result = torch.empty(len(value), dtype=dtype, device=device)
            # Bound copies to avoid waking a large thread pool for each segment.
            for start in range(0, len(value), 16384):
                result[start : start + 16384].copy_(view[start : start + 16384])
            return result
    return torch.tensor(value, dtype=dtype)


def prepare_dense_loss_inputs(loss_fn_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert known flat numeric inputs once, retaining SDK dtype inference.

    Datum accepts existing tensors directly. Ordinary dense lists otherwise
    undergo a Python element-by-element jagged-array scan before the same
    tensor conversion. Driver buffers use the same loss dtypes as their list
    equivalents. Leave custom fields and existing NumPy/tensor objects alone.
    Let Datum handle nested inputs and conversion failures to preserve its
    jagged-input behavior and validation errors. This never mutates the
    caller's input mapping or lists.
    """
    prepared = dict(loss_fn_inputs)
    for key in _DENSE_LOSS_INPUT_KEYS:
        value = prepared.get(key)
        if isinstance(value, array):
            # Buffers retain compact storage until this one request is built.
            # Match the historical list inference: integer targets and default
            # floating dtype for the loss arrays, including the binary mask.
            prepared[key] = _copy_dense_array(
                value,
                key,
                dtype=torch.int64 if key == "target_tokens" else torch.get_default_dtype(),
            )
            continue
        if not isinstance(value, (list, tuple)):
            continue
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError):
            # Some SDK-supported inputs are jagged rather than dense tensors.
            continue
        if tensor.ndim == 1:
            prepared[key] = tensor
    return prepared
