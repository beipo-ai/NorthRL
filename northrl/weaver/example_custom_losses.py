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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from weaver import types


def build_example_loss(params: dict[str, Any]):
    """Example custom loss factory for Weaver forward_backward_custom."""
    scale = float(params.get("scale", 1.0))

    def loss_fn(
        data: Sequence[types.Datum],
        logprob_tensors: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del data
        loss = -scale * torch.stack([logprobs.mean() for logprobs in logprob_tensors]).mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "scale": scale,
            "num_samples": len(logprob_tensors),
        }
        return loss, metrics

    return loss_fn
