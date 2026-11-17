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

"""Rollout sampling requirements survive canonical-only config serialization."""

from omegaconf import OmegaConf

from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.config_utils import get_algorithm_loss_mode, get_derived_sampling_flags


def test_score_centering_flags_survive_canonical_serialization():
    raw = {
        "type": "grpo",
        "optim": {"lr": 1e-6},
        "loss": {"mode": "score_centering", "params": {"topk": 16}},
        "sampling_logprob_source": "inference",
    }
    canonical = AlgorithmConfig.model_validate(raw).model_dump()
    configs = [
        OmegaConf.create({"algorithm": algorithm, "rollout": {}}) for algorithm in (raw, canonical)
    ]
    assert "loss_fn" not in canonical
    assert [get_algorithm_loss_mode(config) for config in configs] == ["score_centering"] * 2
    assert (
        get_derived_sampling_flags(configs[0])
        == get_derived_sampling_flags(configs[1])
        == {
            "return_sampling_mask": False,
            "return_old_logprob": False,
            "return_moe_topk_indices": False,
            "topk_output_logprobs": 16,
        }
    )


def test_canonical_router_replay_and_mis_keep_probability_requirements():
    algorithm = AlgorithmConfig.model_validate(
        {
            "type": "grpo",
            "optim": {"lr": 1e-6},
            "loss": {"mode": "ppo_clip"},
            "router_replay": {"enabled": True, "mode": "R3"},
            "mis": {"enabled": True},
        }
    ).model_dump()
    config = OmegaConf.create({"algorithm": algorithm, "rollout": {"keep_sampling_mask": True}})
    assert get_derived_sampling_flags(config) == {
        "return_sampling_mask": True,
        "return_old_logprob": True,
        "return_moe_topk_indices": True,
    }
