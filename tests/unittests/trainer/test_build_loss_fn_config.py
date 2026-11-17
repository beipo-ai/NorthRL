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

"""Tests for RemoteApiTrainer._build_loss_fn_config().

Use the real loss adapter and validated configuration without starting a service.
"""

from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.config_schema import AlgorithmConfig
from northrl.weaver.loss_adapter import build_loss_adapter


class _ConcreteTrainer(RemoteApiTrainer):
    """Minimal concrete subclass so we can instantiate without full init."""

    async def _prepare_trajectories(self, trajectories, metrics):
        return trajectories


def _make_stub(**overrides):
    """Keep the trainer-to-adapter boundary while constructing no service clients."""
    obj = object.__new__(_ConcreteTrainer)
    clip_fields = {
        "_truncation_c": "truncation_c",
        "_clip_ratio": "ratio",
        "_clip_ratio_low": "ratio_low",
        "_clip_ratio_high": "ratio_high",
        "_clip_ratio_c": "ratio_c",
    }
    algorithm = AlgorithmConfig.model_validate(
        {
            "type": "grpo",
            "optim": {"lr": 1e-6},
            "entropy_coeff": overrides.get("_entropy_coeff", 1e-4),
            "loss": {
                "mode": "importance_sampling",
                "params": {
                    "clip": {
                        field: overrides[attr]
                        for attr, field in clip_fields.items()
                        if attr in overrides
                    }
                },
                "reduction": {"mode": overrides.get("reduction", "token_mean")},
            },
        }
    )
    obj._loss_adapter = build_loss_adapter(algorithm)
    obj._enable_debug_dump = overrides.get("_enable_debug_dump", False)
    return obj


class TestBuildLossFnConfig:
    def test_defaults_no_clip_params(self):
        """No clip keys when none are configured."""
        stub = _make_stub()
        cfg = stub._build_loss_fn_config()
        assert cfg == {
            "entropy_coeff": 1e-4,
            "enable_debug_dump": False,
            "loss_agg_mode": "sum",
        }
        # Make sure no clip-related keys leak in
        assert "truncation_c" not in cfg
        assert "clip_ratio" not in cfg

    def test_truncation_c_forwarded(self):
        stub = _make_stub(_truncation_c=6.0)
        cfg = stub._build_loss_fn_config()
        assert cfg["truncation_c"] == 6.0

    def test_all_clip_params_forwarded(self):
        stub = _make_stub(
            _truncation_c=10.0,
            _clip_ratio=5,
            _clip_ratio_low=5,
            _clip_ratio_high=5,
        )
        cfg = stub._build_loss_fn_config()
        assert cfg["truncation_c"] == 10.0
        assert cfg["clip_ratio"] == 5
        assert cfg["clip_ratio_low"] == 5
        assert cfg["clip_ratio_high"] == 5

    def test_partial_clip_params(self):
        """Only set keys should appear."""
        stub = _make_stub(_clip_ratio_high=3)
        cfg = stub._build_loss_fn_config()
        assert "clip_ratio_high" in cfg
        assert cfg["clip_ratio_high"] == 3
        assert "clip_ratio" not in cfg
        assert "truncation_c" not in cfg

    def test_entropy_coeff_override(self):
        stub = _make_stub(_entropy_coeff=0.01)
        cfg = stub._build_loss_fn_config()
        assert cfg["entropy_coeff"] == 0.01

    def test_debug_dump_flag(self):
        stub = _make_stub(_enable_debug_dump=True)
        cfg = stub._build_loss_fn_config()
        assert cfg["enable_debug_dump"] is True

    def test_loss_agg_mode_is_hardcoded(self):
        stub = _make_stub(reduction="datum_mean")
        assert stub._build_loss_fn_config()["loss_agg_mode"] == "sum"
