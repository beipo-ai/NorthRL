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

"""Prompt-mean restrictions also apply to direct trainer construction."""

import re
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from northrl.trainer import remote_api_trainer as trainer_module
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.config_schema import KLConfig
from tests.unittests.trainer.test_remote_api_trainer_custom_loss import (
    _DummyDataDumper,
    _make_config,
)


def _direct_config():
    config = _make_config(batch_size=None, mini_batch_size=2)
    config.algorithm.loss.reduction.mode = "prompt_mean"
    config.algorithm.kl = KLConfig().model_dump()
    config.models = {"reference": None}
    return config


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo_clip"])
def test_direct_prompt_mean_construction_accepts_native_losses(monkeypatch, loss_fn):
    config = _direct_config()
    config.algorithm.loss.mode = loss_fn
    monkeypatch.setattr(trainer_module, "get_data_dumper", lambda *a, **kw: _DummyDataDumper())

    trainer = RemoteApiGrpoTrainer(config)

    assert trainer._loss_agg_mode == "prompt-mean"
    assert trainer._batch_size is None
    assert trainer._mini_batch_size == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("type", "opd", "requires algorithm.type='grpo'"),
        ("loss.mode", "truncated_importance_sampling", "native importance_sampling or ppo_clip"),
        ("entropy_coeff", 1.0e-4, "requires algorithm.entropy_coeff=0"),
        ("kl", {"mode": "in_loss"}, "requires algorithm.kl.mode=null"),
        (
            "loss",
            {
                "mode": "custom",
                "params": {"name": "direct_logprob"},
                "reduction": {"mode": "prompt_mean"},
            },
            "does not support algorithm.loss.mode='custom'",
        ),
        ("optim.batch_size", 8, "requires algorithm.optim.batch_size=null"),
    ],
)
def test_direct_prompt_mean_construction_rejects_before_setup(monkeypatch, field, value, message):
    config = _direct_config()
    OmegaConf.update(config, "algorithm." + field, value, merge=False)
    dumper_factory = Mock()
    monkeypatch.setattr(trainer_module, "get_data_dumper", dumper_factory)

    with pytest.raises(ValidationError, match=re.escape(message)):
        RemoteApiGrpoTrainer(config)

    dumper_factory.assert_not_called()
