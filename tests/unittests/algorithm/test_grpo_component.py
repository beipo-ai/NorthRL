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

"""GRPO formulas, trajectory ownership and estimator construction."""

import pytest
import torch

from northrl.algorithm.advantage import build_estimator
from northrl.algorithm.advantage.grpo import (
    GRPOEstimator,
    compute_grpo_advantage_for_trajectories,
)
from northrl.northrl_types import Trajectory
from northrl.utils.algorithm_config import EstimatorConfig


def _parents():
    return [
        Trajectory(
            segments=[],
            reward=reward,
            extra_fields={"group_id": "g", "run_id": run, "expected_rollouts": 3},
        )
        for run, reward in enumerate([1.0, 3.0, 8.0])
    ]


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("sample_std", [False, True])
def test_direct_component_preserves_numerics_and_ownership(norm, sample_std):
    direct = _parents()
    assert (
        compute_grpo_advantage_for_trajectories(
            direct, use_run_ids=sample_std, norm_adv_by_std=norm
        )
        is direct
    )
    values = torch.tensor([1.0, 3.0, 8.0], dtype=torch.float32 if sample_std else torch.float64)
    expected = values - values.mean()
    if norm:
        expected /= values.std(correction=int(sample_std)) + 1e-6
    assert [float(parent["advantage"]) for parent in direct] == pytest.approx(expected.tolist())
    assert all(isinstance(parent["advantage"], torch.Tensor) == sample_std for parent in direct)


def test_factory_builds_the_grpo_module_and_step_order_remains_stable():
    estimator = build_estimator(EstimatorConfig(mode="grpo", params={"norm_adv_by_std": False}))
    assert isinstance(estimator, GRPOEstimator)
    parents = list(reversed(_parents()))
    original_order = list(parents)
    result = estimator.estimate(parents, {}, {})
    assert [parent["run_id"] for parent in result] == [0, 1, 2]
    assert [float(parent["advantage"]) for parent in result] == [-3.0, -1.0, 4.0]
    assert all(left is right for left, right in zip(parents, original_order, strict=True))
    assert {id(parent) for parent in parents} == {id(parent) for parent in result}
