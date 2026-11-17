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

import pytest
from omegaconf import OmegaConf

from northrl.algorithm.advantage import build_estimator
from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.utils.async_utils import run_coro
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.logging_utils import log_rollout_metrics


@pytest.fixture
def trainer():
    trainer = object.__new__(RemoteApiGrpoTrainer)
    trainer._config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 16, "max_sequence_length": 64},
            "rollout": {"sampling": {"max_tokens": 8}},
        }
    )
    trainer._kl_enabled = False
    trainer._algorithm_config = AlgorithmConfig(type="grpo", optim={"lr": 1e-6})
    trainer._advantage_estimator = build_estimator(trainer._algorithm_config.advantage.estimator)
    return trainer


def _trajectory(prompt_length=2, response_length=3, **metadata):
    return Trajectory(
        tokens=[1] * prompt_length + [2] * response_length,
        loss_mask=[0] * prompt_length + [1] * response_length,
        reward=1.0,
        extra_fields={"is_truncated": False, **metadata},
    )


def _prepare_metrics(trainer, trajectories):
    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    prepared = run_coro(trainer._prepare_trajectories(trajectories, metrics))
    trainer._advantage_estimator.estimate(prepared, {}, metrics)
    return metrics


def test_identical_short_sequences_report_saved_response_lengths(trainer):
    metrics = _prepare_metrics(trainer, [_trajectory(), _trajectory()])
    assert metrics["trajectory/response_tokens_mean"] == 3
    assert metrics["segment/response_tokens_mean"] == 3
    assert metrics["segment/truncated_count"] == 0
    assert not any(
        key.startswith(("response_length/", "prompt_length/", "truncation/")) for key in metrics
    )


@pytest.mark.parametrize(
    ("metadata", "expected_clipped"),
    [
        ({"finish_reason": "length"}, 0),
        ({"finish_reason": b"length"}, 0),
        ({"is_truncated": True}, 1),
    ],
)
def test_only_actual_token_clipping_counts_and_clean_round_resets(
    trainer, metadata, expected_clipped
):
    summary = _prepare_metrics(trainer, [_trajectory(**metadata), _trajectory()])
    assert summary["segment/truncated_count"] == expected_clipped
    assert summary["outcome/truncated_trajectory_count"] == expected_clipped
    # A length stop describes generation, not evidence of worker token clipping.
    summary.update(_prepare_metrics(trainer, [_trajectory(), _trajectory()]))
    assert summary["segment/truncated_count"] == 0
    assert summary["outcome/truncated_trajectory_count"] == 0


def test_multiturn_lengths_distinguish_parent_total_and_saved_segments(trainer):
    segments = [
        {
            "tokens": [1, 2] + [3] * 12,
            "loss_mask": [0, 0] + [1] * 12,
            "finish_reason": "stop",
            "is_truncated": False,
        },
        {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "finish_reason": "length",
            "is_truncated": False,
        },
    ]
    metrics = _prepare_metrics(trainer, [Trajectory(segments=segments, reward=1.0)])
    # Per-request max_tokens=8 is not a limit on a saved multi-turn response.
    assert metrics["trajectory/response_tokens_mean"] == 13
    assert metrics["segment/response_tokens_mean"] == 6.5
    assert metrics["segment/response_tokens_max"] == 12
    assert metrics["segment/truncated_count"] == 0
    assert "model_request/output_tokens_mean" not in metrics


def test_prompt_threshold_does_not_claim_actual_clipping(trainer):
    metrics = _prepare_metrics(trainer, [_trajectory(prompt_length=16), _trajectory()])
    assert metrics["segment/truncated_count"] == 0
    assert "prompt_length/clip_ratio" not in metrics
    assert "response_length/clip_ratio" not in metrics


def test_input_statistics_retain_skipped_group_before_prepare_filtering(trainer):
    trajectory = _trajectory(is_truncated=True, group_skipped=True)
    metrics = _prepare_metrics(trainer, [trajectory])
    assert metrics["batch/trajectory_count"] == 1
    assert metrics["batch/segment_count"] == 1
    assert metrics["segment/truncated_count"] == 1
    assert "trajectory/advantage_mean" not in metrics


def test_missing_truncation_observation_is_not_zero(trainer):
    trajectory = Trajectory(tokens=[1, 2], loss_mask=[0, 1], reward=1.0)
    metrics = _prepare_metrics(trainer, [trajectory])
    assert "segment/truncated_count" not in metrics
