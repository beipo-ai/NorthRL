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

"""
Tests for SimpleRolloutWorker using real implementation with mocked inference client
"""

import pytest

from northrl.mock import MockInferenceServiceClient
from northrl.rollout_worker import SimpleRolloutWorker
from northrl.utils.async_utils import run_coro


@pytest.fixture
def simple_rollout_worker(rollout_worker_config):
    """Create SimpleRolloutWorker with MockInferenceServiceClient"""
    worker = SimpleRolloutWorker(rollout_worker_config)
    # Manually set the inference client for testing
    worker._inference_client = MockInferenceServiceClient(rollout_worker_config)
    return worker


def test_simple_rollout_worker_initialization(simple_rollout_worker):
    """Test SimpleRolloutWorker initialization"""
    assert simple_rollout_worker._config is not None
    assert simple_rollout_worker._inference_client is not None
    assert isinstance(simple_rollout_worker._inference_client, MockInferenceServiceClient)


def test_simple_rollout_worker_rollout(simple_rollout_worker):
    """Test SimpleRolloutWorker rollout method"""
    # Mock the _put_trajectory method to capture trajectories
    trajectories = []

    def mock_put_trajectory(trajectory):
        trajectories.append(trajectory)
        return "trajectory_id_123"

    simple_rollout_worker._put_trajectory = mock_put_trajectory

    # Execute a rollout
    task = {"prompt": "What is 2+2?", "id": 1}
    result = run_coro(simple_rollout_worker.rollout(task))

    # Should have processed the task
    assert result == "trajectory_id_123"
    assert len(trajectories) == 1

    # Check trajectory structure
    traj = trajectories[0]
    assert "response" in traj
    assert "finish_reason" in traj
    assert "Mock response" in traj["response"]


def test_simple_rollout_worker_rollout_missing_prompt(simple_rollout_worker):
    """Test SimpleRolloutWorker rollout with missing prompt"""
    # Mock the _put_trajectory method
    simple_rollout_worker._put_trajectory = lambda x: "id"

    # Execute a rollout without prompt
    task = {"id": 1}
    result = run_coro(simple_rollout_worker.rollout(task))

    # Should return None when prompt is missing
    assert result is None


def test_simple_rollout_worker_inference_integration(simple_rollout_worker):
    """Test that SimpleRolloutWorker correctly uses inference client"""
    # Mock the _put_trajectory method to capture trajectories
    trajectories = []
    simple_rollout_worker._put_trajectory = lambda x: trajectories.append(x) or "id"

    # Execute a rollout
    task = {"prompt": "Test prompt", "id": 1, "extra_field": "extra_value"}
    run_coro(simple_rollout_worker.rollout(task))

    # Check that extra fields from task are included in trajectory
    traj = trajectories[0]
    assert traj["id"] == 1
    assert traj["extra_field"] == "extra_value"


def test_simple_rollout_preserves_sc_from_canonical_envelope(simple_rollout_worker):
    q = {
        "sampler_logprobs": [-1.0, -2.0],
        "sampler_topk_ids": [[3, 6], [4, 6]],
        "sampler_topk_logprobs": [[-1.0, -2.0], [-2.0, -1.0]],
        "sampler_topk_mask": [[1, 1], [1, 1]],
        "sampler_distribution": {"schema": "behavior/unfiltered/v1"},
    }
    result = {
        "northrl_train": {
            "prompt_tokens": [1, 2],
            "response_tokens": [3, 4],
            "response_logprobs": [-1.0, -2.0],
            "response_old_logprobs": [-1.0, -2.0],
        },
        "choices": [{"text": "<answer>2</answer>", "finish_reason": "length"}],
        "sampler_distribution_data": q,
    }
    captured = []
    simple_rollout_worker._put_trajectory = lambda t: captured.append(t) or "id"
    simple_rollout_worker._build_trajectory_from_completion(
        {"prompt": "1+1?", "ground_truth": "2"}, result
    )
    (traj,) = captured
    assert traj.tokens == [1, 2, 3, 4]
    assert traj.extra_fields["sampler_distribution_data"] == q
    assert traj.extra_fields["finish_reason"] == "length"
    assert traj.reward == 1.0
