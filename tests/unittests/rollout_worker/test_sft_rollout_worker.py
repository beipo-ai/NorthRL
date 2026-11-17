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

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from northrl.rollout_worker import SFTRolloutWorker
from northrl.utils.async_utils import run_coro


def test_sft_rollout_worker_skips_inference_client(rollout_worker_config) -> None:
    config = OmegaConf.create(OmegaConf.to_container(rollout_worker_config, resolve=True))
    config.algorithm.type = "sft"
    config.rollout.type = None
    worker = SFTRolloutWorker(config)

    worker.init_inference_service_client(service_holder=object())

    assert worker._inference_client is None  # pylint: disable=protected-access


def test_sft_rollout_worker_requires_loss_mask(rollout_worker_config) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)

    with pytest.raises(ValueError, match="requires input_ids and loss_mask"):
        run_coro(worker.rollout({"input_ids": [1, 2], "labels": [-100, 2]}))


def test_sft_rollout_worker_uses_explicit_loss_mask(rollout_worker_config) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    trajectories = []
    worker._put_trajectory = lambda trajectory: trajectories.append(trajectory) or "success"

    result = run_coro(worker.rollout({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}))

    assert result == "success"
    assert len(trajectories) == 1
    assert trajectories[0].tokens == [1, 2, 3]
    assert trajectories[0].loss_mask == [0, 1, 1]


def test_sft_rollout_worker_preserves_token_free_managed_ref(rollout_worker_config) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    trajectories = []
    worker._put_trajectory = lambda trajectory: trajectories.append(trajectory) or "success"

    result = run_coro(
        worker.rollout(
            {
                "sample_ref": {
                    "dataset": "hq-math",
                    "version": "2026-08",
                    "sample_idx": 12,
                },
                "input_token_count": 7,
                "datum_id": "northrl-occurrence-12",
                "content_visibility": "public",
                "sft_batch_id": "sft-0-1",
            }
        )
    )

    assert result == "success"
    assert trajectories[0].tokens is None
    assert trajectories[0].loss_mask is None
    assert trajectories[0].is_managed_sample_ref
    assert trajectories[0].extra_fields["input_token_count"] == 7
    assert trajectories[0].extra_fields["content_visibility"] == "public"
    assert trajectories[0].extra_fields["loss_fn_inputs"] == {}


def test_sft_rollout_worker_routes_managed_ref_without_materializing_tokens(
    rollout_worker_config,
) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    worker._max_sequence_length = 8
    worker._trajectory_pool = MagicMock()
    worker._trajectory_pool.put_trajectory.return_value = "success"

    result = run_coro(
        worker.rollout(
            {
                "sample_ref": {
                    "dataset": "hq-math",
                    "version": "2026-08",
                    "sample_idx": 12,
                },
                # Shifted positions: conceptual input_ids length is exactly 8.
                "input_token_count": 7,
                "datum_id": "northrl-occurrence-12",
            }
        )
    )

    assert result == "success"
    trajectory = worker._trajectory_pool.put_trajectory.call_args.args[0]
    assert trajectory.tokens is None
    assert trajectory.loss_mask is None


def test_sft_rollout_worker_rejects_managed_ref_over_max_sequence_length(
    rollout_worker_config,
) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    worker._max_sequence_length = 8

    with pytest.raises(ValueError, match=r"sequence_length=9, max_sequence_length=8"):
        run_coro(
            worker.rollout(
                {
                    "sample_ref": {
                        "dataset": "hq-math",
                        "version": "2026-08",
                        "sample_idx": 12,
                    },
                    "input_token_count": 8,
                    "datum_id": "northrl-occurrence-12",
                }
            )
        )


def test_sft_rollout_worker_preserves_managed_attached_loss_inputs(
    rollout_worker_config,
) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    trajectories = []
    worker._put_trajectory = lambda trajectory: trajectories.append(trajectory) or "success"

    result = run_coro(
        worker.rollout(
            {
                "sample_ref": {
                    "dataset": "hq-math",
                    "version": "2026-08",
                    "sample_idx": 12,
                },
                "input_token_count": 7,
                "datum_id": "northrl-occurrence-12",
                "loss_fn_inputs": {"auxiliary_scale": [0.25]},
            }
        )
    )

    assert result == "success"
    assert trajectories[0].extra_fields["loss_fn_inputs"] == {"auxiliary_scale": [0.25]}


def test_sft_rollout_worker_rejects_ref_with_tokens(rollout_worker_config) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)

    with pytest.raises(ValueError, match="exactly one"):
        run_coro(
            worker.rollout(
                {
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                    "sample_ref": {
                        "dataset": "hq-math",
                        "version": "2026-08",
                        "sample_idx": 12,
                    },
                    "input_token_count": 1,
                    "datum_id": "northrl-occurrence-12",
                }
            )
        )


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"datum_id": "x" * 256}, "at most 255"),
    ],
)
def test_sft_rollout_worker_rejects_invalid_managed_sdk_inputs(
    rollout_worker_config, updates, error
) -> None:
    worker = SFTRolloutWorker(rollout_worker_config)
    task = {
        "sample_ref": {
            "dataset": "hq-math",
            "version": "2026-08",
            "sample_idx": 12,
        },
        "input_token_count": 1,
        "datum_id": "northrl-occurrence-12",
        "sft_batch_id": "sft-0-1",
    }
    task.update(updates)

    with pytest.raises(ValueError, match=error):
        run_coro(worker.rollout(task))
