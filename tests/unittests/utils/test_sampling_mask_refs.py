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

import hashlib
import struct

import pytest
from weaver.sampling_client import SamplingClient
from weaver.types import ModelInput, SamplingParams

from northrl.northrl_types import Trajectory
from northrl.remote_rollout.trajectory_recorder import TrajectoryRecorder
from northrl.trainer.remote_api_trainer import RemoteApiTrainer
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.sampling_mask_refs import (
    align_sampling_mask_ref,
    join_sampling_masks,
    trim_sampling_mask,
)
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


def mask(tokens):
    return {
        "sampling_mask_ref": {
            "schema": "weaver.sampling_mask.v1",
            "token_count": len(tokens),
            "empty_row": "full-vocabulary",
            "weight_version": "v1",
            "tokens_sha256": hashlib.sha256(
                struct.pack("<" + "i" * len(tokens), *tokens)
            ).hexdigest(),
        }
    }


def test_multi_turn_ref_alignment_and_truncation():
    first, second = mask([3, 4]), mask([7, 8])
    both = join_sampling_masks(first, second)
    retained = trim_sampling_mask(both, 3)
    aligned = align_sampling_mask_ref([1, 2, 3, 4, 6, 7], [0, 0, 1, 1, 0, 1], retained)
    assert [p["sampling_mask_positions"] for p in aligned["sampling_mask_refs"]] == [[1, 2], [4]]
    assert aligned["sampling_mask_refs"][1]["sampling_mask_slice"] == {"start": 0, "stop": 1}
    assert second["sampling_mask_ref"]["token_count"] == 2
    trajectory = Trajectory(
        tokens=[1, 2, 3, 4, 6, 7],
        loss_mask=[0, 0, 1, 1, 0, 1],
        reward=1.0,
        is_val=False,
        extra_fields={
            "old_log_probs": [0.0, 0.0, -1.0, -1.0, 0.0, -1.0],
            "advantage": 1.0,
            "sampling_mask": retained,
        },
    )
    (datum,) = convert_trajectories_to_datums([trajectory])
    assert "sampling_mask" not in datum["loss_fn_inputs"]
    assert datum["metadata"] == aligned
    assert RemoteApiTrainer._sampling_mask_for_compute_logprobs(trajectory) == aligned
    with pytest.raises(ValueError):
        trim_sampling_mask(both, 5)
    with pytest.raises(ValueError):
        align_sampling_mask_ref([1, 3], [0, 1], both)


def test_holder_requests_ref_and_keeps_it_opaque():
    holder = object.__new__(WeaverServiceHolder)
    holder._return_sampling_mask = True
    holder._training_mode = "full_ft"
    kwargs = holder._build_sample_kwargs(
        sampling_client=SamplingClient(service=None, sampling_session_id="s", base_model="m"),
        model_input=ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=SamplingParams(top_k=8),
    )
    assert kwargs["sampling_mask_transport"] == "ref"
    holder._training_mode = "lora"
    legacy = holder._build_sample_kwargs(
        sampling_client=SamplingClient(service=None, sampling_session_id="s", base_model="m"),
        model_input=ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=SamplingParams(top_k=8),
    )
    assert "sampling_mask_transport" not in legacy
    holder._training_mode = "full_ft"
    sequence = {
        "tokens": [3, 4],
        "text": "x",
        "stop_reason": "stop",
        "logprobs": [-1.0, -2.0],
        "weight_version": "v1",
        **mask([3, 4]),
    }
    result = holder._weaver_sequence_to_sampling_result(sequence, [1, 2], "v1")
    assert result["response_sampling_masks"] == mask([3, 4])
    assert "sampling_masks" not in result


def test_recorder_preserves_refs_across_turns():
    recorder = TrajectoryRecorder()
    recorder.record_model_turn(
        prompt_tokens=[1, 2],
        response_tokens=[3, 4],
        response_log_probs=[-1.0, -1.0],
        response_old_log_probs=[-1.0, -1.0],
        sampling_mask=mask([3, 4]),
    )
    recorder.record_model_turn(
        prompt_tokens=[1, 2, 3, 4, 6],
        response_tokens=[7],
        response_log_probs=[-1.0],
        response_old_log_probs=[-1.0],
        sampling_mask=mask([7]),
    )
    assert len(recorder.to_remote_trajectories()) == 1
    assert len(recorder.to_remote_trajectories()[0].sampling_mask["sampling_mask_refs"]) == 2
