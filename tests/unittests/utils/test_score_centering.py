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

"""SC rollout alignment and production Datum conversion."""

from types import SimpleNamespace

import pytest

from northrl.northrl_types import Trajectory
from northrl.utils.config_schema import AlgorithmConfig
from northrl.utils.finetune_service_utils import convert_trajectories_to_datums
from northrl.utils.score_centering import align_sampler_distribution, trim_sampler_distribution


def distribution():
    return {
        "sampler_logprobs": [-1.0, -2.0, -3.0],
        "sampler_topk_ids": [[3, 6], [4, 6], [5, 6]],
        "sampler_topk_logprobs": [[-1.0, -2.0], [-2.0, -1.0], [-3.0, -1.0]],
        "sampler_topk_mask": [[1, 1]] * 3,
        "sampler_distribution": {"schema": "behavior/unfiltered/v1", "temperature": 1.0},
    }


def test_datum_alignment_preserves_last_response_and_independent_behavior():
    q = distribution()
    traj = Trajectory(
        tokens=[1, 2, 3, 4, 5],
        loss_mask=[0, 0, 1, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields={
            "advantage": 2.0,
            "old_log_probs": [0.0, 0.0, -8.0, -8.0, -8.0],
            "sampler_distribution_data": q,
        },
    )
    (datum,) = convert_trajectories_to_datums([traj])
    fields = datum["loss_fn_inputs"]
    assert datum["input_tokens"] == [1, 2, 3, 4]
    assert fields["target_tokens"] == [2, 3, 4, 5]
    assert fields["loss_mask"] == [0.0, 1.0, 1.0, 1.0]
    assert fields["sampler_logprobs"] == [0.0, -1.0, -2.0, -3.0]
    assert fields["logprobs"] == [0.0, -8.0, -8.0, -8.0]
    assert fields["sampler_topk_mask"] == [[0, 0], [1, 1], [1, 1], [1, 1]]
    assert datum["metadata"]["sampler_distribution"] == q["sampler_distribution"]


def test_truncation_keeps_all_fields_in_lockstep():
    q = trim_sampler_distribution(distribution(), 2)
    fields, _ = align_sampler_distribution([1, 2, 3, 4], [0, 0, 1, 1], q)
    assert fields["sampler_topk_ids"] == [[0, 0], [3, 6], [4, 6]]
    assert fields["sampler_logprobs"] == [0.0, -1.0, -2.0]


@pytest.mark.parametrize("mask", [[1, 1, 1], [0, 1, 0], [0, 0, 0]])
def test_reject_non_single_turn(mask):
    with pytest.raises(ValueError):
        align_sampler_distribution([1, 2, 3], mask, distribution())


def test_service_holder_preserves_head_and_skips_it_for_greedy_evaluation():
    from weaver.sampling_client import SamplingClient
    from weaver.types import ModelInput, SamplingParams

    from northrl.weaver.weaver_service_holder import WeaverServiceHolder

    holder = object.__new__(WeaverServiceHolder)
    holder._topk_output_logprobs = 2
    kwargs = dict(
        sampling_client=SamplingClient(service=None, sampling_session_id="s", base_model="m"),
        model_input=ModelInput.from_ints([1, 2]),
        num_samples=1,
    )
    original_params = SamplingParams()
    train = holder._build_sample_kwargs(**kwargs, sampling_params=original_params)
    assert original_params.score_centering is None
    assert "score_centering" not in train
    assert train["sampling_params"].score_centering["head_size"] == 2
    evaluation = holder._build_sample_kwargs(
        **kwargs, sampling_params=SamplingParams(temperature=0, top_k=1)
    )
    assert "score_centering" not in evaluation
    assert evaluation["sampling_params"].score_centering is None
    q = ref_distribution()
    seq = {
        "tokens": [3, 4, 5],
        "logprobs": [-1.0, -2.0, -3.0],
        "weight_version": "v1",
        "text": "answer",
        "stop_reason": "stop",
        **q,
    }
    assert train["sampling_params"].score_centering["transport"] == "ref"
    result = holder._weaver_sequence_to_sampling_result(seq, [1, 2], "v1", expected_topk=2)
    assert result["sampler_distribution_data"] == q
    del seq["sampler_distribution_ref"]
    with pytest.raises(ValueError, match="reference"):
        holder._weaver_sequence_to_sampling_result(seq, [1, 2], "v1", expected_topk=2)
    # Greedy eval is deliberately not required to provide training statistics.
    holder._weaver_sequence_to_sampling_result(seq, [1, 2], "v1", expected_topk=0)


@pytest.mark.parametrize("mode,want", [("token_mean", [1 / 3, 1 / 3]), ("datum_mean", [0.5, 0.25])])
def test_sum_mode_normalizes_sc_advantages_once(mode, want):
    from northrl.trainer.remote_api_trainer import RemoteApiTrainer

    trainer = SimpleNamespace()
    trainer._loss_fn, trainer._train_step = "score_centering", 0
    trainer._algorithm_config = AlgorithmConfig(
        type="grpo",
        optim={"lr": 1e-6},
        loss={"mode": "score_centering", "reduction": {"mode": mode}},
    )
    data = [{"loss_fn_inputs": {"advantages": [1.0] * n, "loss_mask": [1.0] * n}} for n in [1, 2]]
    normalized = RemoteApiTrainer._normalize_advantages_for_sum(trainer, data)
    assert [d["loss_fn_inputs"]["advantages"][0] for d in normalized] == want
    assert data[0]["loss_fn_inputs"]["advantages"] == [1.0]


def ref_distribution():
    import hashlib
    import struct

    return {
        "sampler_logprobs": [-1.0, -2.0, -3.0],
        "sampler_distribution": {"schema": "behavior/unfiltered/v1", "weight_version": "v1"},
        "sampler_distribution_ref": {
            "schema": "weaver.sampler_distribution.v1",
            "distribution_schema": "behavior/unfiltered/v1",
            "top_k": 2,
            "token_count": 3,
            "weight_version": "v1",
            "tokens_sha256": hashlib.sha256(struct.pack("<iii", 3, 4, 5)).hexdigest(),
        },
    }


def test_ref_truncation_and_conversion_keep_handle_immutable():
    original = ref_distribution()
    q = trim_sampler_distribution(original, 2)
    q = trim_sampler_distribution(q, 1)
    trajectory = Trajectory(
        tokens=[1, 2, 3],
        loss_mask=[0, 0, 1],
        reward=1.0,
        is_val=False,
        extra_fields={
            "advantage": 2.0,
            "old_log_probs": [0.0, 0.0, -1.0],
            "sampler_distribution_data": q,
        },
    )
    (datum,) = convert_trajectories_to_datums([trajectory])
    assert datum["metadata"]["sampler_distribution_ref"] == original["sampler_distribution_ref"]
    assert datum["metadata"]["sampler_distribution_slice"] == {"start": 0, "stop": 1}
    assert not any(key.startswith("sampler_") for key in datum["loss_fn_inputs"])
    assert "sampler_distribution_slice" not in original
    with pytest.raises(ValueError, match="extend"):
        trim_sampler_distribution(q, 2)
