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

import pytest

from northrl.utils.async_utils import run_coro
from northrl.weaver.weaver_service_holder import (
    ForwardLogprobsResult,
    WeaverServiceHolder,
    _extract_forward_logprobs_batch_from_result,
    _extract_router_replay_record_ref,
)


def _record_envelope(replay_set_id: str = "record-123") -> dict:
    replay_set_uri = f"weaver://model-a/router-replay/{replay_set_id}"
    return {
        "schema": "weaver.router_replay.index_set.v1",
        "transport": "gpfs_safetensors",
        "uri": replay_set_uri,
        "manifest_uri": f"{replay_set_uri}/manifest.json",
        "future_field": {"preserved_by_producer": True},
    }


def test_forward_response_uses_canonical_result_contract():
    response = {
        "result": {
            "loss_fn_outputs": [
                {"logprobs": {"data": [-0.1, -0.2]}},
                {"logprobs": [-0.3]},
            ]
        }
    }

    assert _extract_forward_logprobs_batch_from_result(response) == [
        [-0.1, -0.2],
        [-0.3],
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"Response": {"result": {"loss_fn_outputs": []}}},
        {"result": {"metrics": {}}},
        None,
    ],
)
def test_forward_response_rejects_legacy_or_incomplete_shapes(response):
    with pytest.raises(ValueError):
        _extract_forward_logprobs_batch_from_result(response)


def test_r2_record_ref_reads_only_canonical_metrics_path():
    envelope = _record_envelope()
    response = {
        "result": {
            "metrics": {"router_replay_recorded_indices": envelope},
            "loss_fn_outputs": [{"logprobs": [-0.1]}],
        }
    }

    assert _extract_router_replay_record_ref(response) == {
        **envelope,
        "index_set_uri": envelope["uri"],
    }


def test_r2_record_ref_does_not_walk_nested_response_graphs():
    legacy_response = {
        "Response": {"result": {"metrics": {"router_replay_recorded_indices": _record_envelope()}}}
    }

    with pytest.raises(ValueError, match="canonical result"):
        _extract_router_replay_record_ref(legacy_response)


def test_r2_forward_returns_logprobs_and_record_ref_together():
    class FakeTrainingClient:
        def __init__(self):
            self.datums = None

        async def forward(self, datums, loss_fn, wait=False):
            assert loss_fn == "forward_logprob"
            assert wait is False
            self.datums = datums
            return {
                "result": {
                    "loss_fn_outputs": [{"logprobs": {"data": [-0.1, -0.2]}}],
                    "metrics": {"router_replay_recorded_indices": _record_envelope("record-1")},
                }
            }

    holder = object.__new__(WeaverServiceHolder)
    holder._closed = False
    # __new__ bypasses provisioning; this is an initialized test holder.
    holder._tokenizer = object()
    holder._training_client = FakeTrainingClient()

    result = run_coro(holder.compute_target_logprobs_batch_with_record([[10, 11, 12]]))

    assert isinstance(result, ForwardLogprobsResult)
    assert result.logprobs == [[-0.1, -0.2]]
    assert result.router_replay_record_ref == {
        **_record_envelope("record-1"),
        "index_set_uri": "weaver://model-a/router-replay/record-1",
    }
    assert holder._training_client.datums[0].metadata["router_replay"] == {
        "schema": "weaver.router_replay.datum.v1",
        "mode": "R2",
        "source": "recompute",
        "fail_fast": True,
        "action": "RECORD",
    }


def test_r3_metadata_is_attached_to_denominator_forward_datums():
    class FakeTrainingClient:
        def __init__(self):
            self.datums = None

        async def forward(self, datums, loss_fn, wait=False):
            assert loss_fn == "forward_logprob"
            assert wait is False
            self.datums = datums
            return {
                "result": {
                    "loss_fn_outputs": [
                        {"logprobs": [-0.1, -0.2]},
                        {"logprobs": [-0.3]},
                    ]
                }
            }

    metadata_batch = [
        {
            "router_replay": {
                "schema": "weaver.router_replay.datum.v1",
                "mode": "R3",
                "source": "rollout",
                "action": "REPLAY",
                "value_ref": {"relative_path": "rollout/sequence-0.safetensors"},
            }
        },
        {
            "router_replay": {
                "schema": "weaver.router_replay.datum.v1",
                "mode": "R3",
                "source": "rollout",
                "action": "REPLAY",
                "value_ref": {"relative_path": "rollout/sequence-1.safetensors"},
            }
        },
    ]
    holder = object.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._tokenizer = object()
    holder._training_client = FakeTrainingClient()

    logprobs = run_coro(
        holder.compute_target_logprobs_batch(
            [[10, 11, 12], [20, 21]],
            datum_metadata_batch=metadata_batch,
        )
    )

    assert logprobs == [[-0.1, -0.2], [-0.3]]
    assert [datum.metadata for datum in holder._training_client.datums] == metadata_batch


def test_denominator_forward_rejects_misaligned_metadata_batch():
    holder = object.__new__(WeaverServiceHolder)
    holder._closed = False
    holder._tokenizer = object()
    holder._training_client = object()

    with pytest.raises(ValueError, match="datum_metadata_batch length"):
        run_coro(
            holder.compute_target_logprobs_batch(
                [[10, 11], [20, 21]],
                datum_metadata_batch=[{}],
            )
        )


def test_async_sampler_kwargs_request_r3_ref_when_enabled():
    class FakeSamplingClient:
        async def sample(
            self,
            *,
            prompt,
            num_samples,
            sampling_params,
            include_prompt_logprobs,
            return_sampling_mask,
            return_old_logprob,
            return_moe_topk_indices,
            wait=False,
        ):
            del (
                prompt,
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                return_sampling_mask,
                return_old_logprob,
                return_moe_topk_indices,
                wait,
            )

    holder = object.__new__(WeaverServiceHolder)
    run_coro(
        holder.configure_sampling_flags(
            return_sampling_mask=True,
            return_old_logprob=True,
            return_moe_topk_indices=True,
        )
    )

    kwargs = holder._build_sample_kwargs(
        sampling_client=FakeSamplingClient(),
        model_input="prompt",
        num_samples=2,
        sampling_params="params",
    )

    assert kwargs["return_sampling_mask"] is True
    assert kwargs["return_old_logprob"] is True
    assert kwargs["return_moe_topk_indices"] is True
    # _sample_result_from_handle owns wait=False for the async SDK operation.
    assert "wait" not in kwargs


def test_sampling_output_keeps_router_ref_opaque():
    holder = object.__new__(WeaverServiceHolder)
    opaque_ref = {
        "storage": "gpfs",
        "relative_path": "rollout/seq-0.safetensors",
        "future_server_field": [1, 2, 3],
    }

    old_logprobs, sampling_masks, routing_ref = holder._extract_sampling_outputs(
        {
            "old_logprobs": [-0.1],
            "sampling_masks": [[1, 2]],
            "moe_topk_indices_ref": opaque_ref,
        },
        [-0.2],
    )

    assert old_logprobs == [-0.1]
    assert sampling_masks == [[1, 2]]
    assert routing_ref == opaque_ref
