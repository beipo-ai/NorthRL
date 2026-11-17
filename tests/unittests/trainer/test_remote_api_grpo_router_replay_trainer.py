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

from northrl.northrl_types import Trajectory
from northrl.trainer.remote_api_grpo_trainer import RemoteApiGrpoTrainer
from northrl.trainer.router_replay_handler import RouterReplayHandler
from northrl.utils.async_utils import run_coro
from northrl.weaver.weaver_service_holder import ForwardLogprobsResult


def _handler(mode: str) -> RouterReplayHandler:
    return RouterReplayHandler(
        {
            "sampling_logprob_source": "recomputed",
            "router_replay": {"enabled": True, "mode": mode},
        }
    )


def _trainer(mode: str) -> RemoteApiGrpoTrainer:
    trainer = RemoteApiGrpoTrainer.__new__(RemoteApiGrpoTrainer)
    trainer._training_temperature = 1.0
    trainer._loss_agg_mode = "token-mean"
    trainer._mini_batch_size = None
    trainer._submit_ahead = 0
    trainer._router_replay = _handler(mode)
    trainer._service_holder = None
    return trainer


def _trajectory(
    *,
    advantage: float = 1.0,
    routing_ref: dict | None = None,
) -> Trajectory:
    extra_fields = {
        "advantage": advantage,
        "old_log_probs": [0.0, -0.1, -0.2],
    }
    if routing_ref is not None:
        extra_fields["moe_topk_indices_ref"] = routing_ref
    return Trajectory(
        tokens=[10, 11, 12],
        loss_mask=[0, 1, 1],
        reward=1.0,
        is_val=False,
        extra_fields=extra_fields,
    )


def _record_ref(replay_set_id: str) -> dict:
    index_set_uri = f"weaver://model-a/router-replay/{replay_set_id}"
    return {
        "schema": "weaver.router_replay.r2_manifest_handle.v1",
        "index_set_uri": index_set_uri,
        "manifest_uri": f"{index_set_uri}/manifest.json",
    }


def test_router_replay_handler_has_no_per_step_pending_state():
    handler = _handler("R2")

    assert vars(handler) == {"enabled": True, "mode": "R2"}


def test_r3_metadata_is_attached_to_zero_advantage_datums_and_preserves_opaque_refs():
    trainer = _trainer("R3")
    zero_advantage_ref = {"relative_path": "zero-advantage.safetensors"}
    retained_ref = {
        "storage": "gpfs",
        "format": "safetensors",
        "relative_path": "retained.safetensors",
        "shape": [2, 8, 2],
        "future_server_field": {"version": 2},
    }
    trajectories = [
        _trajectory(advantage=0.0, routing_ref=zero_advantage_ref),
        _trajectory(advantage=1.0, routing_ref=retained_ref),
    ]

    datums = trainer._convert_to_datums(trajectories)

    assert len(datums) == 2
    replay_refs = [datum["metadata"]["router_replay"]["value_ref"] for datum in datums]
    assert replay_refs == [zero_advantage_ref, retained_ref]
    assert replay_refs[0] is not zero_advantage_ref
    assert replay_refs[1] is not retained_ref


def test_r3_conversion_fails_closed_when_rollout_ref_is_missing():
    trainer = _trainer("R3")

    with pytest.raises(ValueError, match="missing moe_topk_indices_ref"):
        trainer._convert_to_datums([_trajectory()])


def test_r3_recompute_replays_rollout_routing_and_replaces_denominator():
    class FakeWeaverServiceHolder:
        def __init__(self):
            self.datum_metadata_batch = None

        async def compute_target_logprobs_batch(
            self,
            tokens_batch,
            sampling_masks=None,
            datum_metadata_batch=None,
            *,
            loss_fn_config=None,
        ):
            assert loss_fn_config == {"temperature": 1.0}
            assert tokens_batch == [[10, 11, 12]]
            assert sampling_masks == [[[11, 13], [12, 14]]]
            self.datum_metadata_batch = datum_metadata_batch
            return [[-0.3, -0.4]]

    rollout_ref = {
        "storage": "gpfs",
        "relative_path": "rollout/sequence-0.safetensors",
        "future_server_field": {"version": 2},
    }
    holder = FakeWeaverServiceHolder()
    trainer = _trainer("R3")
    trainer._service_holder = holder
    trajectory = _trajectory(routing_ref=rollout_ref)
    trajectory.extra_fields["sampling_mask"] = [[11, 13], [12, 14]]

    prepared = run_coro(trainer._recompute_logprobs([trajectory], {}))

    expected_metadata = {
        "router_replay": {
            "schema": "weaver.router_replay.datum.v1",
            "mode": "R3",
            "source": "rollout",
            "action": "REPLAY",
            "fail_fast": True,
            "value_ref": rollout_ref,
        }
    }
    assert holder.datum_metadata_batch == [expected_metadata]
    assert prepared[0].extra_fields["old_log_probs"] == [0.0, -0.3, -0.4]

    training_datum = trainer._convert_to_datums(prepared)[0]
    assert training_datum["loss_fn_inputs"]["logprobs"] == [-0.3, -0.4]
    assert training_datum["loss_fn_inputs"]["sampling_mask"] == [[11, 13], [12, 14]]
    assert training_datum["metadata"] == expected_metadata


def test_r3_recompute_fails_closed_when_rollout_ref_is_missing():
    class FakeWeaverServiceHolder:
        async def compute_target_logprobs_batch(self, *args, **kwargs):
            raise AssertionError("forward_logprob must not run without the rollout ref")

    trainer = _trainer("R3")
    trainer._service_holder = FakeWeaverServiceHolder()

    with pytest.raises(ValueError, match="missing moe_topk_indices_ref"):
        run_coro(trainer._recompute_logprobs([_trajectory()], {}))


def test_r3_handler_rejects_inference_logprob_source():
    with pytest.raises(ValueError, match="sampling_logprob_source='recomputed'"):
        RouterReplayHandler(
            {
                "sampling_logprob_source": "inference",
                "router_replay": {"enabled": True, "mode": "R3"},
            }
        )


def test_r2_recompute_binds_explicit_record_result_to_each_trajectory():
    class FakeWeaverServiceHolder:
        async def compute_target_logprobs_batch_with_record(
            self,
            tokens_batch,
            sampling_masks=None,
            *,
            loss_fn_config=None,
        ):
            assert loss_fn_config == {"temperature": 1.0}
            assert tokens_batch == [[10, 11, 12], [10, 11, 12]]
            assert sampling_masks is None
            return ForwardLogprobsResult(
                logprobs=[[-0.3, -0.4], [-0.5, -0.6]],
                router_replay_record_ref=_record_ref("record-1"),
            )

    trainer = _trainer("R2")
    trainer._service_holder = FakeWeaverServiceHolder()
    trajectories = [_trajectory(), _trajectory()]

    prepared = run_coro(trainer._recompute_logprobs(trajectories, {}))

    assert prepared[0].extra_fields["old_log_probs"] == [0.0, -0.3, -0.4]
    assert prepared[1].extra_fields["old_log_probs"] == [0.0, -0.5, -0.6]
    assert prepared[0].extra_fields["router_replay_ref"] == {
        **_record_ref("record-1"),
        "sample_index": 0,
    }
    assert prepared[1].extra_fields["router_replay_ref"] == {
        **_record_ref("record-1"),
        "sample_index": 1,
    }

    datums = trainer._convert_to_datums(prepared)
    assert [datum["metadata"]["router_replay"]["value_ref"] for datum in datums] == [
        {**_record_ref("record-1"), "sample_index": 0},
        {**_record_ref("record-1"), "sample_index": 1},
    ]


def test_r2_record_results_do_not_leak_between_recompute_calls():
    class FakeWeaverServiceHolder:
        def __init__(self):
            self.call_count = 0

        async def compute_target_logprobs_batch_with_record(
            self,
            tokens_batch,
            sampling_masks=None,
            *,
            loss_fn_config=None,
        ):
            assert loss_fn_config == {"temperature": 1.0}
            del sampling_masks
            self.call_count += 1
            replay_set_id = f"record-{self.call_count}"
            return ForwardLogprobsResult(
                logprobs=[[-0.1, -0.2] for _ in tokens_batch],
                router_replay_record_ref=_record_ref(replay_set_id),
            )

    trainer = _trainer("R2")
    trainer._service_holder = FakeWeaverServiceHolder()
    first = [_trajectory()]
    second = [_trajectory()]

    run_coro(trainer._recompute_logprobs(first, {}))
    run_coro(trainer._recompute_logprobs(second, {}))

    assert first[0].extra_fields["router_replay_ref"]["index_set_uri"].endswith("record-1")
    assert second[0].extra_fields["router_replay_ref"]["index_set_uri"].endswith("record-2")


def test_r2_recompute_fails_when_same_forward_has_no_record_ref():
    class FakeWeaverServiceHolder:
        async def compute_target_logprobs_batch_with_record(
            self,
            tokens_batch,
            sampling_masks=None,
            *,
            loss_fn_config=None,
        ):
            assert loss_fn_config == {"temperature": 1.0}
            del sampling_masks
            return ForwardLogprobsResult(
                logprobs=[[-0.7, -0.8] for _ in tokens_batch],
                router_replay_record_ref=None,
            )

    trainer = _trainer("R2")
    trainer._service_holder = FakeWeaverServiceHolder()
    trajectory = _trajectory()

    with pytest.raises(ValueError, match="did not return a recorded routing ref"):
        run_coro(trainer._recompute_logprobs([trajectory], {}))

    assert trajectory["old_log_probs"] == [0.0, -0.1, -0.2]
    assert "router_replay_ref" not in trajectory.extra_fields


def test_r2_nested_sequence_refs_survive_parent_preserving_recompute_and_conversion():
    class Holder:
        async def compute_target_logprobs_batch_with_record(
            self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
        ):
            assert loss_fn_config == {"temperature": 1.0}
            assert tokens_batch == [[10, 11, 12], [20, 21], [30, 31]]
            return ForwardLogprobsResult(
                logprobs=[[-0.1, -0.2], [-0.3], [-0.4]],
                router_replay_record_ref=_record_ref("nested"),
            )

    trainer = _trainer("R2")
    trainer._service_holder = Holder()
    plain = _trajectory()
    nested_sequences = [
        {"tokens": [20, 21], "loss_mask": [0, 1], "old_log_probs": [0.0, -1.0]},
        {"tokens": [30, 31], "loss_mask": [0, 1], "old_log_probs": [0.0, -1.0]},
    ]
    nested = Trajectory(segments=nested_sequences, reward=0.0, extra_fields={"advantage": -1.0})
    parents = [plain, nested]

    prepared = run_coro(trainer._recompute_logprobs(parents, {}))
    datums = trainer._convert_to_datums(prepared)

    assert prepared is parents
    assert plain.segments is None
    assert nested.segments is nested_sequences
    assert "router_replay_ref" not in nested.extra_fields
    assert [datum["loss_fn_inputs"]["logprobs"] for datum in datums] == [
        [-0.1, -0.2],
        [-0.3],
        [-0.4],
    ]
    assert [
        datum["metadata"]["router_replay"]["value_ref"]["sample_index"] for datum in datums
    ] == [0, 1, 2]


@pytest.mark.parametrize("fail_second_chunk", [False, True])
@pytest.mark.parametrize("submit_ahead", [0, 1])
def test_r2_recompute_chunks_keep_successful_forward_refs_and_local_indices(
    fail_second_chunk, submit_ahead
):
    trajectories = [_trajectory() for _ in range(3)]

    class Holder:
        def __init__(self):
            self.batch_sizes = []

        async def compute_target_logprobs_batch_with_record(
            self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
        ):
            assert loss_fn_config == {"temperature": 1.0}
            assert sampling_masks is None
            self.batch_sizes.append(len(tokens_batch))
            call = len(self.batch_sizes)
            # Earlier chunks and failed attempts must remain uncommitted.
            for trajectory in trajectories:
                assert trajectory["old_log_probs"] == [0.0, -0.1, -0.2]
                assert "router_replay_ref" not in trajectory.extra_fields
            if fail_second_chunk and call == 2:
                raise RuntimeError("later forward failed")
            return ForwardLogprobsResult(
                logprobs=[[-float(call), -float(call)] for _ in tokens_batch],
                router_replay_record_ref=_record_ref(f"record-{call}"),
            )

    holder = Holder()
    trainer = _trainer("R2")
    trainer._service_holder = holder
    trainer._mini_batch_size = 2
    trainer._submit_ahead = submit_ahead

    prepared = run_coro(trainer._recompute_logprobs(trajectories, {}))
    datums = trainer._convert_to_datums(prepared)

    last_call = 3 if fail_second_chunk else 2
    assert holder.batch_sizes == ([2, 1, 1] if fail_second_chunk else [2, 1])
    assert [datum["metadata"]["router_replay"]["value_ref"] for datum in datums] == [
        {**_record_ref("record-1"), "sample_index": 0},
        {**_record_ref("record-1"), "sample_index": 1},
        {**_record_ref(f"record-{last_call}"), "sample_index": 0},
    ]
    assert [trajectory["old_log_probs"] for trajectory in prepared] == [
        [0.0, -float(call), -float(call)] for call in (1, 1, last_call)
    ]


@pytest.mark.parametrize("submit_ahead", [0, 1])
def test_r3_recompute_chunks_preserve_sampling_masks_and_rollout_refs(submit_ahead):
    refs = [{"relative_path": f"rollout/sequence-{index}.safetensors"} for index in range(3)]
    masks = [[[11, 20 + index], [12, 30 + index]] for index in range(3)]
    trajectories = [_trajectory(routing_ref=ref) for ref in refs]
    for trajectory, mask in zip(trajectories, masks):
        trajectory.extra_fields["sampling_mask"] = mask

    class Holder:
        def __init__(self):
            self.calls = []

        async def compute_target_logprobs_batch(
            self,
            tokens_batch,
            sampling_masks=None,
            datum_metadata_batch=None,
            *,
            loss_fn_config=None,
        ):
            assert loss_fn_config == {"temperature": 1.0}
            self.calls.append((tokens_batch, sampling_masks, datum_metadata_batch))
            return [[-0.3, -0.4] for _ in tokens_batch]

    holder = Holder()
    trainer = _trainer("R3")
    trainer._service_holder = holder
    trainer._mini_batch_size = 2
    trainer._submit_ahead = submit_ahead

    prepared = run_coro(trainer._recompute_logprobs(trajectories, {}))

    assert [len(call[0]) for call in holder.calls] == [2, 1]
    assert [call[1] for call in holder.calls] == [masks[:2], masks[2:]]
    assert [
        [metadata["router_replay"]["value_ref"] for metadata in call[2]] for call in holder.calls
    ] == [refs[:2], refs[2:]]
    datums = trainer._convert_to_datums(prepared)
    assert [datum["metadata"]["router_replay"]["value_ref"] for datum in datums] == refs
    assert [datum["loss_fn_inputs"]["sampling_mask"] for datum in datums] == masks
    assert [trajectory["old_log_probs"] for trajectory in prepared] == [[0.0, -0.3, -0.4]] * 3


@pytest.mark.parametrize("mini_batch_size", [1, 3])
def test_r2_chunk_record_indices_exclude_degenerate_sequences(mini_batch_size):
    first, last = _trajectory(), _trajectory()
    first.extra_fields["sampling_mask"] = [[11, 13], [12, 14]]
    last.extra_fields["sampling_mask"] = [[11, 15], [12, 16]]
    degenerate = Trajectory(
        tokens=[42],
        loss_mask=[0],
        reward=1.0,
        extra_fields={"advantage": 1.0, "old_log_probs": [0.0], "sampling_mask": []},
    )

    class Holder:
        def __init__(self):
            self.calls = []

        async def compute_target_logprobs_batch_with_record(
            self, tokens_batch, sampling_masks=None, *, loss_fn_config=None
        ):
            assert loss_fn_config == {"temperature": 1.0}
            assert tokens_batch and all(len(tokens) > 1 for tokens in tokens_batch)
            self.calls.append((tokens_batch, sampling_masks))
            return ForwardLogprobsResult(
                logprobs=[[-0.3, -0.4] for _ in tokens_batch],
                router_replay_record_ref=_record_ref(f"record-{len(self.calls)}"),
            )

    holder = Holder()
    trainer = _trainer("R2")
    trainer._service_holder = holder
    trainer._mini_batch_size = mini_batch_size

    prepared = run_coro(trainer._recompute_logprobs([first, degenerate, last], {}))

    assert [len(call[0]) for call in holder.calls] == ([1, 1] if mini_batch_size == 1 else [2])
    assert [mask for call in holder.calls for mask in call[1]] == [
        first["sampling_mask"],
        last["sampling_mask"],
    ]
    assert degenerate["old_log_probs"] == [0.0]
    assert "router_replay_ref" not in degenerate.extra_fields
    datums = trainer._convert_to_datums(prepared)
    assert [datum["metadata"]["router_replay"]["value_ref"] for datum in datums] == [
        {**_record_ref("record-1"), "sample_index": 0},
        {
            **_record_ref("record-2" if mini_batch_size == 1 else "record-1"),
            "sample_index": 0 if mini_batch_size == 1 else 1,
        },
    ]
