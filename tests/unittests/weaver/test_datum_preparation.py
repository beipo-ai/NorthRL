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

"""Regression coverage for preparing training data outside the shared SDK loop."""

from __future__ import annotations

import asyncio
import logging
import threading
from array import array
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from northrl.utils.async_utils import flush_sdk_logs, run_coro
from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder

_SDK_TYPES = holder_module.types
_MISSING_LOSS_INPUTS = object()
_PREPARATION_PATHS = ("forward", "backward", "custom", "backward_and_optim")


def _record_envelope():
    uri = "weaver://model/router-replay/record-1"
    return {
        "schema": "weaver.router_replay.index_set.v1",
        "transport": "gpfs_safetensors",
        "uri": uri,
        "manifest_uri": f"{uri}/manifest.json",
    }


class _FakeDatum:
    @staticmethod
    def from_raw(**kwargs):
        return SimpleNamespace(kind="raw", **kwargs)

    @staticmethod
    def from_sample_ref(**kwargs):
        return SimpleNamespace(kind="sample_ref", **kwargs)


class _FakeRouterReplayMetadata:
    @staticmethod
    def r2_record(*, fail_fast):
        return SimpleNamespace(
            to_payload=lambda: {
                "schema": "weaver.router_replay.datum.v1",
                "mode": "R2",
                "source": "recompute",
                "fail_fast": fail_fast,
                "action": "RECORD",
            }
        )


class _RecordingClient:
    def __init__(self):
        self.calls = []
        self.thread_ids = []
        self.forward_outputs = None

    async def forward(self, datums, loss_fn, *, wait):
        self.calls.append(("forward", datums))
        self.thread_ids.append(threading.get_ident())
        assert loss_fn == "forward_logprob"
        assert wait is False
        outputs = self.forward_outputs
        if outputs is None:
            outputs = [
                [-float(index + 1)] * len(datum.loss_fn_inputs["target_tokens"])
                for index, datum in enumerate(datums)
            ]
        return {
            "result": {
                "loss_fn_outputs": [{"logprobs": output} for output in outputs],
                "metrics": {"router_replay_recorded_indices": _record_envelope()},
            }
        }

    async def forward_backward(self, datums, loss_fn, *, loss_fn_config=None, wait=False):
        self.calls.append(("backward", datums))
        self.thread_ids.append(threading.get_ident())
        assert wait is False
        return {"metrics": {"loss": 0.5}}

    async def forward_backward_custom(self, datums, callback):
        self.calls.append(("custom", datums))
        self.thread_ids.append(threading.get_ident())
        return {"loss": 0.5, "metrics": {"loss": 0.5}}

    async def optim_step(self, adam_params, *, wait):
        self.calls.append(("optim", adam_params))
        self.thread_ids.append(threading.get_ident())
        assert wait is False
        return {"metrics": {}}


@pytest.fixture
def holder(monkeypatch):
    monkeypatch.setattr(
        holder_module,
        "types",
        SimpleNamespace(
            Datum=_FakeDatum,
            ModelInput=SimpleNamespace(from_ints=lambda tokens: tuple(tokens)),
            AdamParams=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setattr(holder_module, "RouterReplayMetadata", _FakeRouterReplayMetadata)
    monkeypatch.setattr(holder_module, "parse_custom_loss_spec", lambda spec: spec)
    monkeypatch.setattr(holder_module, "resolve_custom_loss", lambda spec: object())
    instance = WeaverServiceHolder.__new__(WeaverServiceHolder)
    instance._closed = False
    instance._training_client = _RecordingClient()
    instance._ensure_provisioned = AsyncMock()
    return instance


def _raw_datum():
    return {
        "input_tokens": [10, 11],
        "datum_id": "raw-1",
        "loss_fn_inputs": {"target_tokens": [11, 12], "loss_mask": [1, 1]},
        "metadata": {"router_replay": {"mode": "R3", "value_ref": {"uri": "routing"}}},
    }


def _invoke(holder, path, datums_data=None):
    if path == "forward":
        return holder.compute_target_logprobs_batch([[10, 11, 12]])
    data = [_raw_datum()] if datums_data is None else datums_data
    if path == "backward":
        return holder.submit_forward_backward(data)
    if path == "custom":
        return holder.forward_backward_custom(data, {"name": "test_loss"})
    return holder.forward_backward_and_optim_step(data)


def _builder_name(path):
    return "_build_forward_datums" if path == "forward" else "_build_weaver_datums"


@pytest.mark.parametrize("compact_output", [False, True])
@pytest.mark.parametrize("routing_mode", ["R2", "R3"])
@pytest.mark.parametrize("mask_transport", ["inline", "ref"])
def test_forward_preserves_empty_sequences_masks_and_routing(
    holder, routing_mode, mask_transport, compact_output
):
    tokens = [[], [10, 11, 12], [30], [20, 21]]
    if compact_output:
        tokens = [array("q", seq) for seq in tokens]
        original_forward = holder._training_client.forward

        async def forward(datums, loss_fn, *, wait, tensor_output_format="list"):
            assert tensor_output_format == "torch"
            response = await original_forward(datums, loss_fn, wait=wait)
            for output in response["result"]["loss_fn_outputs"]:
                output["logprobs"] = {"data": torch.tensor(output["logprobs"])}
            return response

        holder._training_client.forward = forward
    if mask_transport == "inline":
        masks = [None, [[11], [12]], None, [[21]]]
    else:
        masks = [
            None,
            {"sampling_mask_ref": {"uri": "mask-a"}},
            None,
            {"sampling_mask_ref": {"uri": "mask-b"}},
        ]
    metadata = [
        {
            "router_replay": {
                "mode": "R3",
                "action": "REPLAY",
                "value_ref": {"uri": f"routing-{index}"},
            },
            "sequence_index": index,
        }
        for index in range(len(tokens))
    ]
    original = deepcopy((tokens, masks, metadata))
    if routing_mode == "R2":
        result = run_coro(holder.compute_target_logprobs_batch_with_record(tokens, masks))
        outputs = result.logprobs
        assert result.router_replay_record_ref["index_set_uri"] == _record_envelope()["uri"]
    else:
        outputs = run_coro(holder.compute_target_logprobs_batch(tokens, masks, metadata))
    assert [list(values) for values in outputs] == [[], [-1.0, -1.0], [], [-2.0]]
    if compact_output:
        assert isinstance(outputs[1], array) and isinstance(outputs[3], array)
    datums = holder._training_client.calls[0][1]
    assert [datum.model_input for datum in datums] == [(10, 11), (20,)]
    assert [datum.loss_fn_inputs["target_tokens"].tolist() for datum in datums] == [[11, 12], [21]]
    for datum, index in zip(datums, [1, 3], strict=True):
        if routing_mode == "R2":
            assert (
                datum.metadata["router_replay"]
                == _FakeRouterReplayMetadata.r2_record(fail_fast=True).to_payload()
            )
        else:
            assert datum.metadata["router_replay"] == metadata[index]["router_replay"]
            assert datum.metadata["sequence_index"] == index
        if mask_transport == "inline":
            assert datum.loss_fn_inputs["sampling_mask"] == masks[index]
            assert "sampling_mask_ref" not in datum.metadata
        else:
            assert datum.metadata["sampling_mask_ref"] == masks[index]["sampling_mask_ref"]
            assert "sampling_mask" not in datum.loss_fn_inputs
    assert (tokens, masks, metadata) == original


def test_empty_forward_does_not_submit_a_remote_operation(holder):
    assert run_coro(holder.compute_target_logprobs_batch([[], [1]])) == [[], []]
    with pytest.raises(ValueError, match="non-degenerate"):
        run_coro(holder.compute_target_logprobs_batch_with_record([[], [1]]))
    assert holder._training_client.calls == []


def test_forward_rejects_partial_masks_before_remote_submission(holder):
    with pytest.raises(ValueError, match="every item"):
        run_coro(
            holder.compute_target_logprobs_batch([[1, 2], [3, 4]], sampling_masks=[[[2]], None])
        )
    assert holder._training_client.calls == []


@pytest.mark.parametrize("path", _PREPARATION_PATHS[1:])
@pytest.mark.parametrize(
    "sample_loss_inputs",
    [
        pytest.param({"advantages": [0.2]}, id="provided"),
        pytest.param(None, id="null"),
        pytest.param(_MISSING_LOSS_INPUTS, id="omitted"),
    ],
)
def test_backward_preparation_preserves_raw_and_sample_references(holder, path, sample_loss_inputs):
    data = [
        _raw_datum(),
        {
            "kind": "sample_ref",
            "dataset": "managed-math",
            "version": "v1",
            "sample_idx": 9,
            "datum_id": "managed-1",
            "loss_fn_inputs": sample_loss_inputs,
            "metadata": {"sampling_mask_ref": {"uri": "mask"}},
        },
    ]
    if sample_loss_inputs is _MISSING_LOSS_INPUTS:
        del data[1]["loss_fn_inputs"]
    original = deepcopy(data)
    run_coro(_invoke(holder, path, data))
    raw, referenced = holder._training_client.calls[0][1]
    assert raw.kind == "raw"
    assert raw.model_input == (10, 11)
    assert raw.datum_id == "raw-1"
    assert {field: values.tolist() for field, values in raw.loss_fn_inputs.items()} == data[0][
        "loss_fn_inputs"
    ]
    assert raw.metadata == data[0]["metadata"]
    assert referenced.kind == "sample_ref"
    assert referenced.dataset == "managed-math"
    assert referenced.version == "v1"
    assert referenced.sample_idx == 9
    assert referenced.datum_id == "managed-1"
    assert referenced.loss_fn_inputs == (data[1].get("loss_fn_inputs") or {})
    assert referenced.metadata == data[1]["metadata"]
    assert not hasattr(referenced, "model_input")
    assert data == original


@pytest.mark.parametrize("path", _PREPARATION_PATHS)
def test_preparation_leaves_shared_sdk_loop_available_for_sampling(holder, monkeypatch, path):
    builder_name = _builder_name(path)
    original_builder = getattr(holder, builder_name)
    release = threading.Event()
    builder_threads = []

    async def scenario():
        loop = asyncio.get_running_loop()
        sdk_thread = threading.get_ident()
        entered = asyncio.Event()

        def blocked_builder(*args, **kwargs):
            builder_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise RuntimeError("SDK loop could not release the data-preparation thread")
            return original_builder(*args, **kwargs)

        monkeypatch.setattr(holder, builder_name, blocked_builder)
        preparation = asyncio.create_task(_invoke(holder, path))
        try:
            await asyncio.wait_for(entered.wait(), 2)

            async def fake_sampling_request():
                await asyncio.sleep(0)
                return threading.get_ident()

            sampling_thread = await asyncio.wait_for(fake_sampling_request(), 2)
            assert not preparation.done()
            assert holder._training_client.calls == []
            assert sampling_thread == sdk_thread
            assert len(builder_threads) == 1
            assert builder_threads[0] != sdk_thread
            release.set()
            await asyncio.wait_for(preparation, 3)
            assert all(thread == sdk_thread for thread in holder._training_client.thread_ids)
        finally:
            release.set()
            if not preparation.done():
                preparation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await preparation

    run_coro(scenario())


@pytest.mark.parametrize("path", _PREPARATION_PATHS)
def test_cancelled_preparation_never_submits_a_remote_operation(holder, monkeypatch, path):
    builder_name = _builder_name(path)
    original_builder = getattr(holder, builder_name)
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        finished = asyncio.Event()

        def blocked_builder(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            try:
                if not release.wait(3):
                    raise RuntimeError("cancelled data-preparation thread was not released")
                return original_builder(*args, **kwargs)
            finally:
                loop.call_soon_threadsafe(finished.set)

        monkeypatch.setattr(holder, builder_name, blocked_builder)
        preparation = asyncio.create_task(_invoke(holder, path))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            preparation.cancel()
            await asyncio.sleep(0)
            assert not preparation.done()
            preparation.cancel()
            await asyncio.sleep(0)
            assert not preparation.done()
            assert holder._training_client.calls == []
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(preparation, 2)
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 3)
        # The cancelled worker may finish locally, but must never resume HTTP submission.
        await asyncio.sleep(0)
        assert holder._training_client.calls == []

    run_coro(scenario())


@pytest.mark.parametrize("path", _PREPARATION_PATHS)
def test_holder_closed_during_preparation_does_not_submit(holder, monkeypatch, path):
    builder_name = _builder_name(path)
    original_builder = getattr(holder, builder_name)
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def blocked_builder(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(3):
                raise RuntimeError("closed-holder preparation thread was not released")
            return original_builder(*args, **kwargs)

        monkeypatch.setattr(holder, builder_name, blocked_builder)
        preparation = asyncio.create_task(_invoke(holder, path))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            holder._closed = True
            release.set()
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(preparation, 2)
            assert holder._training_client.calls == []
        finally:
            release.set()
            if not preparation.done():
                preparation.cancel()
            await asyncio.gather(preparation, return_exceptions=True)

    run_coro(scenario())


@pytest.mark.parametrize("cancel_first", [False, True])
def test_preparation_gate_bounds_running_workers_during_cancellation(
    holder, monkeypatch, cancel_first
):
    original_builder = holder._build_weaver_datums
    releases = [threading.Event(), threading.Event()]
    running = 0
    maximum_running = 0
    starts = 0
    state_lock = threading.Lock()

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = [asyncio.Event(), asyncio.Event()]
        second_started = asyncio.Event()

        def blocked_builder(*args, **kwargs):
            nonlocal running, maximum_running, starts
            with state_lock:
                index = starts
                starts += 1
                running += 1
                maximum_running = max(maximum_running, running)
            loop.call_soon_threadsafe(entered[index].set)
            try:
                if not releases[index].wait(3):
                    raise RuntimeError("data-preparation gate test did not release its worker")
                return original_builder(*args, **kwargs)
            finally:
                with state_lock:
                    running -= 1

        monkeypatch.setattr(holder, "_build_weaver_datums", blocked_builder)
        first = asyncio.create_task(_invoke(holder, "backward"))
        second = None
        try:
            await asyncio.wait_for(entered[0].wait(), 2)
            if cancel_first:
                first.cancel()
                await asyncio.sleep(0)
                first.cancel()
                await asyncio.sleep(0)
                assert not first.done()

            async def second_preparation():
                second_started.set()
                return await _invoke(holder, "backward")

            second = asyncio.create_task(second_preparation())
            await asyncio.wait_for(second_started.wait(), 2)
            # The second call must remain queued while the first worker owns its inputs.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(entered[1].wait(), 0.05)
            releases[0].set()
            if cancel_first:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(first, 2)
            else:
                await asyncio.wait_for(first, 2)
            await asyncio.wait_for(entered[1].wait(), 2)
            releases[1].set()
            await asyncio.wait_for(second, 2)
            assert maximum_running == 1
            assert len(holder._training_client.calls) == (1 if cancel_first else 2)
        finally:
            for release in releases:
                release.set()
            tasks = [task for task in [first, second] if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    run_coro(scenario())


@pytest.mark.parametrize("path", _PREPARATION_PATHS)
def test_preparation_failure_propagates_without_remote_submission(holder, monkeypatch, path):
    def invalid_builder(*args, **kwargs):
        raise ValueError("invalid preparation input")

    monkeypatch.setattr(holder, _builder_name(path), invalid_builder)
    with pytest.raises(ValueError, match="invalid preparation input"):
        run_coro(_invoke(holder, path))
    assert holder._training_client.calls == []


@pytest.mark.skipif(_SDK_TYPES is None, reason="Weaver SDK is not installed")
def test_forward_preparation_accepts_real_sdk_datum_types(holder, monkeypatch):
    monkeypatch.setattr(holder_module, "types", _SDK_TYPES)
    holder._training_client.forward_outputs = [[-0.1, -0.2]]
    outputs = run_coro(
        holder.compute_target_logprobs_batch(
            [[10, 11, 12]], datum_metadata_batch=[{"sequence_index": 0}]
        )
    )
    assert outputs == [[-0.1, -0.2]]
    datum = holder._training_client.calls[0][1][0]
    assert isinstance(datum, _SDK_TYPES.Datum)
    assert isinstance(datum.model_input, _SDK_TYPES.ModelInput)
    assert datum.metadata == {"sequence_index": 0}


@pytest.mark.parametrize("raises", [False, True])
def test_preparation_reports_separate_wait_wall_and_worker_cpu_timings(
    holder, monkeypatch, caplog, raises
):
    # These clocks are local to the holder module, so asyncio keeps using its
    # real clock. The values distinguish every timing boundary without sleeps.
    wall_ticks = iter([10.0, 13.0, 17.0, 22.0, 26.0])
    cpu_ticks = iter([100.0, 103.0])
    monkeypatch.setattr(
        holder_module,
        "time",
        SimpleNamespace(
            perf_counter=lambda: next(wall_ticks),
            thread_time=lambda: next(cpu_ticks),
        ),
    )
    caplog.set_level(logging.INFO, logger=holder_module.__name__)
    expected = object()

    def builder():
        if raises:
            raise ValueError("invalid preparation input")
        return expected

    operation = holder._prepare_datums(builder, sequence_count=3, token_count=17)
    if raises:
        with pytest.raises(ValueError, match="invalid preparation input"):
            run_coro(operation)
    else:
        assert run_coro(operation) is expected

    flush_sdk_logs()
    records = [
        record.message for record in caplog.records if "[datum_preparation]" in record.message
    ]
    assert len(records) == 1
    message = records[0]
    assert "sequences=3 tokens=17 duration_seconds=13.000 cancelled=False" in message
    assert "lock_wait_seconds=3.000000" in message
    assert "worker_queue_seconds=4.0" in message
    assert "builder_wall_seconds=5.0" in message
    assert "builder_thread_cpu_seconds=3.0" in message
    assert holder._training_client.calls == []
