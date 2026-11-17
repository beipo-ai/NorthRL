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

"""Failure-path coverage for submitted Weaver training operations."""

from __future__ import annotations

import asyncio
import types as pytypes
from types import SimpleNamespace

import pytest

from northrl.utils.async_utils import run_coro
from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder


class _Operation:
    def __init__(self, *, value=None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error
        self.consumed = False

    async def result(self):
        self.consumed = True
        if self.error is not None:
            raise self.error
        return self.value


def test_collect_training_result_consumes_optimizer_when_forward_backward_fails():
    forward_backward = _Operation(error=RuntimeError("forward/backward failed"))

    async def _exercise_failure():
        optimizer_started = asyncio.Event()
        optimizer_release = asyncio.Event()

        class _BlockingOptimizer(_Operation):
            async def result(self):
                optimizer_started.set()
                await optimizer_release.wait()
                return await super().result()

        optimizer = _BlockingOptimizer(value={"metrics": {"learning_rate": 1e-5}})
        collection = asyncio.create_task(
            WeaverServiceHolder.collect_training_result((forward_backward, optimizer))
        )
        await optimizer_started.wait()
        await asyncio.sleep(0)
        assert not collection.done()
        optimizer_release.set()
        with pytest.raises(RuntimeError, match="forward/backward failed"):
            await collection
        return optimizer

    optimizer = run_coro(_exercise_failure())

    assert forward_backward.consumed
    assert optimizer.consumed


def test_collect_training_result_raises_optimizer_failure():
    forward_backward = _Operation(value={"metrics": {"loss": 0.5}})
    optimizer = _Operation(error=RuntimeError("optimizer failed"))

    with pytest.raises(RuntimeError, match="optimizer failed"):
        run_coro(WeaverServiceHolder.collect_training_result((forward_backward, optimizer)))

    assert forward_backward.consumed
    assert optimizer.consumed


def test_submit_forward_backward_defers_collection():
    operation = _Operation(
        value={
            "loss": 0.5,
            "metrics": {
                "loss": 0.5,
                "loss:sum": 2.0,
                "loss_agg_denominator": 4.0,
            },
        }
    )

    class _TrainingClient:
        async def forward_backward(self, datums, loss_fn, *, loss_fn_config, wait):
            assert datums == ["datum"]
            assert loss_fn == "cross_entropy"
            assert loss_fn_config == {"loss_agg_mode": "token-mean"}
            assert wait is False
            return operation

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._training_client = _TrainingClient()  # pylint: disable=protected-access
    holder._build_weaver_datums = lambda _data, **_kwargs: ["datum"]  # type: ignore[method-assign]

    async def _noop_ensure(self):
        del self

    holder._ensure_provisioned = pytypes.MethodType(  # type: ignore[method-assign]
        _noop_ensure, holder
    )

    datums_data = [{"input_tokens": [1, 2]}]
    handle = run_coro(
        holder.submit_forward_backward(
            datums_data=datums_data,
            loss_fn="cross_entropy",
            loss_fn_config={"loss_agg_mode": "token-mean"},
        )
    )
    assert handle is operation
    assert not operation.consumed

    result = run_coro(holder.collect_forward_backward_result(handle, datums_data=datums_data))
    assert operation.consumed
    assert result == {
        "loss": 0.5,
        "metrics": {
            "loss": 0.5,
            "loss:sum": 2.0,
            "loss_agg_denominator": 4.0,
        },
    }


def test_optimizer_submission_failure_consumes_submitted_forward_backward(monkeypatch):
    forward_backward = _Operation(value={"metrics": {"loss": 0.5}})

    class _TrainingClient:
        async def forward_backward(self, datums, loss_fn, *, loss_fn_config, wait):
            assert datums == ["datum"]
            assert loss_fn == "cross_entropy"
            assert loss_fn_config is None
            assert wait is False
            return forward_backward

        async def optim_step(self, adam_params, *, wait):
            assert adam_params["learning_rate"] == 2e-6
            assert wait is False
            raise RuntimeError("optimizer submission failed")

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._training_client = _TrainingClient()  # pylint: disable=protected-access
    holder._build_weaver_datums = lambda _data, **_kwargs: ["datum"]  # type: ignore[method-assign]

    async def _noop_ensure(self):
        del self

    holder._ensure_provisioned = pytypes.MethodType(  # type: ignore[method-assign]
        _noop_ensure, holder
    )
    monkeypatch.setattr(
        holder_module,
        "types",
        SimpleNamespace(AdamParams=lambda **kwargs: kwargs),
    )

    with pytest.raises(RuntimeError, match="optimizer submission failed"):
        run_coro(
            holder.forward_backward_and_optim_step(
                datums_data=[{"input_tokens": [1, 2]}],
                loss_fn="cross_entropy",
            )
        )

    assert forward_backward.consumed


def coupled_custom_loss(data, logprob_tensors):
    """A non-additive callback that requires the complete optimizer batch."""
    import torch

    loss = torch.stack([tensor.mean() for tensor in logprob_tensors]).mean().square()
    return loss, {"loss": float(loss.detach()), "num_samples": len(data)}


def _custom_training_holder(*, forward_error_at=None, backward_error_at=None):
    weaver_types = pytest.importorskip("weaver").types
    logprob_values = [[-0.2, -0.8], [-0.6, -0.1, -0.3], [-0.9]]
    datums = [
        weaver_types.Datum.from_raw(
            model_input=weaver_types.ModelInput.from_ints(list(range(len(logprobs)))),
            loss_fn_inputs={
                "target_tokens": list(range(1, len(logprobs) + 1)),
                "sampling_logprobs": [-0.5] * len(logprobs),
                "advantages": list(range(1, len(logprobs) + 1)),
                "loss_mask": [1] * len(logprobs),
                "sampling_mask": [[1]] * len(logprobs),
            },
            metadata={"case_index": index, "router_replay": {"mode": "replay"}},
        )
        for index, logprobs in enumerate(logprob_values)
    ]

    class _CustomTrainingClient:
        def __init__(self):
            self.calls = []
            self.surrogates = []
            self.forward_count = 0
            self.backward_count = 0
            self.operations = []

        async def forward(self, chunk, loss_fn, *, wait):
            assert loss_fn == "forward_logprob"
            assert wait is False
            self.forward_count += 1
            self.calls.append(("forward", len(chunk)))
            outputs = [
                {
                    "logprobs": logprob_values[datum.metadata["case_index"]],
                }
                for datum in chunk
            ]
            error = (
                RuntimeError("custom forward failed")
                if self.forward_count == forward_error_at
                else None
            )
            operation = _Operation(value={"result": {"loss_fn_outputs": outputs}}, error=error)
            self.operations.append(operation)
            return operation

        async def forward_backward(self, chunk, loss_fn, *, wait):
            assert loss_fn == "surrogate"
            assert wait is False
            self.backward_count += 1
            self.calls.append(("backward", len(chunk)))
            self.surrogates.extend(chunk)
            error = (
                RuntimeError("custom backward failed")
                if self.backward_count == backward_error_at
                else None
            )
            operation = _Operation(value={}, error=error)
            self.operations.append(operation)
            return operation

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._training_client = _CustomTrainingClient()  # pylint: disable=protected-access
    holder._build_weaver_datums = lambda _data, **_kwargs: datums  # type: ignore[method-assign]

    async def _noop_ensure():
        pass

    holder._ensure_provisioned = _noop_ensure  # type: ignore[method-assign]
    return holder, datums, logprob_values


@pytest.mark.parametrize(
    "loss_name",
    [
        "importance_sampling",
        "truncated_importance_sampling",
        "direct_logprob",
        "tests.unittests.weaver.test_training_results:coupled_custom_loss",
    ],
)
def test_custom_loss_chunks_match_whole_batch_gradients_and_metrics(loss_name):
    import torch

    from northrl.weaver.custom_losses import resolve_custom_loss

    holder, datums, logprob_values = _custom_training_holder()
    spec = {"name": loss_name, "params": {}}
    expected_tensors = [torch.tensor(values, requires_grad=True) for values in logprob_values]
    expected_loss, expected_metrics = resolve_custom_loss(spec)(datums, expected_tensors)
    expected_loss.backward()

    result = run_coro(
        holder.forward_backward_custom([{}] * len(datums), spec, mini_batch_sizes=[2, 1])
    )

    client = holder._training_client  # pylint: disable=protected-access
    assert client.calls == [("forward", 2), ("forward", 1), ("backward", 2), ("backward", 1)]
    assert all(operation.consumed for operation in client.operations)
    assert result["loss"] == pytest.approx(float(expected_loss.detach()))
    assert result["metrics"] == pytest.approx(expected_metrics)
    for original, surrogate, expected_tensor in zip(datums, client.surrogates, expected_tensors):
        assert surrogate.model_input == original.model_input
        assert surrogate.metadata == original.metadata
        torch.testing.assert_close(
            surrogate.loss_fn_inputs["sampling_mask"], original.loss_fn_inputs["sampling_mask"]
        )
        torch.testing.assert_close(
            surrogate.loss_fn_inputs["surrogate_weights"], expected_tensor.grad
        )
        assert "surrogate_weights" not in original.loss_fn_inputs


def test_custom_loss_forward_failure_never_submits_backward():
    holder, datums, _ = _custom_training_holder(forward_error_at=2)

    with pytest.raises(RuntimeError, match="custom forward failed"):
        run_coro(
            holder.forward_backward_custom(
                [{}] * len(datums),
                {"name": "direct_logprob"},
                mini_batch_sizes=[1, 1, 1],
            )
        )

    client = holder._training_client  # pylint: disable=protected-access
    assert client.calls == [("forward", 1), ("forward", 1)]
    assert all(operation.consumed for operation in client.operations)


def test_custom_loss_backward_failure_stops_remaining_chunks():
    holder, datums, _ = _custom_training_holder(backward_error_at=2)

    with pytest.raises(RuntimeError, match="custom backward failed"):
        run_coro(
            holder.forward_backward_custom(
                [{}] * len(datums),
                {"name": "direct_logprob"},
                mini_batch_sizes=[1, 1, 1],
            )
        )

    client = holder._training_client  # pylint: disable=protected-access
    assert client.calls == [("forward", 1)] * 3 + [("backward", 1)] * 2
    assert all(operation.consumed for operation in client.operations)


@pytest.mark.parametrize("sizes", [[], [0, 3], [-1, 4], [1, 1], [1.5, 1.5], [True, 2]])
def test_custom_loss_rejects_invalid_chunk_plan_before_remote_operations(sizes):
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)

    with pytest.raises(ValueError, match="mini_batch_sizes"):
        run_coro(
            holder.forward_backward_custom(
                [{}] * 3, {"name": "direct_logprob"}, mini_batch_sizes=sizes
            )
        )


def test_build_weaver_datums_uses_sample_ref_factory_without_placeholder_tokens(
    monkeypatch,
):
    raw_calls = []
    sample_ref_calls = []

    class _DatumFactory:
        @staticmethod
        def from_raw(**kwargs):
            raw_calls.append(kwargs)
            return ("raw", kwargs)

        @staticmethod
        def from_sample_ref(**kwargs):
            sample_ref_calls.append(kwargs)
            return ("sample_ref", kwargs)

    monkeypatch.setattr(
        holder_module,
        "types",
        SimpleNamespace(
            Datum=_DatumFactory,
            ModelInput=SimpleNamespace(from_ints=lambda tokens: {"tokens": tokens}),
        ),
    )
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)

    result = holder._build_weaver_datums(  # pylint: disable=protected-access
        [
            {
                "input_tokens": [1, 2],
                "datum_id": "local-1",
                "loss_fn_inputs": {"target_tokens": [2, 3], "loss_mask": [1, 1]},
            },
            {
                "kind": "sample_ref",
                "datum_id": "managed-1",
                "dataset": "hq-math",
                "version": "2026-08",
                "sample_idx": 9,
                "content_visibility": "protected",
                "loss_fn_inputs": {},
            },
        ]
    )

    assert [kind for kind, _ in result] == ["raw", "sample_ref"]
    assert raw_calls[0]["datum_id"] == "local-1"
    assert sample_ref_calls == [
        {
            "dataset": "hq-math",
            "version": "2026-08",
            "sample_idx": 9,
            "loss_fn_inputs": {},
            "metadata": {},
            "datum_id": "managed-1",
        }
    ]
    assert "model_input" not in sample_ref_calls[0]


def test_managed_length_resolution_preserves_and_validates_reference_order(monkeypatch):
    class _SampleRef:
        def __init__(self, *, dataset, version, sample_idx):
            self.dataset = dataset
            self.version = version
            self.sample_idx = sample_idx

    class _TrainingClient:
        async def resolve_sample_ref_lengths(self, refs):
            return [
                SimpleNamespace(
                    sample_ref=ref,
                    input_token_count=10 + ref.sample_idx,
                )
                for ref in refs
            ]

    async def _noop_ensure(self):
        del self

    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._training_client = _TrainingClient()  # pylint: disable=protected-access
    holder._ensure_provisioned = pytypes.MethodType(  # type: ignore[method-assign]
        _noop_ensure, holder
    )
    monkeypatch.setattr(holder_module, "types", SimpleNamespace(SampleRef=_SampleRef))
    refs = [
        {"dataset": "hq-math", "version": "2026-08", "sample_idx": 1},
        {"dataset": "hq-math", "version": "2026-08", "sample_idx": 4},
    ]

    result = run_coro(holder.resolve_managed_sample_ref_lengths(refs))

    assert result == [
        {
            "sample_ref": refs[0],
            "input_token_count": 11,
        },
        {
            "sample_ref": refs[1],
            "input_token_count": 14,
        },
    ]

    class _ReorderedTrainingClient:
        async def resolve_sample_ref_lengths(self, requested):
            return [
                SimpleNamespace(
                    sample_ref=ref,
                    input_token_count=10 + ref.sample_idx,
                )
                for ref in reversed(requested)
            ]

    holder._training_client = _ReorderedTrainingClient()  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="order/reference mismatch"):
        run_coro(holder.resolve_managed_sample_ref_lengths(refs))
