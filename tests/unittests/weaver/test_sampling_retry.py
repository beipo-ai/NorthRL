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

import asyncio
import types as pytypes
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from weaver import WeaverAPIError

from northrl.exceptions import ContextLengthExceededError, WeightVersionMismatchError
from northrl.utils.async_utils import run_coro
from northrl.weaver import weaver_service_holder as holder_module
from northrl.weaver.weaver_service_holder import WeaverServiceHolder

SAMPLE_CALLS = [
    pytest.param(
        "sample_from_messages",
        {"messages": [{"role": "user", "content": "hello"}]},
        id="messages",
    ),
    pytest.param(
        "sample_from_prompt",
        {"prompt": "hello"},
        id="prompt",
    ),
    pytest.param(
        "sample_from_token_ids",
        {"input_ids": [1, 2]},
        id="token-ids",
    ),
]


class FakeTokenizer:
    def encode(self, prompt, add_special_tokens=False):  # pylint: disable=unused-argument
        return [1, 2]

    def decode(self, tokens, skip_special_tokens=True):  # pylint: disable=unused-argument
        return "decoded"


def _valid_sample_result() -> dict[str, Any]:
    return {
        "sequences": [
            {
                "tokens": [3],
                "logprobs": [-0.1],
                "text": "ok",
                "stop_reason": "stop",
                "weight_version": "version-1",
            }
        ]
    }


def _make_holder(side_effect: list[Any]) -> WeaverServiceHolder:
    holder = WeaverServiceHolder.__new__(WeaverServiceHolder)
    holder._tokenizer = FakeTokenizer()  # pylint: disable=protected-access
    operation = pytypes.SimpleNamespace(
        operation_id="operation-1",
        result=AsyncMock(side_effect=side_effect),
        refresh=AsyncMock(),
    )
    holder._sampling_client = pytypes.SimpleNamespace(  # pylint: disable=protected-access
        sample=AsyncMock(return_value=operation)
    )
    holder._max_concurrent_requests = None
    holder._sampling_semaphore = None
    holder._sampling_semaphore_loop = None
    holder._active_sampling_snapshot = (  # pylint: disable=protected-access
        "weaver://weights/version-1",
        "version-1",
    )
    holder._return_sampling_mask = False  # pylint: disable=protected-access
    holder._return_old_logprob = False  # pylint: disable=protected-access
    holder.apply_chat_template = AsyncMock(return_value=[1, 2])

    async def _noop_ensure(self, require_sampling=False):
        del self, require_sampling

    holder._ensure_provisioned = pytypes.MethodType(  # pylint: disable=protected-access
        _noop_ensure, holder
    )
    return holder


def _sample(holder: WeaverServiceHolder, method_name: str, kwargs: dict[str, Any]):
    method = getattr(holder, method_name)
    return run_coro(method(max_tokens=16, **kwargs))


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
@pytest.mark.parametrize(
    "invalid_result",
    [
        pytest.param({}, id="missing-sequences"),
        pytest.param({"sequences": []}, id="empty-sequences"),
        pytest.param({"sequences": [{}, {}]}, id="multiple-sequences"),
    ],
)
def test_sampling_retries_invalid_sequence_count_then_succeeds(
    monkeypatch,
    method_name,
    kwargs,
    invalid_result,
):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([invalid_result, _valid_sample_result()])

    result = _sample(holder, method_name, kwargs)

    assert result["response"] == "ok"
    assert result["weight_version"] == "version-1"
    holder._sampling_client.sample.assert_awaited_once()
    assert holder._sampling_client.sample.return_value.result.await_count == 2
    holder._sampling_client.sample.return_value.refresh.assert_awaited_once()
    assert all(
        sample_call.kwargs["num_samples"] == 1
        for sample_call in holder._sampling_client.sample.await_args_list
    )  # pylint: disable=protected-access
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
def test_sampling_raises_after_empty_sequences_exhaust_retries(
    monkeypatch,
    method_name,
    kwargs,
):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([{}, {"sequences": []}, {}])

    with pytest.raises(ValueError, match="sample result must contain exactly one sequence"):
        _sample(holder, method_name, kwargs)

    holder._sampling_client.sample.assert_awaited_once()
    assert holder._sampling_client.sample.return_value.result.await_count == 3
    assert holder._sampling_client.sample.return_value.refresh.await_count == 2
    assert sleep.await_args_list == [call(0.5), call(1.0)]


def test_sampling_retries_query_connection_error(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([ConnectionError("sampler unavailable"), _valid_sample_result()])

    result = _sample(
        holder,
        "sample_from_messages",
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert result["response"] == "ok"
    holder._sampling_client.sample.assert_awaited_once()
    assert holder._sampling_client.sample.return_value.result.await_count == 2
    sleep.assert_awaited_once_with(0.5)


def test_sampling_does_not_retry_non_retryable_error(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([RuntimeError("invalid sampling request")])

    with pytest.raises(RuntimeError, match="invalid sampling request"):
        _sample(
            holder,
            "sample_from_messages",
            {"messages": [{"role": "user", "content": "hello"}]},
        )

    assert holder._sampling_client.sample.await_count == 1  # pylint: disable=protected-access
    sleep.assert_not_awaited()


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
def test_sampling_rejects_multiple_requested_samples(method_name, kwargs):
    holder = _make_holder([])

    with pytest.raises(ValueError, match="only supports num_samples=1"):
        method = getattr(holder, method_name)
        run_coro(method(max_tokens=16, num_samples=2, **kwargs))

    holder._sampling_client.sample.assert_not_awaited()  # pylint: disable=protected-access


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
def test_sampling_does_not_retry_a_different_response_weight_version(
    monkeypatch, method_name, kwargs
):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    result = _valid_sample_result()
    result["sequences"][0]["weight_version"] = "wrong-version"
    holder = _make_holder([result])

    with pytest.raises(WeightVersionMismatchError, match="expected='version-1'.*wrong-version"):
        _sample(holder, method_name, kwargs)

    assert holder._sampling_client.sample.await_count == 1  # pylint: disable=protected-access
    sleep.assert_not_awaited()


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
@pytest.mark.parametrize(
    ("limit", "expected_max_tokens"),
    [(None, 16), (20, 16), (18, 16), (10, 8), (3, 1)],
    ids=["unconfigured", "under-limit", "exact-fit", "clamped", "one-token-left"],
)
def test_sampling_bounds_generation_by_actual_prompt_length(
    method_name, kwargs, limit, expected_max_tokens
):
    holder = _make_holder([_valid_sample_result()])
    holder._max_sequence_length = limit  # pylint: disable=protected-access

    result = _sample(holder, method_name, kwargs)

    assert result["prompt_tokens"] == [1, 2]
    assert result["response_tokens"] == [3]
    holder._sampling_client.sample.assert_awaited_once()  # pylint: disable=protected-access
    sampling_params = holder._sampling_client.sample.await_args.kwargs["sampling_params"]
    assert sampling_params.to_payload()["max_tokens"] == expected_max_tokens


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
@pytest.mark.parametrize("limit", [1, 2], ids=["prompt-over-limit", "no-room-left"])
def test_sampling_exhausted_context_does_not_submit_request(method_name, kwargs, limit):
    holder = _make_holder([])
    holder._max_sequence_length = limit  # pylint: disable=protected-access

    with pytest.raises(ContextLengthExceededError) as caught:
        _sample(holder, method_name, kwargs)

    assert caught.value.length_details == {
        "prompt_tokens": 2,
        "requested_max_tokens": 16,
        "max_sequence_length": limit,
    }
    assert not hasattr(caught.value, "rollout_status")
    holder._sampling_client.sample.assert_not_awaited()  # pylint: disable=protected-access


@pytest.mark.parametrize(("method_name", "kwargs"), SAMPLE_CALLS)
@pytest.mark.parametrize("configured_limit", [None, 20])
def test_sampling_normalizes_backend_context_rejection(
    monkeypatch, method_name, kwargs, configured_limit
):
    error = WeaverAPIError(
        400,
        "sequence_too_long",
        "sequence length exceeds model max_seq_len: prompt 2 + max_new_tokens 16 = 18 > 10",
        False,
    )
    holder = _make_holder([error])
    holder._max_sequence_length = configured_limit  # pylint: disable=protected-access
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)

    with pytest.raises(ContextLengthExceededError) as caught:
        _sample(holder, method_name, kwargs)

    assert caught.value.length_details == {
        "prompt_tokens": 2,
        "requested_max_tokens": 16,
        "max_sequence_length": 10,
    }
    assert caught.value.__cause__ is error
    holder._sampling_client.sample.assert_awaited_once()  # pylint: disable=protected-access
    sleep.assert_not_awaited()


@pytest.mark.parametrize(("status", "code"), [(400, "invalid_request"), (500, "sequence_too_long")])
def test_sampling_does_not_reclassify_unrelated_backend_errors(status, code):
    error = WeaverAPIError(status, code, "sequence_too_long: diagnostic text > 10", False)
    holder = _make_holder([error] * holder_module.SAMPLE_MAX_ATTEMPTS)
    holder._max_sequence_length = 20  # pylint: disable=protected-access

    with pytest.raises(WeaverAPIError) as caught:
        _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    assert caught.value is error
    holder._sampling_client.sample.assert_awaited_once()  # pylint: disable=protected-access


def test_sampling_context_rejection_uses_configured_limit_if_server_omits_it():
    error = WeaverAPIError(400, "sequence_too_long", "context window exceeded", False)
    holder = _make_holder([error])
    holder._max_sequence_length = 20  # pylint: disable=protected-access

    with pytest.raises(ContextLengthExceededError) as caught:
        _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    assert caught.value.max_sequence_length == 20


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_holder_rejects_invalid_context_limit(limit):
    with pytest.raises(ValueError, match="max_sequence_length must be a positive integer"):
        WeaverServiceHolder(max_sequence_length=limit)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("result stream interrupted"),
        httpx.ReadTimeout("result query timed out"),
        WeaverAPIError(503, "unavailable", "query unavailable", True),
    ],
)
def test_query_failure_retries_same_operation_without_resubmission(monkeypatch, error):
    monkeypatch.setattr(holder_module.asyncio, "sleep", AsyncMock())
    holder = _make_holder([error, {"result": _valid_sample_result()}])

    result = _sample(holder, "sample_from_token_ids", {"input_ids": [1, 2]})

    assert result["response"] == "ok"
    holder._sampling_client.sample.assert_awaited_once()
    assert holder._sampling_client.sample.call_args.kwargs["wait"] is False
    operation = holder._sampling_client.sample.return_value
    assert operation.result.await_count == 2
    operation.refresh.assert_awaited_once_with()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError("submission response lost"),
        httpx.ReadTimeout("submission timed out"),
        httpx.RemoteProtocolError("response disconnected"),
        ConnectionError("submission outcome unknown"),
        OSError("socket failed after write"),
    ],
)
def test_ambiguous_submission_is_not_retried_without_idempotency(monkeypatch, error):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([_valid_sample_result()])
    holder._sampling_client.sample.side_effect = error

    with pytest.raises(type(error)) as caught:
        _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    assert caught.value is error
    holder._sampling_client.sample.assert_awaited_once()
    holder._sampling_client.sample.return_value.result.assert_not_awaited()
    sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connection timed out"),
        httpx.PoolTimeout("pool busy"),
        WeaverAPIError(429, "rate_limit", "not admitted", True),
    ],
)
def test_rejected_or_unsent_submission_can_be_retried(monkeypatch, error):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([_valid_sample_result()])
    operation = holder._sampling_client.sample.return_value
    holder._sampling_client.sample.side_effect = [error, operation]

    assert _sample(holder, "sample_from_prompt", {"prompt": "hello"})["response"] == "ok"

    assert holder._sampling_client.sample.await_count == 2
    assert all(
        call.kwargs["wait"] is False for call in holder._sampling_client.sample.call_args_list
    )
    operation.result.assert_awaited_once()
    sleep.assert_awaited_once_with(0.5)


def test_submission_reuses_explicit_sdk_idempotency_key(monkeypatch):
    monkeypatch.setattr(holder_module.asyncio, "sleep", AsyncMock())
    holder = _make_holder([_valid_sample_result()])
    operation = holder._sampling_client.sample.return_value
    keys = []

    async def sample(*, idempotency_key, wait, **kwargs):
        assert wait is False
        keys.append(idempotency_key)
        if len(keys) == 1:
            raise httpx.ReadError("accepted but response lost")
        return operation

    holder._sampling_client.sample = sample

    assert _sample(holder, "sample_from_prompt", {"prompt": "hello"})["response"] == "ok"
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert keys[0]
    operation.result.assert_awaited_once()


def test_terminal_operation_failure_never_submits_again(monkeypatch):
    from weaver.operations import WeaverOperationError

    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    error = WeaverOperationError({"status": "error", "error_code": "generation_failed"})
    holder = _make_holder([error])

    with pytest.raises(WeaverOperationError) as caught:
        _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    assert caught.value is error
    holder._sampling_client.sample.assert_awaited_once()
    sleep.assert_not_awaited()


def test_retry_after_hint_is_respected_for_submission_rejection(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    error = WeaverAPIError(429, "rate_limit", "not admitted", True, retry_after=2.5)
    holder = _make_holder([_valid_sample_result()])
    operation = holder._sampling_client.sample.return_value
    holder._sampling_client.sample.side_effect = [error, operation]

    _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    sleep.assert_awaited_once_with(2.5)


def test_cancelled_query_is_not_retried_or_resubmitted(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    holder = _make_holder([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(holder.sample_from_prompt(prompt="hello", max_tokens=16))

    holder._sampling_client.sample.assert_awaited_once()
    holder._sampling_client.sample.return_value.refresh.assert_not_awaited()
    sleep.assert_not_awaited()


def test_real_sdk_handle_refreshes_malformed_cached_response(monkeypatch):
    from weaver.operations import AsyncOperationHandle

    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    http_client = pytypes.SimpleNamespace(
        get=AsyncMock(
            return_value={
                "id": "operation-real",
                "status": "done",
                "response": {"result": _valid_sample_result()},
            }
        )
    )
    handle = AsyncOperationHandle(
        client=http_client,
        operation_id="operation-real",
        _cached={"id": "operation-real", "status": "done", "response": {}},
    )
    holder = _make_holder([])
    holder._sampling_client.sample.return_value = handle

    assert _sample(holder, "sample_from_prompt", {"prompt": "hello"})["response"] == "ok"

    holder._sampling_client.sample.assert_awaited_once()
    http_client.get.assert_awaited_once_with("/api/v1/operations/operation-real")
    sleep.assert_awaited_once_with(0.5)


def test_request_id_alone_does_not_enable_ambiguous_submission_retries(monkeypatch):
    async def sample(*, request_id=None, **kwargs):
        raise httpx.ReadError("submission outcome unknown")

    client = pytypes.SimpleNamespace(sample=sample)
    holder = _make_holder([])
    sleep = AsyncMock()
    monkeypatch.setattr(holder_module.asyncio, "sleep", sleep)
    with pytest.raises(httpx.ReadError):
        asyncio.run(holder._sample_operation_with_retries(client))
    sleep.assert_not_awaited()


def test_real_sdk_handle_recovers_result_read_error_without_new_sample(monkeypatch):
    from weaver.operations import AsyncOperationHandle

    monkeypatch.setattr(holder_module.asyncio, "sleep", AsyncMock())
    http_client = pytypes.SimpleNamespace(
        get=AsyncMock(
            side_effect=[
                httpx.ReadError("query response interrupted"),
                {
                    "id": "operation-real",
                    "status": "done",
                    "response": {"result": _valid_sample_result()},
                },
            ]
        )
    )
    handle = AsyncOperationHandle(
        client=http_client,
        operation_id="operation-real",
        _cached={"id": "operation-real", "status": "pending"},
    )
    holder = _make_holder([])
    holder._sampling_client.sample.return_value = handle

    assert _sample(holder, "sample_from_prompt", {"prompt": "hello"})["response"] == "ok"

    holder._sampling_client.sample.assert_awaited_once()
    assert http_client.get.call_args_list == [
        call("/api/v1/operations/operation-real"),
        call("/api/v1/operations/operation-real"),
    ]


@pytest.mark.parametrize(
    "tokens,stop_reason,text", [([3], "stop", "decoded text"), ([], "abort", "")]
)
def test_raw_operation_decodes_missing_text_without_losing_opaque_fields(tokens, stop_reason, text):
    raw = _valid_sample_result()
    sequence = raw["sequences"][0]
    sequence.pop("text")
    sequence["tokens"] = tokens
    sequence["logprobs"] = [-0.1] * len(tokens)
    sequence["stop_reason"] = stop_reason
    mask_ref = {"uri": "weaver://masks/sample-1"}
    router_ref = {"uri": "weaver://routing/sample-1"}
    extra_data = {"head_size": 4}
    sequence.update(
        sampling_mask_ref=mask_ref, moe_topk_indices_ref=router_ref, custom_field=extra_data
    )
    holder = _make_holder([{"result": raw}])
    holder._tokenizer.decode = Mock(return_value=text)

    prepared = run_coro(holder._sample_with_retries(holder._sampling_client))

    assert prepared["text"] == text
    assert prepared["sampling_mask_ref"] is mask_ref
    assert prepared["moe_topk_indices_ref"] is router_ref
    assert prepared["custom_field"] is extra_data
    assert "text" not in sequence
    holder._tokenizer.decode.assert_called_once_with(tokens, skip_special_tokens=False)
    holder._sampling_client.sample.assert_awaited_once()


@pytest.mark.parametrize("tokens", [None, (3,), [True], ["3"]])
def test_missing_text_does_not_decode_invalid_tokens(tokens):
    raw = _valid_sample_result()
    sequence = raw["sequences"][0]
    sequence.pop("text")
    sequence["tokens"] = tokens
    holder = _make_holder([{"result": raw}])
    holder._tokenizer.decode = Mock()

    with pytest.raises(TypeError, match="sequence tokens"):
        _sample(holder, "sample_from_prompt", {"prompt": "hello"})

    holder._tokenizer.decode.assert_not_called()
