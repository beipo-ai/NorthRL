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

"""Keep upstream HTTP, stream errors, and private request diagnostics distinct."""

import asyncio
import json
from itertools import count

import httpx
import pytest

from northrl.exceptions import ContextLengthExceededError
from northrl.remote_rollout.metrics import RolloutMetrics
from northrl.remote_rollout.northgate_relay import (
    NorthGateRelay,
    _CallObservation,
    _response_request_id,
    _StreamObservation,
    _UpstreamFailure,
)


def _sse(event):
    return b"data: " + json.dumps(event).encode() + b"\n\n"


def _observed_call():
    relay = object.__new__(NorthGateRelay)
    relay._call_ids = count()
    relay._failed_calls = {}
    relay._observations = set()
    relay._metrics = RolloutMetrics()
    calls = []
    relay._on_service_call = lambda operation, error: calls.append((operation, error))
    payload = {"messages": [{"role": "user", "content": "private-prompt"}], "max_tokens": 17}
    observation = _CallObservation(relay, "openai", "/v1/chat/completions", payload)
    return relay, observation, calls, payload


def _failure_log(caplog):
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("NorthGate request failed: ")
    ]
    assert len(messages) == 1
    return json.loads(messages[0].split(": ", 1)[1])


def test_http_error_preserves_actual_status_and_header_request_id():
    failure = _UpstreamFailure(
        503,
        {"request_id": "body-id", "error": {"code": "overloaded", "status": 429}},
        request_id="header-id",
    )
    assert failure.status == failure.http_status == 503
    assert failure.source == "http"
    assert failure.sse_status is None
    assert failure.code == "overloaded"
    assert failure.request_id == "header-id"
    assert "HTTP 503" in str(failure)


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("reported_status", [503, "503"])
def test_sse_error_preserves_http_200_and_separate_service_status(api, reported_status):
    reader = _StreamObservation(api, http_status=200, request_id="header-id")
    with pytest.raises(_UpstreamFailure) as caught:
        reader.observe(
            _sse(
                {
                    "type": "error",
                    "request_id": "body-id",
                    "error": {"code": "overloaded", "status": reported_status},
                }
            )
        )
    failure = caught.value
    assert failure.status == failure.http_status == 200
    assert failure.sse_status == 503
    assert failure.source == "sse"
    assert failure.code == "overloaded"
    assert failure.request_id == "header-id"
    assert "HTTP 400" not in str(failure)


@pytest.mark.parametrize("status", [None, True, "invalid", 0, 700])
def test_sse_missing_or_invalid_status_remains_unknown(status):
    reader = _StreamObservation("anthropic", http_status=200)
    with pytest.raises(_UpstreamFailure) as caught:
        reader.observe(_sse({"error": {"type": "api_error", "status": status}}))
    assert caught.value.http_status == 200
    assert caught.value.sse_status is None
    assert caught.value.code == "api_error"


def test_sse_without_http_metadata_does_not_invent_status():
    reader = _StreamObservation("openai")
    with pytest.raises(_UpstreamFailure) as caught:
        reader.observe(_sse({"status_code": 503, "request_id": "body-id", "error": "unavailable"}))
    assert caught.value.http_status is None
    assert caught.value.sse_status == 503
    assert caught.value.request_id == "body-id"


@pytest.mark.parametrize("header", ["x-northgate-request-id", "x-request-id", "request-id"])
def test_response_request_id_supported_headers(header):
    assert _response_request_id(httpx.Headers({header: "request-123"})) == "request-123"


def test_response_request_id_prefers_gateway_id():
    assert (
        _response_request_id(
            httpx.Headers({"x-northgate-request-id": "gateway", "x-request-id": "backend"})
        )
        == "gateway"
    )
    assert _response_request_id(httpx.Headers()) is None


@pytest.mark.parametrize("body", [{"request_id": "body"}, {"error": {"request_id": "body"}}])
def test_http_error_body_request_id_fallback(body):
    assert _UpstreamFailure(400, body).request_id == "body"


@pytest.mark.parametrize("context_limit", [False, True])
@pytest.mark.parametrize("partial_output", [False, True])
def test_stream_failure_logs_real_status_and_keeps_context_classification(
    context_limit, partial_output, caplog
):
    async def scenario():
        relay, observation, calls, payload = _observed_call()
        code = "context_length_exceeded" if context_limit else "overloaded"
        event = {
            "error": {
                "code": code,
                "status": 400 if context_limit else 503,
                "message": "private-upstream-body",
            }
        }

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                if partial_output:
                    yield _sse({"choices": [{"delta": {"content": "partial"}}]})
                yield _sse(event)

        upstream = httpx.Response(
            200,
            stream=Stream(),
            headers={"x-request-id": "service-id", "authorization": "private-token"},
        )
        expected = ContextLengthExceededError if context_limit else _UpstreamFailure
        with pytest.raises(expected) as caught:
            async for _ in relay._stream(upstream, "openai", payload, observation, "northrl"):
                pass
        assert upstream.is_closed
        assert len(calls) == 1 and calls[0][1] is caught.value
        assert relay._metrics.snapshot()["model_requests_error"] == 1
        diagnostic = _failure_log(caplog)
        assert diagnostic["api"] == "openai"
        assert diagnostic["path"] == "/v1/chat/completions"
        assert diagnostic["source"] == "sse"
        assert diagnostic["http_status"] == 200
        assert diagnostic["sse_status"] == (400 if context_limit else 503)
        assert diagnostic["error_code"] == code
        assert diagnostic["request_id"] == "service-id"
        assert diagnostic["response_emitted"] == partial_output
        assert diagnostic["elapsed_seconds"] >= 0
        assert len(diagnostic["payload_sha256"]) == 64
        for private in ("private-prompt", "private-upstream-body", "private-token"):
            assert private not in caplog.text

    asyncio.run(scenario())


def test_transport_failure_does_not_reuse_previous_http_error(caplog):
    relay, observation, calls, _ = _observed_call()
    observation.remember_failure(_UpstreamFailure(503, {"error": {"code": "old"}}))
    observation.http_status = None
    observation.request_id = None
    error = httpx.ConnectTimeout("private-network-error")
    observation.finish(error)
    diagnostic = _failure_log(caplog)
    assert diagnostic["source"] == "transport"
    assert diagnostic["http_status"] is None
    assert diagnostic["sse_status"] is None
    assert diagnostic["error_code"] is None
    assert diagnostic["request_id"] is None
    assert calls == [(observation.operation, error)]
    assert "private-network-error" not in caplog.text
    assert relay._metrics.snapshot()["model_requests_timeout"] == 1


@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
@pytest.mark.parametrize("partial_output", [False, True])
def test_response_body_transport_failure_logs_header_request_id(error_type, partial_output, caplog):
    async def scenario():
        relay, observation, calls, payload = _observed_call()
        error = error_type("private-read-failure")

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                if partial_output:
                    yield _sse({"choices": [{"delta": {"content": "partial"}}]})
                raise error

        upstream = httpx.Response(
            200, stream=Stream(), headers={"x-request-id": "successful-http-request"}
        )
        with pytest.raises(error_type) as caught:
            async for _ in relay._stream(upstream, "openai", payload, observation, "northrl"):
                pass
        assert caught.value is error
        assert error.__cause__ is None
        assert observation._upstream_failure is None
        assert upstream.is_closed
        assert calls == [(observation.operation, error)]
        diagnostic = _failure_log(caplog)
        assert diagnostic["source"] == "transport"
        assert diagnostic["error_type"] == error_type.__name__
        assert diagnostic["http_status"] == 200
        assert diagnostic["request_id"] == "successful-http-request"
        assert diagnostic["sse_status"] is None
        assert diagnostic["error_code"] is None
        assert diagnostic["response_emitted"] == partial_output
        assert "private-read-failure" not in caplog.text

    asyncio.run(scenario())


def test_retry_deadline_preserves_explicit_upstream_cause(caplog):
    relay, observation, calls, _ = _observed_call()
    failure = _UpstreamFailure(429, {"error": {"code": "rate_limit"}}, request_id="limited-request")
    error = TimeoutError("retry delay exceeds remaining budget")
    error.__cause__ = failure
    observation.finish(error)
    diagnostic = _failure_log(caplog)
    assert diagnostic["error_type"] == "TimeoutError"
    assert diagnostic["source"] == "http"
    assert diagnostic["http_status"] == 429
    assert diagnostic["error_code"] == "rate_limit"
    assert diagnostic["request_id"] == "limited-request"
    assert calls == [(observation.operation, error)]
    assert relay._metrics.snapshot()["model_requests_timeout"] == 1


def test_failure_diagnostics_do_not_follow_implicit_exception_context(caplog):
    _, observation, _, _ = _observed_call()
    error = TimeoutError("retry budget exhausted")
    error.__context__ = _UpstreamFailure(503, {"error": {"code": "previous-attempt"}})
    observation.finish(error)
    diagnostic = _failure_log(caplog)
    assert diagnostic["http_status"] is None
    assert diagnostic["error_code"] is None


def test_logger_failure_preserves_service_failure_and_matching_recovery(monkeypatch):
    relay, first, calls, payload = _observed_call()

    def broken_logger(*_args, **_kwargs):
        raise RuntimeError("logger broken")

    monkeypatch.setattr("northrl.remote_rollout.northgate_relay.logger.warning", broken_logger)
    error = _UpstreamFailure(400, {"error": {"code": "invalid_request"}})
    first.finish(error)
    first.finish(error)
    assert calls == [(first.operation, error)]
    assert relay._failed_calls[first.key] == {first.operation}
    recovered = _CallObservation(relay, "openai", "/v1/chat/completions", payload)
    recovered.finish()
    assert calls == [(first.operation, error), (first.operation, None), (recovered.operation, None)]
    assert relay._failed_calls == {}
    assert relay._observations == set()
