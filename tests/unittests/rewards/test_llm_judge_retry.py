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

import asyncio
import logging
import ssl
from datetime import datetime, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import openai
import pytest

from northrl.rewards import llm_judge_retry as retry

LOGGER = logging.getLogger("test.llm_judge_retry")


@pytest.fixture(autouse=True)
def backoff(monkeypatch):
    sleep = AsyncMock()
    jitter = MagicMock(return_value=1.5)
    monkeypatch.setattr(retry, "sleep", sleep)
    monkeypatch.setattr(retry.random, "uniform", jitter)
    return sleep, jitter


def _status_error(status, *, headers=None, body=None):
    request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers, request=request)
    return openai.APIStatusError("PRIVATE_ERROR_BODY", response=response, body=body)


def _wait(error, attempt=1, **kwargs):
    return asyncio.run(
        retry.wait_for_judge_retry(
            error,
            failed_attempt=attempt,
            judge_name="test",
            logger=LOGGER,
            elapsed_s=180.2,
            **kwargs,
        )
    )


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_http_errors_retry_with_backoff(status, backoff):
    sleep, jitter = backoff
    assert _wait(_status_error(status))
    sleep.assert_awaited_once_with(1.5)
    jitter.assert_called_once_with(1.0, 3.0)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 422])
def test_permanent_http_errors_do_not_wait(status, backoff):
    sleep, jitter = backoff
    assert not _wait(_status_error(status))
    sleep.assert_not_awaited()
    jitter.assert_not_called()


@pytest.mark.parametrize("code", ["insufficient_quota", "billing_hard_limit_reached"])
@pytest.mark.parametrize("nested", [False, True])
def test_quota_errors_are_not_retried(code, nested, backoff):
    body = {"code": code}
    if nested:
        body = {"error": body}
    assert not _wait(_status_error(429, body=body))
    backoff[0].assert_not_awaited()


@pytest.mark.parametrize("timeout", [False, True])
def test_network_timeout_and_connection_error_retry(timeout, backoff):
    request = httpx.Request("POST", "http://judge.test/v1")
    error = (
        openai.APITimeoutError(request=request)
        if timeout
        else openai.APIConnectionError(request=request)
    )
    assert _wait(error)
    backoff[0].assert_awaited_once_with(1.5)


@pytest.mark.parametrize(
    "cause",
    [ssl.SSLCertVerificationError("bad certificate"), RuntimeError("CERTIFICATE_VERIFY_FAILED")],
)
def test_wrapped_certificate_errors_do_not_retry(cause, backoff):
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://judge.test/v1"))
    error.__cause__ = cause
    assert not _wait(error)
    backoff[0].assert_not_awaited()


def test_cyclic_exception_context_does_not_loop(backoff):
    error = ValueError("invalid API response")
    error.__context__ = error
    assert _wait(error)
    backoff[0].assert_awaited_once()


@pytest.mark.parametrize("attempt,expected", [(1, 1.5), (2, 3.0), (3, 6.0), (4, 12.0)])
def test_exponential_delays(attempt, expected, backoff):
    assert _wait(ValueError("invalid verdict"), attempt)
    backoff[0].assert_awaited_once_with(expected)


@pytest.mark.parametrize("jitter", [1.0, 3.0])
@pytest.mark.parametrize("attempt", [1, 2, 3, 4])
def test_delay_range_boundaries(monkeypatch, backoff, jitter, attempt):
    backoff[1].return_value = jitter
    assert _wait(ValueError("invalid response"), attempt)
    backoff[0].assert_awaited_once_with(jitter * 2 ** (attempt - 1))


def test_exhausted_budget_does_not_sleep(backoff, caplog):
    assert not _wait(ValueError("invalid verdict"), 5, timeout_s=300.0)
    backoff[0].assert_not_awaited()
    backoff[1].assert_not_called()
    assert "attempts_exhausted" in caplog.text
    assert "attempt=5/5" in caplog.text
    assert "timeout_s=300.0" in caplog.text


@pytest.mark.parametrize("attempt", [0, 6])
def test_invalid_attempt_is_rejected(attempt, backoff):
    with pytest.raises(ValueError, match="attempt budget"):
        _wait(ValueError("invalid response"), attempt)
    backoff[0].assert_not_awaited()


@pytest.mark.parametrize("seconds", ["0", "0.5", "10", "60"])
def test_retry_after_waits_at_least_server_hint(seconds, backoff):
    assert _wait(_status_error(429, headers={"Retry-After": seconds}))
    backoff[0].assert_awaited_once_with(float(seconds) + 1.5)


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-1"])
def test_invalid_retry_after_uses_normal_backoff(value, backoff):
    assert _wait(_status_error(503, headers={"Retry-After": value}))
    backoff[0].assert_awaited_once_with(1.5)


def test_http_date_retry_after(monkeypatch, backoff):
    monkeypatch.setattr(retry.time, "time", lambda: 1000.0)
    future = format_datetime(datetime.fromtimestamp(1010.0, timezone.utc), usegmt=True)
    assert _wait(_status_error(503, headers={"Retry-After": future}))
    backoff[0].assert_awaited_once_with(11.5)


def test_past_http_date_does_not_add_delay(monkeypatch, backoff):
    monkeypatch.setattr(retry.time, "time", lambda: 1000.0)
    past = format_datetime(datetime.fromtimestamp(990.0, timezone.utc), usegmt=True)
    assert _wait(_status_error(503, headers={"Retry-After": past}))
    backoff[0].assert_awaited_once_with(1.5)


def test_long_retry_after_is_not_truncated_to_early_retry(backoff, caplog):
    assert not _wait(_status_error(503, headers={"Retry-After": "61"}))
    backoff[0].assert_not_awaited()
    backoff[1].assert_not_called()
    assert "retry_after_exceeds_budget" in caplog.text


def test_logs_include_diagnostics_but_not_credentials_or_error_body(caplog):
    assert _wait(
        _status_error(503),
        context={"sample_id": "sample-123", "judge_model": "test-model", "api_key": "SECRET_KEY"},
    )
    assert "attempt=1/5" in caplog.text
    assert "elapsed_s=180.200" in caplog.text
    assert "delay_s=1.500" in caplog.text
    assert "sample-123" in caplog.text
    assert "PRIVATE_ERROR_BODY" not in caplog.text
    assert "SECRET_KEY" not in caplog.text


@pytest.mark.parametrize("base_timeout", [7.5, 180.0, 600.0])
@pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5])
def test_only_final_attempt_overrides_configured_timeout(base_timeout, attempt):
    expected = 300.0 if attempt == 5 else base_timeout
    assert retry.judge_timeout_for_attempt(base_timeout, attempt) == expected


@pytest.mark.parametrize("attempt", [0, 6])
def test_timeout_selection_rejects_invalid_attempt(attempt):
    with pytest.raises(ValueError, match="attempt budget"):
        retry.judge_timeout_for_attempt(180.0, attempt)


def test_concurrent_answers_do_not_share_attempt_state(backoff):
    async def run():
        return await asyncio.gather(
            *[
                retry.wait_for_judge_retry(
                    ValueError("invalid response"),
                    failed_attempt=attempt,
                    judge_name="test",
                    logger=LOGGER,
                    elapsed_s=0.0,
                )
                for attempt in [1, 1, 2]
            ]
        )

    assert asyncio.run(run()) == [True, True, True]
    assert backoff[0].await_args_list == [call(1.5), call(1.5), call(3.0)]


def test_real_async_wait_yields_and_propagates_cancellation(monkeypatch):
    monkeypatch.setattr(retry, "sleep", asyncio.sleep)

    async def run():
        task = asyncio.create_task(
            retry.wait_for_judge_retry(
                ValueError("invalid response"),
                failed_attempt=1,
                judge_name="test",
                logger=LOGGER,
                elapsed_s=0.0,
            )
        )
        try:
            await asyncio.sleep(0)
            assert not task.done()
            # Another coroutine completes while this answer is backing off.
            await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
            assert not task.done()
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
