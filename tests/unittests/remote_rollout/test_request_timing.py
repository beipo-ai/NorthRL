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

"""Request journals retain native usage counts and cancelled request durations."""

import asyncio

import pytest

from northrl.remote_rollout.metrics import RolloutMetrics


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": 10, "completion_tokens": 2}, {"input_tokens": 10, "output_tokens": 2}],
)
def test_request_journal_reads_both_native_usage_formats(usage):
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)
    with metrics.track_request("native") as timing:
        timing.result = {"usage": usage}
    assert records[0]["prompt_tokens"] == 10
    assert records[0]["completion_tokens"] == 2
    assert records[0]["duration_seconds"] > 0
    assert metrics.snapshot()["model_requests_success"] == 1
    assert metrics.snapshot()["model_request_inflight"] == 0


def test_request_journal_can_count_exact_gateway_tokens_without_usage():
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)
    with metrics.track_request("native") as timing:
        timing.result = {"token_data": {"prompt_tokens": [1, 2], "response_tokens": [3, 4, 99]}}
    assert records[0]["prompt_tokens"] == 2
    assert records[0]["completion_tokens"] == 3
    assert metrics.snapshot()["request_stat_total_tokens_sum"] == 5


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError(), "error"),
        (TimeoutError(), "timeout"),
        (asyncio.CancelledError(), "cancelled"),
    ],
)
def test_request_journal_records_failed_and_cancelled_calls(error, status):
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)
    with pytest.raises(type(error)):
        with metrics.track_request("native"):
            raise error
    values = metrics.snapshot()
    assert values["model_requests_started"] == values["model_requests_completed"] == 1
    assert values[f"model_requests_{status}"] == 1
    assert values["model_request_seconds_sum"] > 0
    assert values["model_request_inflight"] == 0
    assert records[0]["status"] == status
    assert records[0]["completion_tokens"] is None
