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

"""Tests for rollout observations independent of successful trajectory recording."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout.metrics import (
    TOOL_FORMAT_REASONS,
    RolloutMetrics,
    runtime_metrics_from_snapshot,
)
from northrl.utils import tool_parser as tool_parser_module
from northrl.weaver import WeaverInferenceServiceClient


def _result(arguments="{}", diagnostics=None):
    return {
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "call1", "function": {"name": "Read", "arguments": arguments}}
                    ]
                }
            }
        ],
        "northrl_parse_diagnostics": diagnostics or {},
    }


def test_results_are_deduplicated_across_repeated_and_compacted_history():
    metrics = RolloutMetrics()
    history = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call1", "is_error": True},
                {"type": "tool_result", "tool_use_id": "call2"},
            ],
        }
    ]
    metrics.observe_messages("anthropic", history)
    metrics.observe_messages("anthropic", history)
    metrics.observe_messages("anthropic", [])
    metrics.observe_messages(
        "openai",
        [
            {"role": "tool", "tool_call_id": "call1", "is_error": True},
            {"role": "tool", "tool_call_id": "call3"},
        ],
    )
    values = metrics.snapshot()
    assert values["tool_results_observed"] == 3
    assert values["tool_result_errors"] == 1
    assert values["model_turns"] == 0


def test_snapshot_is_fixed_and_trace_segments_are_not_model_turns():
    metrics = RolloutMetrics()
    initial = metrics.snapshot()
    assert initial and all(
        value == 0.0 for key, value in initial.items() if key != "runtime_tool_observations"
    )
    assert initial["runtime_tool_observations"] == 1
    metrics.record_model_result(_result())
    metrics.record_model_result(_result())
    with metrics.emitted_tool_calls(2):
        pass
    metrics.set_trace_segments(3)
    metrics.set_trace_segments(1)
    current = metrics.snapshot()
    assert current.keys() == initial.keys()
    assert current["model_turns"] == 2
    assert current["generated_tokens"] == 8
    assert current["max_context_tokens"] == 14
    assert current["trace_segments"] == 1
    assert current["tool_calls_emitted"] == 2
    assert all(value == 0 for key, value in initial.items() if key != "runtime_tool_observations")


@pytest.mark.parametrize(
    "arguments,json_errors,type_errors", [("{bad", 1, 0), ("[]", 0, 1), ("null", 0, 1), ({}, 0, 0)]
)
def test_argument_errors_are_classified_without_rejecting_model_result(
    arguments, json_errors, type_errors
):
    metrics = RolloutMetrics()
    metrics.record_model_result(
        _result(
            arguments,
            {
                "tool_calls_detected": 2,
                "tool_calls_invalid": 1,
                "tool_format_error": 1,
            },
        )
    )
    values = metrics.snapshot()
    assert values["model_turns"] == 1
    assert values["generated_tokens"] == 4
    assert values["tool_calls_detected"] == 2
    assert values["tool_calls_invalid"] == 1 + json_errors + type_errors
    assert values["tool_format_error_turns"] == 1
    assert values["tool_arguments_json_errors"] == json_errors
    assert values["tool_arguments_type_errors"] == type_errors
    assert values["tool_calls_emitted"] == 0


@pytest.mark.parametrize("use_tokens", [False, True])
def test_generate_transports_parser_diagnostics_outside_assistant_message(
    monkeypatch: pytest.MonkeyPatch, use_tokens: bool
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)
    config = OmegaConf.create(
        {
            "training": {"weight_sync": {"freeze_inflight": False}},
            "rollout": {
                "parsers": {"type": "builtin", "reasoning": None, "tool": "qwen25"},
                "sampling": {"max_tokens": 16, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
            },
            "models": {"actor": {"name": "test"}},
        }
    )
    raw = {
        "prompt_tokens": [1],
        "response_tokens": [2, 3],
        "response_logprobs": [-0.1, -0.2],
        "finish_reason": "stop",
        "response": '<tool_call>{"name":"Read","arguments":{}}</tool_call>'
        "<tool_call>bad</tool_call>",
    }
    holder = SimpleNamespace(
        sample_from_messages=lambda **kwargs: raw,
        sample_from_token_ids=lambda **kwargs: raw,
    )
    client = WeaverInferenceServiceClient(config, holder)
    if use_tokens:
        result = asyncio.run(client.generate_with_token([1]))
    else:
        result = asyncio.run(client.generate([{"role": "user", "content": "hi"}]))
    assert result["northrl_parse_diagnostics"]["tool_calls_invalid"] == 1
    assert "northrl_parse_diagnostics" not in result["choices"][0]["message"]
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    assert metrics.snapshot()["tool_format_error_turns"] == 1
    assert metrics.snapshot()["generated_tokens"] == 2


@pytest.mark.parametrize("recording_fails", [False, True])
def test_emitted_call_transaction_blocks_concurrent_snapshot_until_commit_or_rollback(
    recording_fails: bool,
) -> None:
    metrics = RolloutMetrics()
    with metrics.emitted_tool_calls(2):
        pass
    reading = Event()
    finished = Event()
    observed: list[float] = []

    def read_snapshot() -> None:
        reading.set()
        observed.append(metrics.snapshot()["tool_calls_emitted"])
        finished.set()

    reader = Thread(target=read_snapshot)
    with (
        pytest.raises(RuntimeError, match="recording failed") if recording_fails else nullcontext()
    ):
        with metrics.emitted_tool_calls(1):
            # A recorder callback can snapshot the pending response on this thread.
            assert metrics.snapshot()["tool_calls_emitted"] == 3
            reader.start()
            assert reading.wait(timeout=1)
            # Heartbeat snapshots must wait until recording has succeeded or failed.
            assert not finished.wait(timeout=0.05)
            if recording_fails:
                raise RuntimeError("recording failed")
    reader.join(timeout=1)
    assert not reader.is_alive()
    assert observed == [2 if recording_fails else 3]


@pytest.mark.parametrize("format_checked", [None, 0, 1])
@pytest.mark.parametrize("has_tool_calls", [False, True])
def test_clean_format_requires_parser_coverage(format_checked, has_tool_calls):
    result = _result()
    if not has_tool_calls:
        result["choices"][0]["message"] = {"content": "plain answer"}
    if format_checked is None:
        result.pop("northrl_parse_diagnostics")
    else:
        result["northrl_parse_diagnostics"] = {"tool_format_checked": format_checked}
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["model_turns"] == 1
    assert values["tool_format_error_turns"] == 0
    assert values["tool_format_unknown_turns"] == int(format_checked != 1)


@pytest.mark.parametrize(
    "arguments,diagnostics",
    [
        ("{}", {"tool_format_error": 1}),
        ("{}", {"tool_calls_invalid": 1, "tool_format_checked": 0}),
        ("{bad", {"tool_format_checked": 0}),
        ("[]", {}),
        ("{bad", {"tool_format_error": 1, "tool_calls_invalid": 1, "tool_format_checked": 1}),
    ],
)
@pytest.mark.parametrize(
    "diagnostics_key", ["northrl_parse_diagnostics", "nexrl_parse_diagnostics"]
)
def test_proven_format_error_is_known_and_counted_once(arguments, diagnostics, diagnostics_key):
    metrics = RolloutMetrics()
    result = _result(arguments, diagnostics)
    result[diagnostics_key] = result.pop("northrl_parse_diagnostics")
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["model_turns"] == 1
    assert values["tool_format_error_turns"] == 1
    assert values["tool_format_unknown_turns"] == 0


@pytest.mark.parametrize("training_key", ["token_data", "northrl_train", "nexrl_train"])
def test_token_metrics_accept_current_and_historical_training_envelopes(training_key):
    result = _result()
    result.pop("usage")
    result[training_key] = {"prompt_tokens": [1, 2], "response_tokens": [3, 4, 5]}
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["generated_tokens"] == 3
    assert values["max_context_tokens"] == 5


def test_northgate_counts_take_priority_without_double_counting_native_errors():
    result = _result(
        "{bad",
        {"tool_calls_detected": 9, "tool_calls_invalid": 8, "tool_format_checked": 0},
    )
    result["tool_call_diagnostics"] = {
        "checked": True,
        "malformed": True,
        "detected": 2,
        "invalid": 1,
        "reasons": ["invalid_arguments_json"],
    }
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["tool_calls_detected"] == 2
    assert values["tool_calls_invalid"] == 1
    assert values["tool_arguments_json_errors"] == 1
    assert values["tool_format_error_turns"] == 1
    assert values["tool_format_unknown_turns"] == 0
    assert values["tool_call_counts_unknown_turns"] == 0


@pytest.mark.parametrize("native_calls", [[], [{"id": "repaired", "function": {"name": "Read"}}]])
def test_northgate_diagnostics_preserve_invalid_calls_after_repair_or_discard(native_calls):
    result = _result()
    result["choices"][0]["message"]["tool_calls"] = native_calls
    result["tool_call_diagnostics"] = {
        "checked": True,
        "malformed": True,
        "detected": 2,
        "invalid": 1,
        "reasons": ["invalid_arguments_json"],
    }
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["tool_calls_detected"] == 2
    assert values["tool_calls_invalid"] == 1
    assert values["tool_format_error_turns"] == 1
    assert values["tool_format_unknown_turns"] == 0
    assert values["tool_arguments_json_errors"] == 0
    assert values["tool_calls_emitted"] == 0


@pytest.mark.parametrize("checked", [False, True])
def test_northgate_checked_controls_clean_format_coverage_even_with_legacy_errors(checked):
    metrics = RolloutMetrics()
    metrics.record_model_result(
        {
            "tool_call_diagnostics": {
                "checked": checked,
                "malformed": False,
                "detected": 0,
                "invalid": 0,
                "reasons": [],
            },
            "northrl_parse_diagnostics": {"tool_format_error": 1, "tool_calls_invalid": 3},
        }
    )
    values = metrics.snapshot()
    assert values["tool_calls_detected"] == values["tool_calls_invalid"] == 0
    assert values["tool_format_error_turns"] == 0
    assert values["tool_format_unknown_turns"] == int(not checked)
    assert values["tool_call_counts_unknown_turns"] == 0


@pytest.mark.parametrize("malformed,arguments", [(True, "{}"), (False, "{bad")])
def test_northgate_unknown_counts_remain_unknown_when_format_error_is_proven(malformed, arguments):
    result = _result(arguments)
    result["tool_call_diagnostics"] = {
        "checked": False,
        "malformed": malformed,
        "detected": None,
        "invalid": None,
        "reasons": [],
    }
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["tool_calls_detected"] == values["tool_calls_invalid"] == 0
    assert values["tool_format_error_turns"] == 1
    assert values["tool_format_unknown_turns"] == 0
    assert values["tool_call_counts_unknown_turns"] == 1
    assert values["tool_arguments_json_errors"] == int(arguments == "{bad")


@pytest.mark.parametrize(
    "result,counts_unknown",
    [
        ({}, 0),
        ({"northrl_parse_diagnostics": {"tool_format_checked": 1}}, 0),
        ({"tool_call_diagnostics": None}, 0),
        ({"tool_call_diagnostics": {}}, 1),
        ({"tool_call_diagnostics": {"detected": 0, "invalid": None}}, 1),
        ({"tool_call_diagnostics": {"detected": None, "invalid": 0}}, 1),
        ({"tool_call_diagnostics": {"detected": 0, "invalid": 0}}, 0),
    ],
)
def test_unknown_count_coverage_requires_explicit_northgate_diagnostics(result, counts_unknown):
    metrics = RolloutMetrics()
    metrics.record_model_result(result)
    values = metrics.snapshot()
    assert values["tool_call_counts_unknown_turns"] == counts_unknown


def test_northgate_reasons_are_fixed_per_turn_counters_in_runtime_metadata():
    expected_reasons = {
        "incomplete_tool_call",
        "invalid_arguments_json",
        "missing_tool_name",
        "invalid_tool_call_structure",
        "arguments_not_object",
        "duplicate_parameter",
        "undeclared_tool",
    }
    assert set(TOOL_FORMAT_REASONS) == expected_reasons
    metrics = RolloutMetrics()
    initial_keys = metrics.snapshot().keys()
    for reasons in (
        list(TOOL_FORMAT_REASONS) * 2 + ["future_reason", {"unexpected": "object"}],
        ["undeclared_tool", "undeclared_tool"],
    ):
        metrics.record_model_result(
            {
                "tool_call_diagnostics": {
                    "checked": True,
                    "malformed": True,
                    "detected": None,
                    "invalid": None,
                    "reasons": reasons,
                }
            }
        )
    values = metrics.snapshot()
    assert values.keys() == initial_keys
    runtime_counts = runtime_metrics_from_snapshot(values)["tool_parse_counts"]
    assert runtime_counts["call_counts_unknown_turn"] == 2
    assert runtime_counts["format_error_turn"] == 2
    assert runtime_counts["format_unknown_turn"] == 0
    assert runtime_counts["calls_detected"] == runtime_counts["calls_invalid"] == 0
    for reason in expected_reasons:
        expected_turns = 2 if reason == "undeclared_tool" else 1
        assert values[f"tool_format_reason_{reason}_turns"] == expected_turns
        assert runtime_counts[f"format_reason_{reason}_turn"] == expected_turns
    assert not any("future_reason" in name for name in (*values, *runtime_counts))


def test_request_latency_histograms_and_records(monkeypatch, caplog):
    from northrl.remote_rollout import metrics as metrics_module

    now = [100.0]
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)
    durations = (0.5, 5.0, 61.0, 1900.0)
    for duration in durations:
        with metrics.track_request("anthropic") as request:
            request.result = _result()
            now[0] += duration
    values = metrics.snapshot()
    assert values["model_requests_started"] == values["model_requests_completed"] == 4
    assert values["model_requests_success"] == 4
    assert values["model_request_seconds_sum"] == sum(durations)
    assert values["model_request_seconds_max"] == 1900
    assert values["model_request_seconds_bucket_le_1"] == 1
    assert values["model_request_seconds_bucket_le_5"] == 2
    assert values["model_request_seconds_bucket_le_60"] == 2
    assert values["model_request_seconds_bucket_le_120"] == 3
    assert values["model_request_seconds_bucket_le_1800"] == 3
    assert values["model_request_seconds_bucket_le_inf"] == 4
    assert values["model_request_inflight"] == 0
    assert values["model_request_oldest_inflight_seconds"] == 0
    assert [record["request_seq"] for record in records] == [1, 2, 3, 4]
    assert [record["duration_seconds"] for record in records] == list(durations)
    assert all(record["prompt_tokens"] == 10 for record in records)
    assert all(record["completion_tokens"] == 4 for record in records)
    assert all("result" not in record and "messages" not in record for record in records)
    assert len([record for record in caplog.records if "Slow model request" in record.message]) == 2


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("failed"), "error"),
        (TimeoutError("late"), "timeout"),
        (asyncio.CancelledError(), "cancelled"),
    ],
)
def test_request_failures_keep_duration_and_propagate(monkeypatch, error, status):
    from northrl.remote_rollout import metrics as metrics_module

    now = [10.0]
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)
    with pytest.raises(type(error)) as caught:
        with metrics.track_request("openai"):
            now[0] = 14.0
            raise error
    assert caught.value is error
    values = metrics.snapshot()
    assert values["model_requests_completed"] == 1
    assert values[f"model_requests_{status}"] == 1
    assert values["model_requests_success"] == 0
    assert values["model_request_seconds_sum"] == 4
    assert values["model_request_inflight"] == 0
    assert records[0]["status"] == status
    assert records[0]["error_type"] == type(error).__name__
    assert records[0]["prompt_tokens"] is None
    assert records[0]["completion_tokens"] is None


def test_concurrent_requests_keep_independent_ages_and_cancelled_duration(monkeypatch):
    from northrl.remote_rollout import metrics as metrics_module

    now = [10.0]
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    metrics = RolloutMetrics()
    records = []
    metrics.set_request_observer(records.append)

    async def scenario():
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def generate(index):
            with metrics.track_request("openai") as request:
                entered[index].set()
                await release[index].wait()
                request.result = _result()

        first = asyncio.create_task(generate(0))
        await entered[0].wait()
        now[0] = 13.0
        second = asyncio.create_task(generate(1))
        await entered[1].wait()
        snapshot = metrics.snapshot()
        assert snapshot["model_request_inflight"] == 2
        assert snapshot["model_request_oldest_inflight_seconds"] == 3
        now[0] = 15.0
        release[1].set()
        await second
        snapshot = metrics.snapshot()
        assert snapshot["model_request_inflight"] == 1
        assert snapshot["model_request_oldest_inflight_seconds"] == 5
        now[0] = 17.0
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(scenario())
    values = metrics.snapshot()
    assert values["model_request_seconds_sum"] == 9
    assert values["model_request_seconds_max"] == 7
    assert values["model_requests_success"] == values["model_requests_cancelled"] == 1
    assert values["model_request_inflight"] == values["model_request_oldest_inflight_seconds"] == 0
    assert [record["request_seq"] for record in records] == [2, 1]


def test_request_observer_failure_is_reported_once_and_preserves_result(caplog):
    metrics = RolloutMetrics()

    def broken_observer(record):
        raise OSError("journal unavailable")

    metrics.set_request_observer(broken_observer)
    with metrics.track_request("anthropic") as request:
        request.result = _result()
    error = RuntimeError("backend failed")
    with pytest.raises(RuntimeError) as caught:
        with metrics.track_request("anthropic"):
            raise error
    assert caught.value is error
    assert metrics.snapshot()["model_requests_completed"] == 2
    assert (
        len(
            [
                record
                for record in caplog.records
                if "Could not record model request" in record.message
            ]
        )
        == 1
    )


def test_request_record_and_slow_warning_identify_rollout_attempt(monkeypatch, caplog):
    from northrl.remote_rollout import metrics as metrics_module

    now = [10.0]
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    metrics = RolloutMetrics(rollout_id="rollout-7", attempt_id="attempt-2")
    records = []
    metrics.set_request_observer(records.append)
    with metrics.track_request("anthropic"):
        now[0] = 75.0
    assert records[0]["rollout_id"] == "rollout-7"
    assert records[0]["attempt_id"] == "attempt-2"
    assert "rollout_id=rollout-7 attempt_id=attempt-2" in caplog.text


@pytest.mark.parametrize("error_name", ["ReadTimeout", "ConnectTimeout", "PoolTimeout"])
def test_transport_timeouts_are_not_generic_errors(error_name):
    import httpx

    error = getattr(httpx, error_name)("transport timeout")
    metrics = RolloutMetrics()
    with pytest.raises(type(error)):
        with metrics.track_request("openai"):
            raise error
    assert metrics.snapshot()["model_requests_timeout"] == 1
    assert metrics.snapshot()["model_requests_error"] == 0


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (504, "timeout", "timeout"),
        (504, "gateway_timeout", "timeout"),
        (500, "unavailable", "error"),
    ],
)
def test_weaver_polling_timeout_uses_error_fields(status, code, expected):
    weaver = pytest.importorskip("weaver")
    error = weaver.WeaverAPIError(status, code, "operation failed", True)
    metrics = RolloutMetrics()
    with pytest.raises(weaver.WeaverAPIError):
        with metrics.track_request("anthropic"):
            raise error
    assert metrics.snapshot()[f"model_requests_{expected}"] == 1
