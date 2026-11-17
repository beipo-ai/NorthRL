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

"""Runtime metadata is execution-scoped and can be aggregated without content."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RemoteRolloutResult, RemoteSegment, RolloutInput
from northrl.remote_rollout.metrics import RolloutMetrics
from northrl.remote_rollout.trajectory_recorder import TrajectoryRecorder
from northrl.rollout_worker.runtime_metrics import ObservedInferenceClient


def _response(tool_id="call-1", *, prompt=None, response=None):
    prompt = [1] if prompt is None else prompt
    response = [2] if response is None else response
    return {
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(response)},
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": tool_id, "function": {"name": "Bash", "arguments": "{}"}}]
                }
            }
        ],
        "northrl_train": {
            "prompt_tokens": prompt,
            "response_tokens": response,
            "response_logprobs": [-0.2] * len(response),
        },
    }


def test_request_statistics_keep_missing_usage_out_of_token_denominator():
    metrics = RolloutMetrics(observe_tools=False)
    with metrics.track_request("generate") as request:
        request.result = _response(prompt=[1, 2], response=[3, 4, 5])
    with pytest.raises(TimeoutError):
        with metrics.track_request("generate"):
            raise TimeoutError("transport timeout")
    runtime = metrics.runtime_snapshot()
    assert runtime["model_requests"] == runtime["model_requests_completed"] == 2
    assert runtime["model_request_timeouts"] == 1
    assert runtime["request_usage_count"] == 1
    assert runtime["request_stats"]["input_tokens"] == {"count": 1, "sum": 2, "min": 2, "max": 2}
    assert runtime["request_stats"]["output_tokens"]["sum"] == 3
    assert runtime["request_stats"]["total_tokens"]["sum"] == 5
    assert runtime["request_stats"]["duration_seconds"]["count"] == 2
    assert "tool_calls" not in runtime


@pytest.mark.parametrize("message_format", ["openai", "anthropic"])
def test_history_timeout_receipt_is_deduplicated_and_keeps_issuing_segment(message_format):
    metrics = RolloutMetrics()
    recorder = TrajectoryRecorder()
    recorder.set_metrics(metrics)
    for result in (_response(), _response("call-2", prompt=[9])):
        with metrics.track_request("generate") as request:
            request.result = result
        with metrics.emitted_tool_calls(1):
            recorder.record_inference_result(result)
    # Adapters observe receipts in a later model request's history. Repeated
    # history does not double-count or move a result to the latest segment.
    if message_format == "openai":
        history = [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "is_error": True,
                "content": "Command timed out after 120 seconds",
            }
        ]
    else:
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "is_error": True,
                        "content": "Command timed out after 120 seconds",
                    }
                ],
            }
        ]
    metrics.observe_messages(message_format, history)
    metrics.observe_messages(message_format, history)
    runtime = metrics.runtime_snapshot()
    assert runtime["model_requests"] == runtime["model_requests_completed"] == 2
    assert runtime["tool_calls"] == 2
    assert runtime["tool_results"] == runtime["tool_timeouts"] == 1
    assert runtime["tool_unobserved"] == 1
    assert runtime["tool_unassigned"] == 0
    segments = recorder.to_remote_segments()
    assert segments[0].segment_metrics == {"tool_calls": 1, "tool_timeouts": 1, "tool_results": 1}
    assert segments[1].segment_metrics == {"tool_calls": 1, "tool_timeouts": 0, "tool_results": 0}


def test_final_call_without_a_later_history_stays_unobserved_not_timed_out():
    metrics = RolloutMetrics()
    recorder = TrajectoryRecorder()
    recorder.set_metrics(metrics)
    result = _response()
    with metrics.track_request("generate") as request:
        request.result = result
    with metrics.emitted_tool_calls(1):
        recorder.record_inference_result(result)
    # The runtime sees the emitted call but no result receipt is submitted in
    # another model request. It cannot infer whether execution timed out.
    runtime = metrics.runtime_snapshot()
    assert runtime["model_requests"] == runtime["model_requests_completed"] == 1
    assert runtime["request_usage_count"] == 1
    assert runtime["request_stats"]["total_tokens"]["sum"] == 2
    assert runtime["tool_calls"] == runtime["tool_unobserved"] == 1
    assert runtime["tool_results"] == runtime["tool_timeouts"] == 0
    assert runtime["model_request_timeouts"] == 0


def test_successful_text_mentioning_timeout_is_not_a_timeout():
    metrics = RolloutMetrics()
    metrics.observe_messages(
        "openai",
        [
            {
                "role": "tool",
                "tool_call_id": "read",
                "content": "TimeoutError: example in a source file",
            }
        ],
    )
    assert metrics.runtime_snapshot()["tool_timeouts"] == 0
    assert metrics.runtime_snapshot()["tool_unassigned"] == 1


def test_tool_ids_are_isolated_between_execution_collectors():
    first, second = RolloutMetrics(attempt_id="one"), RolloutMetrics(attempt_id="two")
    receipt = [{"role": "tool", "tool_call_id": "same", "timed_out": True}]
    first.observe_messages("openai", receipt)
    second.observe_messages("openai", receipt)
    assert (
        first.runtime_snapshot()["tool_timeouts"] == second.runtime_snapshot()["tool_timeouts"] == 1
    )


def test_sandbox_retries_do_not_mix_request_or_tool_observations(monkeypatch):
    final_metrics = RolloutMetrics()
    with final_metrics.track_request("generate") as request:
        request.result = _response()
    with final_metrics.emitted_tool_calls(1):
        final_metrics.record_emitted_ids(_response())
    results = [
        RemoteRolloutResult(
            rollout_id="parent",
            status="env_error",
            reward=0.0,
            metrics={"tool_calls_emitted": 90.0, "model_requests_started": 40.0},
        ),
        RemoteRolloutResult(
            rollout_id="parent",
            reward=1.0,
            segments=[
                RemoteSegment(
                    tokens=[1, 2],
                    loss_mask=[0, 1],
                    log_probs=[0.0, -0.2],
                    old_log_probs=[0.0, -0.2],
                )
            ],
            metrics=final_metrics.snapshot(),
        ),
    ]

    async def attempt(*args, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(e2b_executor, "_run_rollout_in_e2b_once", attempt)
    result = asyncio.run(
        e2b_executor.run_rollout_in_e2b(
            RolloutInput(rollout_id="parent", task={}),
            OmegaConf.create({"rollout": {"execution": {"env_error_retries": 1}}}),
            gateway={
                "base_url": "https://northgate.example",
                "model": "weaver-sampler",
                "sampling_session_id": "sampler",
                "weight_version": "default",
            },
            template="template",
            timeout_seconds=10,
            prepared_artifacts=SimpleNamespace(),
        )
    )
    assert result.metrics["sandbox_retries"] == 1
    assert result.metrics["tool_calls_emitted"] == result.metrics["model_requests_started"] == 1
    assert result.metrics["rollout_finished_at"] >= result.metrics["rollout_started_at"]
    assert result.metrics["rollout_duration_seconds"] >= 0


def test_normal_worker_client_records_legacy_completion_token_lengths():
    async def completion(prompt):
        return {"prompt_tokens": [1, 2], "response_tokens": [3], "response_logprobs": [-0.1]}

    metrics = RolloutMetrics(observe_tools=False)
    client = ObservedInferenceClient(SimpleNamespace(completion=completion), metrics)
    asyncio.run(client.completion("prompt"))
    runtime = metrics.runtime_snapshot()
    assert runtime["model_requests"] == 1
    assert runtime["request_stats"]["input_tokens"]["sum"] == 2
    assert runtime["request_stats"]["output_tokens"]["sum"] == 1
    assert "tool_calls" not in runtime


def test_reused_tool_id_does_not_invent_segment_ownership():
    metrics = RolloutMetrics()
    recorder = TrajectoryRecorder()
    recorder.set_metrics(metrics)
    for result in (_response(), _response(prompt=[9])):
        with metrics.emitted_tool_calls(1):
            recorder.record_inference_result(result)
    metrics.observe_messages(
        "openai", [{"role": "tool", "tool_call_id": "call-1", "timed_out": True}]
    )
    assert all(segment.segment_metrics == {} for segment in recorder.to_remote_segments())
    assert "tool_unobserved" not in metrics.runtime_snapshot()
    assert metrics.runtime_snapshot()["tool_unassigned"] == 1


def test_json_timeout_receipt_requires_execution_envelope():
    from northrl.remote_rollout.tool_errors import is_tool_timeout

    assert is_tool_timeout({"is_error": True, "content": '{"timed_out": true}'})
    assert is_tool_timeout({"content": '{"status": "timeout", "exit_code": 124}'})
    assert not is_tool_timeout({"content": '{"timed_out": true}'})
