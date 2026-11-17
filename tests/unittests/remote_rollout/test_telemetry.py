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

"""Heartbeat coverage while generation is idle, evaluating, failing, or cancelled."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from northrl.remote_rollout import telemetry as telemetry_module
from northrl.remote_rollout.telemetry import RolloutTelemetry


@pytest.fixture
def fast_heartbeat(monkeypatch):
    monkeypatch.setattr(telemetry_module, "LIVE_METRICS_INTERVAL_SECONDS", 0.01)


def test_telemetry_is_enabled_by_default_with_thirty_second_interval(tmp_path):
    async def scenario():
        async with RolloutTelemetry("run-1", tmp_path / "result.json") as telemetry:
            assert telemetry._interval == 30.0
            snapshot = json.loads((tmp_path / "metrics.json").read_text())
            assert snapshot["heartbeat"] == 1
            assert snapshot["phase"] == "runtime"
        assert json.loads((tmp_path / "metrics.json").read_text())["phase"] == "finished"

    asyncio.run(scenario())


async def _wait_for_heartbeat(path: Path, after: int):
    async with asyncio.timeout(2):
        while True:
            value = json.loads(path.read_text()) if path.exists() else None
            if value is not None and value["heartbeat"] > after:
                return value
            await asyncio.sleep(0.002)


def test_heartbeat_runs_without_model_output_and_during_evaluation(tmp_path, fast_heartbeat):
    async def scenario():
        metrics_path = tmp_path / "metrics.json"
        async with RolloutTelemetry("run-1", tmp_path / "result.json") as telemetry:
            first = json.loads(metrics_path.read_text())
            next_value = await _wait_for_heartbeat(metrics_path, first["heartbeat"])
            assert next_value["metrics"]["model_turns"] == 0.0
            telemetry.metrics.record_model_result(
                {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
            )
            telemetry.metrics.set_trace_segments(1)
            telemetry.set_phase("evaluator")
            evaluation = json.loads(metrics_path.read_text())
            assert evaluation["phase"] == "evaluator"
            assert evaluation["metrics"]["generated_tokens"] == 3.0
            assert evaluation["metrics"]["trace_segments"] == 1.0
            assert (await _wait_for_heartbeat(metrics_path, evaluation["heartbeat"]))[
                "phase"
            ] == "evaluator"
        final = json.loads(metrics_path.read_text())
        assert final["phase"] == "finished"
        assert final["metrics"]["model_turns"] == 1.0
        assert metrics_path.stat().st_mode & 0o777 == 0o600

    asyncio.run(scenario())


def test_cancellation_preserves_last_metrics_and_stops_heartbeat(tmp_path, fast_heartbeat):
    async def scenario():
        path = tmp_path / "metrics.json"

        async def rollout():
            async with RolloutTelemetry("run-1", tmp_path / "result.json") as telemetry:
                with telemetry.metrics.emitted_tool_calls(2):
                    pass
                await asyncio.Event().wait()

        task = asyncio.create_task(rollout())
        await _wait_for_heartbeat(path, 0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        final = json.loads(path.read_text())
        assert final["phase"] == "failed"
        assert final["metrics"]["tool_calls_emitted"] == 2.0
        await asyncio.sleep(0.03)
        assert json.loads(path.read_text())["heartbeat"] == final["heartbeat"]

    asyncio.run(scenario())


def test_unwritable_telemetry_does_not_break_rollout(tmp_path, fast_heartbeat):
    async def scenario():
        blocked = tmp_path / "file-not-directory"
        blocked.write_text("occupied")
        async with RolloutTelemetry("run-2", blocked / "result.json"):
            await asyncio.sleep(0.02)
        assert blocked.read_text() == "occupied"

    asyncio.run(scenario())


def test_heartbeat_continues_when_runtime_event_loop_is_blocked(tmp_path, fast_heartbeat):
    async def scenario():
        path = tmp_path / "metrics.json"
        async with RolloutTelemetry("run-1", tmp_path / "result.json"):
            before = json.loads(path.read_text())["heartbeat"]
            # Models load tokenizers synchronously before serving requests.
            time.sleep(0.1)
            assert json.loads(path.read_text())["heartbeat"] > before

    asyncio.run(scenario())


def test_agent_and_evaluation_processes_preserve_heartbeat_and_model_metrics(tmp_path):
    async def scenario():
        output = tmp_path / "result.json"
        path = tmp_path / "metrics.json"
        async with RolloutTelemetry(
            "run-1", output, attempt_id="attempt-1", completion_phase="agent_completed"
        ) as agent:
            agent.metrics.record_model_result(
                {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
            )
            metrics = agent.metrics.snapshot()
        before = json.loads(path.read_text())
        assert before["phase"] == "agent_completed"
        async with RolloutTelemetry(
            "run-1",
            output,
            attempt_id="attempt-1",
            phase="evaluator",
            initial_metrics=metrics,
            resume=True,
        ) as evaluation:
            after = json.loads(path.read_text())
            assert after["heartbeat"] > before["heartbeat"]
            assert after["phase"] == "evaluator"
            assert after["metrics"]["generated_tokens"] == 3.0
            evaluation.update_metrics({"task_score": 0.5})
        final = json.loads(path.read_text())
        assert final["phase"] == "finished"
        assert final["metrics"]["model_turns"] == 1.0
        assert final["metrics"]["task_score"] == 0.5

    asyncio.run(scenario())


@pytest.mark.parametrize("identity", [("another-run", "attempt-1"), ("run-1", "attempt-2")])
def test_telemetry_does_not_resume_another_attempt(tmp_path, identity):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"rollout_id": "run-1", "attempt_id": "attempt-1", "heartbeat": 900})
    )

    async def scenario():
        async with RolloutTelemetry(
            identity[0], tmp_path / "result.json", attempt_id=identity[1], resume=True
        ):
            assert json.loads(path.read_text())["heartbeat"] == 1

    asyncio.run(scenario())


def test_request_gauges_decrease_and_are_cleared_when_evaluation_resumes(tmp_path):
    async def scenario():
        output = tmp_path / "result.json"
        path = tmp_path / "metrics.json"
        async with RolloutTelemetry("run-1", output, attempt_id="attempt-1") as agent:
            with agent.metrics.track_request("anthropic"):
                agent.update_metrics(agent.metrics.snapshot())
                assert json.loads(path.read_text())["metrics"]["model_request_inflight"] == 1
            agent.set_phase("agent_completed")
            finished = json.loads(path.read_text())["metrics"]
            assert finished["model_request_inflight"] == 0
            assert finished["model_request_oldest_inflight_seconds"] == 0
        # Even a checkpoint captured before the request finished cannot retain a
        # live model request in the separate evaluator process.
        previous = {
            **finished,
            "model_request_inflight": 2.0,
            "model_request_oldest_inflight_seconds": 70.0,
        }
        async with RolloutTelemetry(
            "run-1",
            output,
            attempt_id="attempt-1",
            phase="evaluator",
            initial_metrics=previous,
            resume=True,
        ) as evaluation:
            resumed = evaluation.metrics_snapshot()
            assert resumed["model_requests_completed"] == 1
            assert resumed["model_request_seconds_sum"] == finished["model_request_seconds_sum"]
            assert resumed["model_request_inflight"] == 0
            assert resumed["model_request_oldest_inflight_seconds"] == 0

    asyncio.run(scenario())
