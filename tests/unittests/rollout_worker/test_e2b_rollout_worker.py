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

"""Tests for the E2B rollout worker."""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from pydantic_core import PydanticSerializationError

from northrl.remote_rollout import e2b_executor
from northrl.remote_rollout.contracts import RemoteRolloutResult, RemoteSegment
from northrl.rollout_worker import E2BRolloutWorker
from northrl.rollout_worker import e2b_rollout_worker as worker_module
from northrl.utils.async_utils import run_coro


def _gateway(model="weaver-default", version="default"):
    return {
        "base_url": "https://northgate.example",
        "model": model,
        "sampling_session_id": model.removeprefix("weaver-"),
        "weight_version": version,
    }


def _remote_segment() -> RemoteSegment:
    return RemoteSegment(
        tokens=[10, 11, 12],
        loss_mask=[0, 1, 1],
        log_probs=[0.0, -0.1, -0.2],
        old_log_probs=[0.0, -0.3, -0.4],
        sampling_mask=[[11], [12]],
    )


def test_result_profiles_separate_postprocessing_from_actual_pool_publication(
    e2b_worker, monkeypatch
):
    from northrl.utils import profiling

    monkeypatch.setenv("NORTHRL_PROFILE_RESULTS", "1")
    events = Mock()
    monkeypatch.setattr(profiling, "profile_event", events)

    async def execute(request, *args, **kwargs):
        return RemoteRolloutResult(
            rollout_id=request.rollout_id, segments=[_remote_segment()], reward=1.0
        )

    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", execute)
    pool = Mock()

    def publish(trajectory):
        # The postprocess record is already complete. Staging must not have
        # emitted a misleading pool-insertion record before the real call.
        assert [call.args[0] for call in events.call_args_list] == ["driver.result-postprocess"]
        assert trajectory["rollout_id"] == "profiled-rollout"
        return "success"

    pool.put_trajectory.side_effect = publish
    e2b_worker._trajectory_pool = pool
    assert (
        run_coro(
            e2b_worker.execute_rollout(
                {"rollout_id": "profiled-rollout", "template": "test", "instruction": "fix"}
            )
        )
        == "success"
    )
    pool.put_trajectory.assert_called_once()
    profiles = events.call_args_list
    assert [call.args[0] for call in profiles] == [
        "driver.result-postprocess",
        "driver.result-pool-insert",
    ]
    assert all(
        "parent_wall_seconds" in call.kwargs and "parent_cpu_seconds" in call.kwargs
        for call in profiles
    )


@pytest.fixture(params=["sync", "fully-async"])
def e2b_worker(rollout_worker_config, request):
    rollout_worker_config.backend.type = "weaver"
    rollout_worker_config.rollout.update(
        {
            "type": "e2b",
            "execution": {
                "timeout_seconds": 120,
                "input_fields": ["instruction", "interface"],
            },
        }
    )
    rollout_worker_config.training.weight_sync.mode = request.param

    worker = E2BRolloutWorker(rollout_worker_config)
    holder = Mock()
    holder.get_current_gateway_snapshot = AsyncMock(return_value=_gateway())
    worker.init_inference_service_client(holder)

    return worker


@pytest.mark.parametrize("validation", [False, True])
def test_rollout_converts_remote_result(e2b_worker, monkeypatch, validation):
    if validation:
        e2b_worker.begin_validate()
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=1.0,
        metrics={"swebench_resolved": 1.0},
    )
    executor = AsyncMock(return_value=result)
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    put_trajectory = Mock(return_value="success")
    e2b_worker._put_trajectory = put_trajectory

    task = {
        "instruction": "Fix the bug",
        "template": "task-specific-template",
        "interface": "Expose Rule.set_split_amount(amount).",
        "patch": "reference solution must stay on the driver",
        "test_patch": "hidden tests must stay on the driver",
        "FAIL_TO_PASS": ["hidden_test"],
        "PASS_TO_PASS": ["regression_test"],
        "pr_description": "description of the reference solution",
        "install_config": {"test_cmd": "pytest hidden_test"},
        "meta": {"reasoning": "analysis of the reference solution"},
        "n2_pass_count": 4,
        "ds_pass_count": 3,
        "custom": {"checks": ["public", {"enabled": True}], "optional": None},
        "group_id": "group-1",
        "run_id": 2,
        "task_id": "instance-1",
        "data_source": "swerebench",
        "is_val": True,
    }
    assert run_coro(e2b_worker.rollout(task)) == "success"

    request, config = executor.await_args.args
    assert request.task == {
        "instruction": "Fix the bug",
        "interface": "Expose Rule.set_split_amount(amount).",
    }
    assert request.metadata == {
        "task_id": "instance-1",
        "group_id": "group-1",
        "run_id": 2,
        "weight_version": "default",
        **({"is_validation": True} if validation else {}),
    }
    assert config is e2b_worker._config
    assert executor.await_args.kwargs == {
        "gateway": _gateway(),
        "template": "task-specific-template",
        "timeout_seconds": 120,
        "trace_output_path": None,
    }

    trajectory = put_trajectory.call_args.args[0]
    assert trajectory.segments[0]["tokens"] == [10, 11, 12]
    assert trajectory.segments[0]["loss_mask"] == [0, 1, 1]
    assert trajectory.reward == 1.0
    assert trajectory.is_val is True
    assert trajectory["group_id"] == "group-1"
    assert trajectory["run_id"] == 2
    assert trajectory["task_id"] == "instance-1"
    assert trajectory["model_tag"] == "actor"
    assert trajectory.segments[0]["old_log_probs"] == [0.0, -0.3, -0.4]
    assert trajectory.segments[0]["sampling_mask"] == [[11], [12]]
    assert trajectory["score"] == {
        "swebench_resolved": 1.0,
        "e2b_timed_out": 0.0,
        "e2b_task_failed": 0.0,
        "e2b_env_error": 0.0,
        "reward": 1.0,
    }
    assert trajectory.segments[0]["is_truncated"] is False
    assert trajectory["e2b_rollout_id"] == "remote-id"
    assert trajectory["e2b_timed_out"] is False
    assert trajectory["weight_version"] == "default"


@pytest.mark.parametrize("validation", [False, True])
def test_rollout_forwards_shared_prepared_artifacts(e2b_worker, monkeypatch, tmp_path, validation):
    if validation:
        e2b_worker.begin_validate()
    artifacts = e2b_executor.PreparedE2BArtifacts(
        entrypoint=e2b_executor._EntrypointSpec(script_path=tmp_path / "entrypoint.py"),
        source_archive=b"shared source",
        evaluation_archives={None: b"shared evaluation"},
    )
    e2b_worker.set_prepared_artifacts(artifacts)
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock(return_value="success")

    for _ in range(2):
        assert (
            run_coro(
                e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
            )
            == "success"
        )

    assert executor.await_count == 2
    for call in executor.await_args_list:
        assert call.kwargs["prepared_artifacts"] is artifacts
        assert call.args[0].task == {"instruction": "Fix the bug"}


def test_rollout_passes_runtime_debug_dump_path(e2b_worker, monkeypatch, tmp_path):
    trace_dir = tmp_path / "rollout_traces"
    trace_dir.mkdir()
    e2b_worker._rollout_trace_dir = str(trace_dir)
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=1.0,
    )
    executor = AsyncMock(return_value=result)
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock(return_value="success")

    task = {"instruction": "Fix the bug", "template": "swebench-template"}
    assert run_coro(e2b_worker.rollout(task)) == "success"

    request = executor.await_args.args[0]
    assert executor.await_args.kwargs["trace_output_path"] == (
        trace_dir / f"e2b_rollout_{request.rollout_id}.json.gz"
    )


def test_rollout_emits_all_segments_with_shared_reward(e2b_worker, monkeypatch):
    second_remote = RemoteSegment(
        tokens=[20, 21],
        loss_mask=[0, 1],
        log_probs=[0.0, -0.5],
        old_log_probs=[0.0, -0.6],
        sampling_mask=[[21]],
    )
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment(), second_remote],
        reward=0.75,
        metrics={"swebench_resolved": 1.0},
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    put_trajectory = Mock(side_effect=["success", "success"])
    e2b_worker._put_trajectory = put_trajectory

    task = {
        "instruction": "Fix the bug",
        "template": "swebench-template",
        "group_id": "group-1",
        "run_id": 2,
    }
    assert run_coro(e2b_worker.rollout(task)) == "success"

    put_trajectory.assert_called_once()
    trajectory = put_trajectory.call_args.args[0]
    assert [segment["tokens"] for segment in trajectory.segments] == [[10, 11, 12], [20, 21]]
    assert trajectory.reward == 0.75
    assert trajectory["group_id"] == "group-1"
    assert trajectory["run_id"] == 2
    assert trajectory["score"] == {
        "swebench_resolved": 1.0,
        "e2b_timed_out": 0.0,
        "e2b_task_failed": 0.0,
        "e2b_env_error": 0.0,
        "reward": 0.75,
    }


def test_rollout_marks_timed_out_partial_trajectory(e2b_worker, monkeypatch):
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=0.0,
        metrics={"swebench_resolved": 0.0, "e2b_timed_out": 1.0},
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    put_trajectory = Mock(return_value="success")
    e2b_worker._put_trajectory = put_trajectory

    assert (
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )
        == "success"
    )

    trajectory = put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.0
    assert trajectory.segments[0]["finish_reason"] == "timeout"
    assert trajectory.segments[0]["is_truncated"] is False
    assert trajectory["e2b_rollout_id"] == "remote-id"
    assert trajectory["e2b_timed_out"] is True
    assert trajectory["score"] == {
        "swebench_resolved": 0.0,
        "e2b_timed_out": 1.0,
        "e2b_task_failed": 0.0,
        "e2b_env_error": 0.0,
        "reward": 0.0,
    }


def test_rollout_truncates_oversized_trajectory_and_resets_shared_reward(e2b_worker, monkeypatch):
    e2b_worker._max_sequence_length = 6
    oversized_remote = RemoteSegment(
        tokens=[20, 21, 22, 23, 24, 25, 26],
        loss_mask=[0, 1, 1, 0, 0, 1, 1],
        log_probs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.5, -0.6],
        old_log_probs=[0.0, -0.1, -0.2, 0.0, 0.0, -0.5, -0.6],
        sampling_mask=[[21], [22], [25], [26]],
    )
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment(), oversized_remote],
        reward=1.0,
        metrics={"swebench_resolved": 1.0},
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    e2b_worker._put_trajectory = Mock(side_effect=["success", "success"])

    assert (
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )
        == "success"
    )

    e2b_worker._put_trajectory.assert_called_once()
    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    normal, truncated = trajectory.segments

    assert normal["tokens"] == [10, 11, 12]
    assert normal["loss_mask"] == [0, 1, 1]
    assert normal["is_truncated"] is False
    assert normal["finish_reason"] == "stop"

    assert truncated["tokens"] == [20, 21, 22, 23, 24, 25]
    assert truncated["loss_mask"] == [0, 1, 1, 0, 0, 1]
    assert truncated["old_log_probs"] == [0.0, -0.1, -0.2, 0.0, 0.0, -0.5]
    assert truncated["sampling_mask"] == [[21], [22], [25]]
    assert truncated["is_truncated"] is True
    assert truncated["finish_reason"] == "length"

    assert trajectory.reward == 0.0
    assert trajectory["score"]["reward"] == 0.0
    assert trajectory["score"]["swebench_resolved"] == 0.0
    assert trajectory["e2b_rollout_id"] == "remote-id"
    assert trajectory["e2b_timed_out"] is False


@pytest.mark.parametrize("limit", [5, 6, 9])
def test_turn_annotations_clip_only_affected_responses(e2b_worker, limit):
    from northrl.remote_rollout.trajectory_recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder()
    for prompt, response in (
        ([10], [11, 12]),
        ([10, 11, 12, 13], [14, 15]),
        ([10, 11, 12, 13, 14, 15, 16], [17, 18]),
    ):
        recorder.record_model_turn(
            prompt_tokens=prompt,
            response_tokens=response,
            response_log_probs=[-0.1, -0.2],
            response_old_log_probs=[-0.1, -0.2],
            diagnostics={"malformed": True, "reasons": ["invalid_arguments_json"]},
            finish_reason="tool_calls",
            tools_declared=True,
        )
    result = RemoteRolloutResult(
        rollout_id="annotated", reward=1.0, segments=recorder.to_remote_segments()
    )
    e2b_worker._max_sequence_length = limit
    trajectory = e2b_worker._trajectory_from_result(result, {}, "v1")
    [segment] = trajectory.segments
    annotations = segment["turn_annotations"]
    assert annotations[0]["response_span"] == [1, 3]
    assert annotations[0]["truncated"] is False
    assert annotations[1]["response_span"] == [4, min(6, limit)]
    assert annotations[1]["truncated"] is (limit == 5)
    assert len(annotations) == (3 if limit == 9 else 2)
    assert all(annotation["finish_reason"] == "tool_calls" for annotation in annotations)
    assert result.segments[0].turn_annotations[1]["response_span"] == [4, 6]
    assert result.segments[0].turn_annotations[1]["truncated"] is False
    annotations[0]["diagnostics"]["reasons"].clear()
    assert result.segments[0].turn_annotations[0]["diagnostics"]["reasons"]


def test_rollout_marks_oversized_timeout_trace_as_length(e2b_worker, monkeypatch):
    e2b_worker._max_sequence_length = 3
    oversized_remote = RemoteSegment(
        tokens=[20, 21, 22, 23],
        loss_mask=[0, 1, 1, 1],
        log_probs=[0.0, -0.1, -0.2, -0.3],
        old_log_probs=[0.0, -0.1, -0.2, -0.3],
        sampling_mask=[[21], [22], [23]],
    )
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[oversized_remote],
        reward=0.0,
        metrics={"swebench_resolved": 0.0, "e2b_timed_out": 1.0},
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    e2b_worker._put_trajectory = Mock(return_value="success")

    assert (
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )
        == "success"
    )

    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert trajectory.segments[0]["finish_reason"] == "length"
    assert trajectory.segments[0]["is_truncated"] is True
    assert trajectory["e2b_timed_out"] is True
    assert trajectory.reward == 0.0


def test_rollout_does_not_truncate_at_exact_max_sequence_length(e2b_worker, monkeypatch):
    e2b_worker._max_sequence_length = 3
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=1.0,
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    e2b_worker._put_trajectory = Mock(return_value="success")

    assert (
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )
        == "success"
    )

    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert trajectory.segments[0]["tokens"] == [10, 11, 12]
    assert trajectory.reward == 1.0
    assert trajectory.segments[0]["is_truncated"] is False


@pytest.mark.parametrize(
    "changed_snapshot",
    [
        _gateway("weaver-next", "default"),
        _gateway("weaver-default", "actor-step-1"),
        _gateway("weaver-next", "actor-step-1"),
    ],
)
def test_rollout_keeps_requested_model_when_holder_changes(
    e2b_worker, monkeypatch, changed_snapshot
):
    initial_snapshot = _gateway()
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=1.0,
    )

    async def run_rollout(*args, **kwargs):
        e2b_worker._service_holder.get_current_gateway_snapshot.return_value = changed_snapshot
        return result

    executor = AsyncMock(side_effect=run_rollout)
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock(return_value="success")

    assert (
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )
        == "success"
    )

    executor.assert_awaited_once()
    assert executor.await_args.kwargs["gateway"] == initial_snapshot
    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert trajectory["weight_version"] == initial_snapshot["weight_version"]
    assert trajectory.segments[0]["tokens"] == [10, 11, 12]
    assert trajectory.reward == 1.0


def test_rollout_propagates_executor_failure(e2b_worker, monkeypatch):
    error = RuntimeError("remote rollout failed")
    executor = AsyncMock(side_effect=error)
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock()

    with pytest.raises(RuntimeError, match="remote rollout failed") as raised:
        run_coro(
            e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
        )

    assert raised.value is error
    executor.assert_awaited_once()
    e2b_worker._put_trajectory.assert_not_called()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda config: config.algorithm.update({"loss": {"mode": "score_centering"}}),
            "score_centering",
        ),
        (
            lambda config: config.algorithm.update(
                {"router_replay": {"enabled": True, "mode": "R3"}}
            ),
            "R3 router replay",
        ),
        (lambda config: setattr(config.backend, "type", "mock"), "backend.type"),
        (
            lambda config: setattr(config.training.weight_sync, "mode", "batch-async"),
            "weight_sync.mode",
        ),
        (
            lambda config: setattr(config.training.weight_sync, "mode", "no-sync"),
            "weight_sync.mode",
        ),
        (
            lambda config: setattr(
                config.rollout,
                "execution",
                {"timeout_seconds": 0},
            ),
            "timeout_seconds",
        ),
    ],
)
def test_worker_validates_minimal_config(rollout_worker_config, change, message):
    rollout_worker_config.backend.type = "weaver"
    rollout_worker_config.rollout.update(
        {
            "type": "e2b",
            "execution": {},
        }
    )
    rollout_worker_config.training.weight_sync.mode = "sync"
    change(rollout_worker_config)

    with pytest.raises(ValueError, match=message):
        E2BRolloutWorker(rollout_worker_config)


@pytest.mark.parametrize(
    "input_fields",
    [None, "instruction", 123, {}, [""], ["   "], [1], [None], [["nested"]], ["x", "x"]],
)
def test_worker_rejects_invalid_input_fields(e2b_worker, input_fields):
    config = e2b_worker._config
    config.rollout.execution.input_fields = input_fields

    with pytest.raises(ValueError, match="rollout.execution.input_fields"):
        E2BRolloutWorker(config)


def test_worker_requires_explicit_input_fields(e2b_worker):
    config = e2b_worker._config
    del config.rollout.execution.input_fields

    with pytest.raises(ValueError, match="rollout.execution.input_fields"):
        E2BRolloutWorker(config)


@pytest.mark.parametrize(
    ("input_fields", "expected"),
    [
        (
            ["question", "context"],
            {"question": "What is 6 * 7?", "context": {"choices": [42, 54]}},
        ),
        (["question", "optional"], {"question": "What is 6 * 7?"}),
        ([], {}),
    ],
)
def test_rollout_selects_configured_fields_for_other_tasks(
    e2b_worker, monkeypatch, input_fields, expected
):
    config = e2b_worker._config
    config.rollout.execution.input_fields = input_fields
    worker = E2BRolloutWorker(config)
    worker.init_inference_service_client(e2b_worker._service_holder)
    worker._put_trajectory = Mock(return_value="success")
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    task = {
        "template": "question-template",
        "task_id": "question-1",
        "question": "What is 6 * 7?",
        "context": {"choices": [42, 54]},
        "answer": 42,
        "interface": "Unselected fields must not be forwarded automatically.",
    }

    assert run_coro(worker.rollout(task)) == "success"

    request = executor.await_args.args[0]
    assert request.task == expected
    assert request.metadata["task_id"] == "question-1"
    assert task["answer"] == 42


@pytest.mark.parametrize("template", [None, "", "   ", 123])
def test_rollout_requires_task_template(e2b_worker, monkeypatch, template):
    executor = AsyncMock()
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    with pytest.raises(ValueError, match="task requires a non-empty template"):
        run_coro(e2b_worker.rollout({"instruction": "Fix the bug", "template": template}))

    executor.assert_not_awaited()


@pytest.mark.parametrize("task_id", [123, "  instance-1  "])
def test_rollout_keeps_task_id_in_metadata(e2b_worker, monkeypatch, task_id):
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment()],
        reward=1.0,
    )
    executor = AsyncMock(return_value=result)
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock(return_value="success")

    task = {
        "template": "generic-task-template",
        "task_id": task_id,
        "question": {"text": "What is 6 * 7?", "choices": [42, 54]},
        "answer": 42,
    }
    assert run_coro(e2b_worker.rollout(task)) == "success"

    input_data = executor.await_args.args[0]
    assert input_data.task == {}
    assert input_data.metadata["task_id"] == task_id


@pytest.mark.parametrize("interface", [None, float("nan")])
def test_rollout_ignores_other_columns_and_normalizes_optional_interface(
    e2b_worker, monkeypatch, interface
):
    task = {
        "instruction": "Fix the bug",
        "interface": interface,
        "template": "task-template",
        "created_at": pd.Timestamp("2025-01-02T03:04:05Z"),
        "license": float("nan"),
        "custom": object(),
    }
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    e2b_worker._put_trajectory = Mock(return_value="success")

    assert run_coro(e2b_worker.rollout(task)) == "success"

    request = executor.await_args.args[0]
    assert request.task == {"instruction": "Fix the bug", "interface": None}
    assert task["interface"] is interface
    assert isinstance(task["created_at"], pd.Timestamp)


def test_rollout_rejects_unserializable_prompt_fields(e2b_worker, monkeypatch):
    executor = AsyncMock()
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    with pytest.raises(PydanticSerializationError, match="Unable to serialize unknown type"):
        run_coro(e2b_worker.rollout({"template": "task-template", "instruction": object()}))

    executor.assert_not_awaited()


@pytest.mark.parametrize(
    ("max_length", "timed_out", "finish_reason"),
    [(3, False, "error"), (3, True, "timeout"), (2, False, "length"), (2, True, "length")],
)
def test_rollout_trains_task_failure_with_zero_reward_and_error_details(
    e2b_worker, monkeypatch, max_length, timed_out, finish_reason
):
    e2b_worker._max_sequence_length = max_length
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        status="task_failed",
        segments=[_remote_segment()],
        reward=0.0,
        metrics={"e2b_timed_out": float(timed_out)},
        error_type="RuntimeError",
        error_message="agent command failed",
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    e2b_worker._put_trajectory = Mock(return_value="success")

    outcome = run_coro(
        e2b_worker.rollout({"instruction": "Fix the bug", "template": "swebench-template"})
    )

    assert outcome == "success"
    e2b_worker._put_trajectory.assert_called_once()
    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.0
    assert trajectory.segments[0]["finish_reason"] == finish_reason
    assert trajectory["e2b_error_type"] == "RuntimeError"
    assert trajectory["e2b_error_message"] == "agent command failed"
    assert trajectory["score"]["e2b_task_failed"] == 1.0
    assert trajectory["score"]["e2b_env_error"] == 0.0
    assert trajectory["score"]["reward"] == 0.0
    assert trajectory.status == "success"
    assert trajectory["e2b_status"] == "task_failed"


@pytest.mark.parametrize("status", ["success", "task_failed", "env_error"])
@pytest.mark.parametrize("segment_count", [0, 2])
def test_e2b_terminal_outcome_completes_one_grpo_slot(
    e2b_worker, monkeypatch, status, segment_count
):
    from omegaconf import OmegaConf

    from northrl.algorithm.advantage.grpo import compute_grpo_advantage_for_trajectories
    from northrl.trajectory_pool import GroupedTrajectoryStore
    from northrl.utils.logging_utils import log_rollout_metrics
    from northrl.utils.trajectory_sequences import iter_training_sequences

    store = GroupedTrajectoryStore(
        OmegaConf.create({"group_key": "group_id", "group_size": 2}), "actor"
    )
    pool = Mock()
    pool.put_trajectory.side_effect = lambda trajectory: (
        "success" if store.put_trajectory(trajectory) else "fail"
    )
    e2b_worker._trajectory_pool = pool

    async def execute(input_data, config, **kwargs):
        del config, kwargs
        first = input_data.rollout_id == "r0"
        return RemoteRolloutResult(
            rollout_id=input_data.rollout_id,
            status=status if first else "success",
            segments=[_remote_segment() for _ in range(segment_count if first else 1)],
            reward=0.0 if first else 1.0,
            termination_reason="context_length_exceeded" if first and segment_count == 0 else None,
            metrics={"swebench_resolved": 0.0 if first else 1.0},
        )

    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", execute)
    for run_id in range(2):
        assert (
            run_coro(
                e2b_worker.execute_rollout(
                    {
                        "instruction": "fix",
                        "template": "template",
                        "task_id": "task",
                        "group_id": "g",
                        "run_id": run_id,
                        "expected_rollouts": 2,
                        "rollout_id": f"r{run_id}",
                    }
                )
            )
            == "success"
        )
        if run_id == 0:
            assert store.take_groups(num_groups=1) is None

    assert pool.put_trajectory.call_count == 2
    trajectories = store.take_groups(num_groups=1)
    assert len(trajectories) == 2
    first, second = trajectories
    assert first["e2b_status"] == status
    assert first["group_id"] == second["group_id"] == "g"
    assert (first["rollout_id"], second["rollout_id"]) == ("r0", "r1")
    assert first["expected_rollouts"] == 2
    assert len(first.segments) == (0 if status == "env_error" else segment_count)
    compute_grpo_advantage_for_trajectories(trajectories)
    if status == "env_error":
        assert first.status == "failed"
        assert first.reward is None
        assert "reward" not in first["score"]
        assert "swebench_resolved" not in first["score"]
        assert all(trajectory["group_skipped"] for trajectory in trajectories)
        assert list(iter_training_sequences(trajectories)) == []
    else:
        assert first.status == "success"
        assert first.reward == 0.0
        assert not first["group_skipped"]
        assert first["advantage"] == pytest.approx(-1.0, abs=2e-6)
        assert second["advantage"] == pytest.approx(1.0, abs=2e-6)
        assert len(list(iter_training_sequences(trajectories))) == segment_count + 1

    metrics = {}
    log_rollout_metrics(trajectories, metrics)
    assert metrics["batch/trajectory_count"] == 2
    assert metrics["trajectory/reward_observed_count"] == (1 if status == "env_error" else 2)
    assert metrics["batch/failed_trajectory_count"] == int(status == "env_error")


@pytest.mark.parametrize("has_tracker", [True, False])
def test_live_events_preserved_for_environment_failure(e2b_worker, monkeypatch, has_tracker):
    tracker = Mock() if has_tracker else None
    e2b_worker._activity_tracker = tracker
    e2b_worker._put_trajectory = Mock(return_value="success")
    events = []

    async def execute(input_data, config, **kwargs):
        del config
        event = {
            "kind": "rollout_finished",
            "rollout_id": input_data.rollout_id,
            "status": "env_error",
        }
        events.append(event)
        assert ("event_callback" in kwargs) is has_tracker
        if has_tracker:
            kwargs["event_callback"](event)
        return RemoteRolloutResult(rollout_id=input_data.rollout_id, status="env_error", reward=0.0)

    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", execute)
    assert run_coro(e2b_worker.rollout({"instruction": "fix", "template": "template"})) == "success"
    assert e2b_worker._put_trajectory.call_args.args[0].status == "failed"
    if has_tracker:
        tracker.record_e2b_event.assert_called_once_with(events[0])


def test_context_stop_preserves_reward_and_marks_all_nested_sequences(e2b_worker, monkeypatch):
    details = {"prompt_tokens": 1024, "requested_max_tokens": 512, "max_sequence_length": 1024}
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        segments=[_remote_segment(), _remote_segment()],
        reward=0.75,
        termination_reason="context_length_exceeded",
        length_details=details,
        metrics={"swebench_resolved": 1.0},
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", AsyncMock(return_value=result))
    e2b_worker._put_trajectory = Mock(return_value="success")
    assert run_coro(e2b_worker.rollout({"instruction": "fix", "template": "template"})) == "success"
    e2b_worker._put_trajectory.assert_called_once()
    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert trajectory.reward == 0.75
    assert trajectory["score"]["swebench_resolved"] == 1.0
    assert trajectory["e2b_termination_reason"] == "context_length_exceeded"
    assert trajectory["e2b_length_details"] == details
    assert len(trajectory.segments) == 2
    for segment in trajectory.segments:
        assert segment["tokens"] == _remote_segment().tokens
        assert segment["finish_reason"] == "length"
        assert segment["is_truncated"] is False


@pytest.mark.parametrize("status", ["success", "task_failed", "env_error"])
def test_request_telemetry_stays_out_of_trajectory_score_averages(e2b_worker, monkeypatch, status):
    from northrl.remote_rollout.metrics import REQUEST_DURATION_BUCKET_METRICS
    from northrl.utils.logging_utils import log_rollout_metrics

    request_metrics = {
        "model_requests_started": 4.0,
        "model_requests_completed": 3.0,
        "model_requests_success": 2.0,
        "model_requests_error": 1.0,
        "model_requests_timeout": 0.0,
        "model_requests_cancelled": 0.0,
        "model_request_seconds_sum": 60.0,
        "model_request_seconds_max": 30.0,
        "model_request_inflight": 1.0,
        "model_request_oldest_inflight_seconds": 5.0,
        **dict.fromkeys(REQUEST_DURATION_BUCKET_METRICS, 3.0),
    }
    result = RemoteRolloutResult(
        rollout_id="remote-id",
        status=status,
        segments=[_remote_segment()],
        reward=1.0 if status == "success" else 0.0,
        metrics={
            **request_metrics,
            "swebench_resolved": float(status == "success"),
            "model_turns": 2.0,
            "generated_tokens": 20.0,
        },
    )
    original_metrics = dict(result.metrics)
    event = {"kind": "snapshot", "rollout_id": result.rollout_id, "metrics": original_metrics}
    tracker = Mock()
    e2b_worker._activity_tracker = tracker
    e2b_worker._put_trajectory = Mock(return_value="success")

    async def execute(*args, **kwargs):
        kwargs["event_callback"](event)
        return result

    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", execute)
    assert run_coro(e2b_worker.rollout({"instruction": "fix", "template": "template"})) == "success"
    trajectory = e2b_worker._put_trajectory.call_args.args[0]
    assert not request_metrics.keys() & trajectory["score"].keys()
    assert trajectory["score"]["model_turns"] == 2.0
    assert trajectory["score"]["generated_tokens"] == 20.0
    if status != "env_error":
        assert trajectory["score"]["swebench_resolved"] == float(status == "success")

    rollout_metrics = {}
    log_rollout_metrics([trajectory], rollout_metrics)
    assert not any(key.startswith("rollout/model_request") for key in rollout_metrics)
    assert "rollout/model_turns" not in rollout_metrics
    assert (
        trajectory["runtime_metrics"]["model_requests"] == request_metrics["model_requests_started"]
    )
    tracker.record_e2b_event.assert_called_once_with(event)
    assert result.metrics == original_metrics


def test_worker_allows_trainer_side_r2_replay(e2b_worker):
    config = e2b_worker._config
    config.algorithm.router_replay = {"enabled": True, "mode": "R2"}
    assert isinstance(E2BRolloutWorker(config), E2BRolloutWorker)


@pytest.fixture
def profiled_e2b_worker(e2b_worker, tmp_path):
    config = e2b_worker._config
    config.rollout.execution = {
        "timeout_seconds": 120,
        "env_error_retries": 3,
        "gateway_publication_timeout_seconds": 45,
        "token_prefix_reuse": False,
        "tokenizer": "common-tokenizer",
        "profiles": {
            "mimo_code": {
                "entrypoint": {
                    "target": str(tmp_path / "code.py"),
                    "task_files_root": str(tmp_path / "code-evaluation"),
                },
                "input_fields": ["instruction", "cwd"],
                "timeout_seconds": 240,
                "env_error_retries": 0,
                "token_prefix_reuse": True,
                "tokenizer": "code-tokenizer",
            },
            "swe": {
                "entrypoint": {
                    "target": str(tmp_path / "swe.py"),
                    "task_files_root": str(tmp_path / "swe-evaluation"),
                },
                "input_fields": ["instruction", "interface"],
            },
        },
    }
    worker = E2BRolloutWorker(config)
    worker.init_inference_service_client(e2b_worker._service_holder)
    worker._put_trajectory = Mock(return_value="success")
    return worker


@pytest.mark.parametrize("validation", [False, True])
def test_profiles_route_inputs_config_and_assets_with_colliding_task_ids(
    profiled_e2b_worker, monkeypatch, tmp_path, validation
):
    worker = profiled_e2b_worker
    if validation:
        worker.begin_validate()
    artifacts = {
        name: e2b_executor.PreparedE2BArtifacts(
            entrypoint=e2b_executor._EntrypointSpec(script_path=tmp_path / f"{name}.py"),
            source_archive=name.encode(),
            evaluation_archives={"shared-task-id": f"{name}-evaluation".encode()},
        )
        for name in ("mimo_code", "swe")
    }
    worker.set_prepared_artifacts_by_profile(artifacts)
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    for profile in ("mimo_code", "swe", "mimo_code"):
        task = {
            "rollout_profile": profile,
            "data_source": profile,
            "template": f"{profile}-template",
            "task_id": "shared-task-id",
            "instruction": "Fix the bug",
            "cwd": "/workspace",
            "interface": "Public API",
            "hidden_test": "must stay on the host",
        }
        assert run_coro(worker.rollout(task)) == "success"
        trajectory = worker._put_trajectory.call_args.args[0]
        assert trajectory["rollout_profile"] == profile
        assert trajectory["data_source"] == profile
        assert trajectory["task_id"] == "shared-task-id"

    for call, profile in zip(executor.await_args_list, ("mimo_code", "swe", "mimo_code")):
        request, config = call.args
        is_code = profile == "mimo_code"
        assert request.task == {
            "instruction": "Fix the bug",
            **({"cwd": "/workspace"} if is_code else {"interface": "Public API"}),
        }
        assert request.metadata["task_id"] == "shared-task-id"
        assert request.metadata["rollout_profile"] == profile
        assert call.kwargs["template"] == f"{profile}-template"
        assert call.kwargs["timeout_seconds"] == (240 if is_code else 120)
        assert call.kwargs["prepared_artifacts"] is artifacts[profile]
        assert "profiles" not in config.rollout.execution
        entrypoint = config.rollout.execution.entrypoint
        assert entrypoint.target == str(tmp_path / ("code.py" if is_code else "swe.py"))
        assert entrypoint.task_files_root == str(
            tmp_path / ("code-evaluation" if is_code else "swe-evaluation")
        )
        assert config.rollout.execution.env_error_retries == (0 if is_code else 3)
        assert config.rollout.execution.token_prefix_reuse is is_code
        assert config.rollout.execution.tokenizer == (
            "code-tokenizer" if is_code else "common-tokenizer"
        )
        assert config.rollout.sampling == worker._config.rollout.sampling
        assert config.models == worker._config.models
    # Configs and archives are selected from initialization caches for every task.
    assert executor.await_args_list[0].args[1] is executor.await_args_list[2].args[1]
    assert executor.await_args_list[0].args[1] is not executor.await_args_list[1].args[1]
    assert "profiles" in worker._config.rollout.execution
    assert worker._service_holder.get_current_gateway_snapshot.await_count == 3
    for call in worker._service_holder.get_current_gateway_snapshot.await_args_list:
        assert call.kwargs == {"timeout_seconds": 45}


@pytest.mark.parametrize("profile", [None, "", "unknown", "../swe.py", 123, ["swe"]])
def test_profiles_reject_missing_or_unknown_task_profile_before_gateway(
    profiled_e2b_worker, monkeypatch, profile
):
    worker = profiled_e2b_worker
    executor = AsyncMock()
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)
    task = {"template": "task-template"}
    if profile is not None:
        task["rollout_profile"] = profile

    with pytest.raises(ValueError, match="task.rollout_profile"):
        run_coro(worker.rollout(task))

    worker._service_holder.get_current_gateway_snapshot.assert_not_awaited()
    executor.assert_not_awaited()
    worker._put_trajectory.assert_not_called()


def test_profiles_never_fall_back_to_legacy_artifacts(profiled_e2b_worker, monkeypatch, tmp_path):
    worker = profiled_e2b_worker
    worker.set_prepared_artifacts(
        e2b_executor.PreparedE2BArtifacts(
            entrypoint=e2b_executor._EntrypointSpec(script_path=tmp_path / "legacy.py"),
            source_archive=b"legacy source",
            evaluation_archives={"shared-task-id": b"legacy evaluation"},
        )
    )
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    assert (
        run_coro(
            worker.rollout(
                {
                    "rollout_profile": "swe",
                    "template": "swe-template",
                    "task_id": "shared-task-id",
                }
            )
        )
        == "success"
    )
    assert "prepared_artifacts" not in executor.await_args.kwargs
    assert executor.await_args.args[1].rollout.execution.entrypoint.target.endswith("swe.py")


def test_profiles_require_artifact_registry_to_match_config(profiled_e2b_worker):
    with pytest.raises(ValueError, match="keys must match"):
        profiled_e2b_worker.set_prepared_artifacts_by_profile({})


def test_profile_without_configured_timeout_uses_default(profiled_e2b_worker, monkeypatch):
    config = profiled_e2b_worker._config
    del config.rollout.execution.timeout_seconds
    worker = E2BRolloutWorker(config)
    worker.init_inference_service_client(profiled_e2b_worker._service_holder)
    worker._put_trajectory = Mock(return_value="success")
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    assert (
        run_coro(worker.rollout({"rollout_profile": "swe", "template": "swe-template"}))
        == "success"
    )
    assert executor.await_args.kwargs["timeout_seconds"] == 3600


def test_explicit_agent_budget_is_forwarded_in_legacy_executor_argument(e2b_worker, monkeypatch):
    config = e2b_worker._config
    config.rollout.execution.agent_timeout_seconds = 900
    config.rollout.execution.preparation_timeout_seconds = 30
    worker = E2BRolloutWorker(config)
    worker.init_inference_service_client(e2b_worker._service_holder)
    worker._put_trajectory = Mock(return_value="success")
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    assert run_coro(worker.rollout({"template": "task-template"})) == "success"

    assert executor.await_args.kwargs["timeout_seconds"] == 900
    assert executor.await_args.args[1].rollout.execution.timeout_seconds == 120
    assert executor.await_args.args[1].rollout.execution.preparation_timeout_seconds == 30


@pytest.mark.parametrize(
    "key",
    [
        "preparation_timeout_seconds",
        "agent_timeout_seconds",
        "agent_finalize_timeout_seconds",
        "evaluation_timeout_seconds",
    ],
)
@pytest.mark.parametrize("value", [False, 0])
def test_worker_validates_every_stage_budget_before_starting(e2b_worker, key, value):
    config = e2b_worker._config
    config.rollout.execution[key] = value

    with pytest.raises(ValueError, match=rf"rollout\.execution\.{key} must be a positive integer"):
        E2BRolloutWorker(config)

    e2b_worker._service_holder.get_current_gateway_snapshot.assert_not_awaited()


def test_profiles_inherit_and_override_independent_budgets(profiled_e2b_worker, monkeypatch):
    config = profiled_e2b_worker._config
    config.rollout.execution.agent_timeout_seconds = 600
    config.rollout.execution.preparation_timeout_seconds = 50
    config.rollout.execution.evaluation_timeout_seconds = 80
    config.rollout.execution.profiles.mimo_code.agent_timeout_seconds = 900
    config.rollout.execution.profiles.mimo_code.evaluation_timeout_seconds = 100
    worker = E2BRolloutWorker(config)
    worker.init_inference_service_client(profiled_e2b_worker._service_holder)
    worker._put_trajectory = Mock(return_value="success")
    executor = AsyncMock(
        return_value=RemoteRolloutResult(
            rollout_id="remote-id", segments=[_remote_segment()], reward=1.0
        )
    )
    monkeypatch.setattr(worker_module, "run_rollout_in_e2b", executor)

    for profile in ("swe", "mimo_code"):
        assert (
            run_coro(worker.rollout({"rollout_profile": profile, "template": "task-template"}))
            == "success"
        )

    swe_call, code_call = executor.await_args_list
    assert swe_call.kwargs["timeout_seconds"] == 600
    assert code_call.kwargs["timeout_seconds"] == 900
    assert swe_call.args[1].rollout.execution.preparation_timeout_seconds == 50
    assert code_call.args[1].rollout.execution.preparation_timeout_seconds == 50
    assert swe_call.args[1].rollout.execution.evaluation_timeout_seconds == 80
    assert code_call.args[1].rollout.execution.evaluation_timeout_seconds == 100


@pytest.mark.parametrize("profile", ["swe", "mimo_code"])
@pytest.mark.parametrize(
    "key",
    ["agent_finalize_timeout_seconds", "evaluation_timeout_seconds"],
)
def test_invalid_budget_in_any_profile_prevents_worker_initialization(
    profiled_e2b_worker, profile, key
):
    config = profiled_e2b_worker._config
    config.rollout.execution.profiles[profile][key] = False

    with pytest.raises(ValueError, match=rf"rollout\.execution\.{key} must be a positive integer"):
        E2BRolloutWorker(config)

    profiled_e2b_worker._service_holder.get_current_gateway_snapshot.assert_not_awaited()
