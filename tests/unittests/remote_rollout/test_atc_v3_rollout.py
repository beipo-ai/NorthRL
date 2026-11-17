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

"""Test NorthRL's adapter boundary with a synthetic template runtime."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout.contracts import (
    AgentResult,
    RolloutEnvironmentError,
    RolloutInput,
)
from northrl.rollout_entrypoints import atc_v3_rollout as rollout


def _input(**task):
    return RolloutInput(
        rollout_id="test-rollout",
        metadata={"task_id": "task_00000"},
        task={"instruction": "Complete the operation", **task},
    )


@pytest.fixture
def runtime(monkeypatch):
    module = SimpleNamespace(
        prepare=Mock(),
        run_agent=AsyncMock(return_value={"exit_code": 2, "final_answer": "completed"}),
        evaluate=Mock(return_value={"score": 0.5, "passed_checks": 1, "check_count": 2}),
    )
    monkeypatch.setattr(rollout, "_load_runtime", lambda: module)
    return module


def test_prepare_supplies_bound_task_identity(runtime):
    input_data = _input()
    workspace = rollout.resolve_workspace(input_data)
    asyncio.run(rollout.prepare(input_data, workspace))
    runtime.prepare.assert_called_once_with(
        {"task_id": "task_00000", "instruction": "Complete the operation"}, workspace
    )
    assert "task_id" not in input_data.task


def test_identity_mismatch_is_rejected_before_loading_runtime(monkeypatch):
    monkeypatch.setattr(rollout, "_load_runtime", lambda: pytest.fail("invalid task identity"))
    with pytest.raises(RolloutEnvironmentError, match="does not match"):
        asyncio.run(rollout.prepare(_input(task_id="task_00001"), rollout.WORKSPACE))


def test_agent_result_and_gateway_environment_cross_adapter(runtime):
    environment = {
        "ANTHROPIC_BASE_URL": "http://localhost:8000",
        "ANTHROPIC_API_KEY": "northrl",
    }
    result = asyncio.run(rollout.run_agent(_input(), rollout.WORKSPACE, environment))
    assert result == AgentResult(exit_code=2, final_answer="completed")
    assert runtime.run_agent.call_args.args[2] is environment


def test_adapter_waits_for_runtime_completion_before_returning(runtime):
    async def exercise():
        cleanup_started, cleanup_done = asyncio.Event(), asyncio.Event()

        async def run(*_):
            cleanup_started.set()
            await cleanup_done.wait()
            return {"exit_code": 0, "final_answer": "done"}

        runtime.run_agent.side_effect = run
        result = asyncio.create_task(rollout.run_agent(_input(), rollout.WORKSPACE, {}))
        await cleanup_started.wait()
        assert not result.done()
        cleanup_done.set()
        assert (await result).final_answer == "done"

    asyncio.run(exercise())


@pytest.mark.parametrize("score", [0, 0.5, 1])
def test_evaluation_preserves_score_counts_and_agent_diagnostics(runtime, tmp_path, score):
    runtime.evaluate.return_value = {
        "score": score,
        "passed_checks": int(score * 2),
        "check_count": 2,
    }
    agent = AgentResult(exit_code=2, final_answer="answer")
    result = asyncio.run(rollout.evaluate(_input(), rollout.WORKSPACE, agent, tmp_path))
    assert result.reward == score
    assert result.status == ("task_failed" if score == 0 else "success")
    assert result.metrics == {
        "atc_passed_checks": float(score * 2),
        "atc_check_count": 2.0,
        "atc_agent_exit_code": 2.0,
    }
    assert runtime.evaluate.call_args.args[2:] == (agent.model_dump(), tmp_path)


@pytest.mark.parametrize("stage", ["prepare", "run_agent", "evaluate"])
def test_explicit_external_environment_errors_remain_environment_errors(runtime, tmp_path, stage):
    error = RuntimeError("external runtime failed")
    error.rollout_status = "env_error"
    getattr(runtime, stage).side_effect = error
    calls = {
        "prepare": lambda: rollout.prepare(_input(), rollout.WORKSPACE),
        "run_agent": lambda: rollout.run_agent(_input(), rollout.WORKSPACE, {}),
        "evaluate": lambda: rollout.evaluate(_input(), rollout.WORKSPACE, AgentResult(), tmp_path),
    }
    with pytest.raises(RolloutEnvironmentError) as raised:
        asyncio.run(calls[stage]())
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("stage", ["prepare", "run_agent"])
def test_unclassified_setup_and_agent_failures_remain_environment_errors(runtime, stage):
    getattr(runtime, stage).side_effect = OSError("runtime failure")
    call = (
        rollout.prepare(_input(), rollout.WORKSPACE)
        if stage == "prepare"
        else rollout.run_agent(_input(), rollout.WORKSPACE, {})
    )
    with pytest.raises(RolloutEnvironmentError):
        asyncio.run(call)


@pytest.mark.parametrize(
    "failure", [RuntimeError, AttributeError, NotImplementedError, TimeoutError, OSError]
)
def test_scoring_failures_return_zero_without_fabricated_check_counts(
    runtime, tmp_path, caplog, failure
):
    error = failure("broken scoring implementation")
    error.rollout_status = "template_specific_failure"
    runtime.evaluate.side_effect = error
    result = asyncio.run(
        rollout.evaluate(_input(), rollout.WORKSPACE, AgentResult(exit_code=7), tmp_path)
    )
    assert result.reward == 0.0
    assert result.status == "task_failed"
    assert result.metrics == {"atc_agent_exit_code": 7.0}
    assert "assigning reward 0" in caplog.text
    assert "broken scoring implementation" in caplog.text


def test_invalid_agent_result_is_environment_error(runtime):
    runtime.run_agent.return_value = {"exit_code": "0"}
    with pytest.raises(RolloutEnvironmentError):
        asyncio.run(rollout.run_agent(_input(), rollout.WORKSPACE, {}))


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"score": True, "passed_checks": 1, "check_count": 1},
        {"score": float("nan"), "passed_checks": 1, "check_count": 1},
        {"score": 1.1, "passed_checks": 1, "check_count": 1},
        {"score": 0, "passed_checks": 0, "check_count": 0},
        {"score": 0, "passed_checks": True, "check_count": 1},
    ],
)
def test_invalid_evaluation_output_returns_zero(runtime, tmp_path, result):
    runtime.evaluate.return_value = result
    evaluation = asyncio.run(
        rollout.evaluate(_input(), rollout.WORKSPACE, AgentResult(exit_code=3), tmp_path)
    )
    assert evaluation.reward == 0.0
    assert evaluation.status == "task_failed"
    assert evaluation.metrics == {"atc_agent_exit_code": 3.0}


@pytest.fixture
def installed_runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    package = root / "atc_v3_runtime"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text(
        "API_VERSION = 1\n"
        "def prepare(*args): pass\n"
        "async def run_agent(*args): return {'exit_code': 0}\n"
        "def evaluate(*args): return {'score': 1, 'passed_checks': 1, 'check_count': 1}\n"
    )
    monkeypatch.setattr(rollout, "_RUNTIME_ROOT", root)
    monkeypatch.setattr(rollout.os, "geteuid", lambda: 0)
    monkeypatch.setitem(sys.modules, "atc_v3_runtime", None)
    original = Path.lstat

    def root_owned(path):
        values = list(original(path))
        values[4] = 0
        if path not in (root, package) and root not in path.parents:
            values[0] &= ~0o022
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", root_owned)
    return package


def test_loads_api_only_from_protected_template_path(installed_runtime):
    module = rollout._load_runtime()
    assert module.API_VERSION == 1
    assert module.__file__ == str(installed_runtime / "__init__.py")
    assert rollout._load_runtime() is module


@pytest.mark.parametrize("entry", ["package", "source"])
def test_runtime_writable_entry_is_rejected(installed_runtime, entry):
    target = installed_runtime if entry == "package" else installed_runtime / "__init__.py"
    target.chmod(0o777)
    with pytest.raises(RolloutEnvironmentError, match="untrusted"):
        rollout._load_runtime()


def test_runtime_non_root_owned_entry_is_rejected(installed_runtime, monkeypatch):
    original = Path.lstat

    def untrusted(path):
        values = list(original(path))
        if path == installed_runtime / "__init__.py":
            values[4] = 1001
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", untrusted)
    with pytest.raises(RolloutEnvironmentError, match="untrusted"):
        rollout._load_runtime()


def test_runtime_symlink_is_rejected_without_executing_target(installed_runtime, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('must not execute')")
    initializer = installed_runtime / "__init__.py"
    initializer.unlink()
    initializer.symlink_to(outside)
    with pytest.raises(RolloutEnvironmentError, match="untrusted"):
        rollout._load_runtime()


def test_runtime_api_version_is_checked(installed_runtime):
    (installed_runtime / "__init__.py").write_text("API_VERSION = 2")
    with pytest.raises(RolloutEnvironmentError) as error:
        rollout._load_runtime()
    assert "API_VERSION 1" in str(error.value.__cause__)


def test_previously_imported_runtime_from_other_path_is_rejected(installed_runtime, monkeypatch):
    del installed_runtime
    monkeypatch.setitem(
        sys.modules,
        "atc_v3_runtime",
        SimpleNamespace(__file__="/workspace/untrusted.py"),
    )
    with pytest.raises(RolloutEnvironmentError) as error:
        rollout._load_runtime()
    assert "unexpected path" in str(error.value.__cause__)


def test_loader_failure_remains_environment_error_during_evaluation(monkeypatch, tmp_path):
    def missing():
        raise RolloutEnvironmentError("missing installed runtime")

    monkeypatch.setattr(rollout, "_load_runtime", missing)
    with pytest.raises(RolloutEnvironmentError, match="missing installed runtime"):
        asyncio.run(rollout.evaluate(_input(), rollout.WORKSPACE, AgentResult(), tmp_path))


@pytest.mark.parametrize(
    "failure", [TimeoutError("scoring timed out"), AttributeError("unsupported ctx")]
)
def test_scoring_failure_keeps_real_runtime_trace_in_zero_reward_result(
    runtime, tmp_path, monkeypatch, failure
):
    from northrl.remote_rollout import e2b_runtime
    from tests.unittests.remote_rollout import test_e2b_runtime as runtime_tests

    runtime_tests._fake_runtime_dependencies.__wrapped__(monkeypatch)
    adapter, events, options = runtime_tests._phase_setup(tmp_path, monkeypatch)
    runtime.evaluate.side_effect = failure

    async def evaluate(_input_data, workspace, agent_result, evaluation_assets):
        # Bind the generic staged fixture to a valid ATC task identity.
        return await rollout.evaluate(_input(), workspace, agent_result, evaluation_assets)

    adapter.evaluate = evaluate
    asyncio.run(e2b_runtime._run_phase(phase="agent", **options))
    stage = runtime_tests._load_dense_agent_stage(options["stage_path"].read_bytes())
    assert events == ["prepare", "agent"]
    assert len(stage.trace.segments) == 1
    asyncio.run(e2b_runtime._run_phase(phase="evaluate", **options))
    result = runtime_tests._load_dense_result(options["output_path"].read_bytes())
    assert result.status == "task_failed"
    assert result.reward == 0.0
    assert result.error_type is None
    assert result.segments == stage.trace.segments
    assert list(result.segments[0].tokens) == [1, 2, 3]
    assert result.metrics["atc_agent_exit_code"] == stage.agent_result.exit_code
    assert "atc_passed_checks" not in result.metrics
    assert "atc_check_count" not in result.metrics
