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

"""Tests for the Claude Code SWE-bench sandbox entrypoint."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from northrl.remote_rollout.contracts import AgentResult, RolloutEnvironmentError, RolloutInput
from northrl.rollout_entrypoints import swerebench_claude_code_rollout as rollout


class _ClaudeProcess:
    def __init__(self, return_code: int = 0) -> None:
        self.returncode = return_code

    async def wait(self) -> int:
        return self.returncode


class _EvaluatorProcess:
    returncode = 0

    async def communicate(self) -> tuple[bytes, None]:
        return (
            b"SWEBench results starts here\nPASSED\nSWEBench results ends here\n",
            None,
        )


@pytest.fixture
def prepared_network_policy(monkeypatch):
    """Provide a prepared sandbox policy without touching the host's firewall."""

    def environment(environment, agent_user):
        assert agent_user == "northrl-agent"
        return {
            **environment,
            "HTTP_PROXY": "http://127.0.0.1:18080",
            "HTTPS_PROXY": "http://127.0.0.1:18080",
            "NO_PROXY": "localhost,127.0.0.1,::1",
        }

    def command(command, agent_user):
        assert agent_user == "northrl-agent"
        return ["/usr/bin/setpriv", "--no-new-privs", "--", *command]

    monkeypatch.setattr(rollout, "agent_network_environment", environment)
    monkeypatch.setattr(rollout, "agent_network_command", command)


@pytest.mark.parametrize(
    ("task_fields", "expected_prompt"),
    [
        ({}, "fix it"),
        ({"interface": None}, "fix it"),
        ({"interface": ""}, "fix it"),
        ({"interface": " \n"}, "fix it"),
        (
            {"interface": "Expose Rule.set_split_amount(amount).\nPreserve existing behavior."},
            "fix it\n\nExpose Rule.set_split_amount(amount).\nPreserve existing behavior.",
        ),
    ],
)
def test_claude_code_uses_local_api_without_weaver_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_fields: dict[str, Any],
    expected_prompt: str,
    prepared_network_policy,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def create_process(*args: Any, **kwargs: Any) -> _ClaudeProcess:
        calls.append((args, kwargs))
        return _ClaudeProcess()

    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setenv("WEAVER_API_KEY", "secret")
    monkeypatch.setenv("NORTHGATE_API_KEY", "relay-secret")
    monkeypatch.setenv("WEAVER_BASE_URL", "https://weaver.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "wrong-oauth")

    asyncio.run(
        rollout._run_claude_code(
            RolloutInput(
                rollout_id="rollout-1",
                task={
                    "instruction": "fix it",
                    "patch": "reference solution for evaluation",
                    "test_patch": "hidden evaluation tests",
                    "custom": {"other_task_data": True},
                    **task_fields,
                },
            ),
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
                "ANTHROPIC_API_KEY": "northrl",
            },
            workspace=tmp_path / "wtforms",
        )
    )

    [(command, kwargs)] = calls
    assert command == (
        "/usr/bin/setpriv",
        "--no-new-privs",
        "--",
        "runuser",
        "--user",
        "northrl-agent",
        "--preserve-environment",
        "--",
        "/usr/local/bin/claude",
        "--print",
        "--bare",
        "--model",
        "northrl",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        expected_prompt,
    )
    environment = kwargs["env"]
    assert Path(kwargs["cwd"]) == tmp_path / "wtforms"
    assert environment["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert environment["ANTHROPIC_API_KEY"] == "northrl"
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert "WEAVER_API_KEY" not in environment
    assert "NORTHGATE_API_KEY" not in environment
    assert "WEAVER_BASE_URL" not in environment
    assert "ANTHROPIC_AUTH_TOKEN" not in environment
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in environment


@pytest.mark.parametrize("helper", ["agent_network_environment", "agent_network_command"])
def test_network_policy_failure_prevents_agent_launch(
    tmp_path, monkeypatch, prepared_network_policy, helper
):
    create_process = AsyncMock()
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(rollout, "_trusted_runtime_path", lambda *_: "/usr/bin")

    def unavailable(*_):
        raise RolloutEnvironmentError("SWE network policy is unavailable")

    monkeypatch.setattr(rollout, helper, unavailable)
    with pytest.raises(RolloutEnvironmentError, match="network policy is unavailable"):
        asyncio.run(
            rollout._run_claude_code(
                RolloutInput(rollout_id="network-policy", task={"instruction": "fix it"}),
                {
                    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
                    "ANTHROPIC_API_KEY": "test",
                },
                workspace=tmp_path,
            )
        )

    create_process.assert_not_awaited()


def test_claude_launcher_does_not_restrict_network_without_policy(tmp_path, monkeypatch):
    create_process = AsyncMock(return_value=_ClaudeProcess())
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(rollout, "_trusted_runtime_path", lambda *_: "/usr/bin")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)

    asyncio.run(
        rollout._run_claude_code(
            RolloutInput(rollout_id="no-policy", task={"instruction": "fix it"}),
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8000", "ANTHROPIC_API_KEY": "test"},
            workspace=tmp_path,
        )
    )

    command, kwargs = create_process.await_args
    assert command[0] == "runuser"
    assert "HTTP_PROXY" not in kwargs["env"]
    assert "HTTPS_PROXY" not in kwargs["env"]


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({}, "task.instruction must be a non-empty string"),
        ({"instruction": ""}, "task.instruction must be a non-empty string"),
        ({"instruction": "  "}, "task.instruction must be a non-empty string"),
        ({"instruction": 123}, "task.instruction must be a non-empty string"),
        ({"instruction": "fix it", "interface": 123}, "task.interface must be a string"),
        ({"instruction": "fix it", "interface": {}}, "task.interface must be a string"),
    ],
)
def test_claude_code_validates_task_fields_before_launch(tmp_path, monkeypatch, task, message):
    create_process = AsyncMock()
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            rollout._run_claude_code(
                RolloutInput(rollout_id="rollout-1", task=task),
                {},
                workspace=tmp_path,
            )
        )

    create_process.assert_not_awaited()


def test_agent_stops_agent_processes_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def run_claude_code(
        input_data: RolloutInput,
        model_environment: dict[str, str],
        *,
        workspace: Path,
    ) -> int:
        assert input_data.task == {"instruction": "fix it"}
        assert model_environment == {}
        assert workspace == tmp_path / "wtforms"
        events.append("claude")
        return 0

    async def stop_agent_processes() -> None:
        events.append("stop-agent")

    monkeypatch.setattr(rollout, "_run_claude_code", run_claude_code)
    monkeypatch.setattr(
        rollout,
        "_stop_agent_processes",
        stop_agent_processes,
    )
    result = asyncio.run(
        rollout.run_agent(
            RolloutInput(rollout_id="rollout-1", task={"instruction": "fix it"}),
            tmp_path / "wtforms",
            {},
        )
    )

    assert result == AgentResult()
    assert events == ["claude", "stop-agent"]


@pytest.mark.parametrize("exit_code", [0, 1, 124, -9])
def test_agent_preserves_cli_exit_after_cleanup(
    tmp_path, monkeypatch, exit_code, prepared_network_policy
):
    events = []

    async def create_process(*args, **kwargs):
        events.append("claude")
        return _ClaudeProcess(exit_code)

    async def cleanup():
        events.append("stop-agent")

    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(rollout, "_trusted_runtime_path", lambda *_: "/usr/bin")
    monkeypatch.setattr(rollout, "_stop_agent_processes", cleanup)
    result = asyncio.run(
        rollout.run_agent(
            RolloutInput(rollout_id="cli-exit", task={"instruction": "fix it"}),
            tmp_path,
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
                "ANTHROPIC_API_KEY": "test",
            },
        )
    )

    assert result.exit_code == exit_code
    assert events == ["claude", "stop-agent"]


def test_evaluator_parses_reward_without_weaver_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def create_process(*args: Any, **kwargs: Any) -> _EvaluatorProcess:
        calls.append((args, kwargs))
        return _EvaluatorProcess()

    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setenv("WEAVER_API_KEY", "secret")
    monkeypatch.setenv("WEAVER_BASE_URL", "https://weaver.example")
    monkeypatch.setenv("UNRELATED_SECRET", "also-secret")
    monkeypatch.setenv("PYTHONPATH", "/agent-controlled")
    monkeypatch.setenv("PATH", "/trusted/bin")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:18080")
    for helper in ("agent_network_environment", "agent_network_command"):
        monkeypatch.setattr(
            rollout, helper, lambda *_: pytest.fail("root evaluator must keep its own environment")
        )
    monkeypatch.setattr(rollout, "_trusted_runtime_path", lambda *_args: "/trusted/bin")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("TZ", raising=False)

    assert asyncio.run(rollout._run_evaluator(tmp_path / "wtforms")) == 1.0
    [(command, kwargs)] = calls
    assert command == (
        "/bin/bash",
        "/opt/northrl/evaluator/run-tests.sh",
    )
    assert Path(kwargs["cwd"]) == tmp_path / "wtforms"
    assert kwargs["env"] == {
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "PATH": "/trusted/bin",
        "LANG": "C.UTF-8",
    }


def test_stop_agent_processes_accepts_no_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def create_process(*args: Any, **kwargs: Any) -> _ClaudeProcess:
        calls.append((args, kwargs))
        return _ClaudeProcess(1)

    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)

    asyncio.run(rollout._stop_agent_processes())

    assert calls == [
        (
            ("/usr/bin/pkill", "-KILL", "-u", "northrl-agent"),
            {},
        ),
    ]


@pytest.mark.parametrize("failure", [RuntimeError("agent failed"), asyncio.CancelledError()])
def test_agent_cleanup_runs_on_error_and_cancellation(tmp_path, monkeypatch, failure):
    events = []

    async def fail(*args, **kwargs):
        events.append("agent")
        raise failure

    async def cleanup():
        await asyncio.sleep(0)
        events.append("cleaned")

    monkeypatch.setattr(rollout, "_run_claude_code", fail)
    monkeypatch.setattr(rollout, "_stop_agent_processes", cleanup)
    input_data = RolloutInput(rollout_id="failure", task={"instruction": "fix"})
    with pytest.raises(type(failure)):
        asyncio.run(rollout.run_agent(input_data, tmp_path, {}))
    assert events == ["agent", "cleaned"]


def test_agent_cleanup_survives_cancellation_while_cleaning(tmp_path, monkeypatch):
    events = []

    async def agent(*args, **kwargs):
        events.append("agent")

    async def exercise():
        cleaning = asyncio.Event()
        finish = asyncio.Event()

        async def cleanup():
            cleaning.set()
            await finish.wait()
            events.append("cleaned")

        monkeypatch.setattr(rollout, "_stop_agent_processes", cleanup)
        input_data = RolloutInput(rollout_id="cancel", task={"instruction": "fix"})
        task = asyncio.create_task(rollout.run_agent(input_data, tmp_path, {}))
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()  # Repeated cancellation must still leave cleanup running.
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(rollout, "_run_claude_code", agent)
    asyncio.run(exercise())
    assert events == ["agent", "cleaned"]


def test_prepare_never_reads_or_installs_hidden_evaluation_assets(tmp_path, monkeypatch):
    events = []
    evaluation_assets = tmp_path / "evaluation-not-uploaded"
    monkeypatch.setattr(
        rollout, "_prepare_claude", lambda workspace: events.append(("claude", workspace))
    )
    monkeypatch.setattr(
        rollout, "_ensure_agent_user", lambda workspace: events.append(("user", workspace))
    )

    async def prepare_network(agent_user):
        assert agent_user == "northrl-agent"
        events.append(("network", agent_user))

    monkeypatch.setattr(rollout, "prepare_network_policy", prepare_network)
    monkeypatch.setattr(
        rollout,
        "_install_evaluator_assets",
        lambda *args: pytest.fail("prepare must not install evaluator assets"),
    )
    input_data = RolloutInput(rollout_id="prepare", task={"instruction": "fix"})

    asyncio.run(rollout.prepare(input_data, tmp_path))

    assert events == [("claude", tmp_path), ("user", tmp_path), ("network", "northrl-agent")]
    assert not evaluation_assets.exists()
    assert not hasattr(rollout, "_ensure_runtime")
    assert not hasattr(rollout, "main")


def test_prepare_network_failure_aborts_preparation(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(rollout, "_prepare_claude", lambda _: events.append("claude"))
    monkeypatch.setattr(rollout, "_ensure_agent_user", lambda _: events.append("user"))

    async def unavailable(agent_user):
        assert agent_user == "northrl-agent"
        assert events == ["claude", "user"]
        raise RolloutEnvironmentError("SWE network firewall is unavailable")

    monkeypatch.setattr(rollout, "prepare_network_policy", unavailable)
    with pytest.raises(RolloutEnvironmentError, match="network firewall is unavailable"):
        asyncio.run(rollout.prepare(RolloutInput(rollout_id="network-prepare", task={}), tmp_path))


@pytest.fixture
def evaluation_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "_ROOT_UID", os.getuid())
    uploaded = tmp_path / "evaluation"
    uploaded.mkdir()
    destinations = {}
    for name in rollout._EVALUATOR_DESTINATIONS:
        source = uploaded / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"trusted {name}")
        destinations[name] = tmp_path / "installed" / name
    (uploaded / "tests/config.json").write_text(json.dumps({"instance_id": "repo__name-1"}))
    monkeypatch.setattr(rollout, "_EVALUATOR_DESTINATIONS", destinations)
    return uploaded, destinations


@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.parametrize(
    ("task_fields", "metadata"),
    [
        ({"task_id": "repo__name-1"}, {}),
        ({}, {"task_id": "repo__name-1"}),
        ({"task_id": "wrong-legacy-id"}, {"task_id": "repo__name-1"}),
    ],
)
def test_evaluate_installs_delayed_package_before_scoring(
    evaluation_paths, tmp_path, monkeypatch, reward, task_fields, metadata
):
    uploaded, destinations = evaluation_paths
    assert all(not destination.exists() for destination in destinations.values())

    async def evaluator(workspace):
        assert workspace == tmp_path
        for name, destination in destinations.items():
            assert destination.read_bytes() == (uploaded / name).read_bytes()
            assert destination.stat().st_mode & 0o777 == (
                0o700 if name == "run-tests.sh" else 0o600
            )
        return reward

    monkeypatch.setattr(rollout, "_run_evaluator", evaluator)
    input_data = RolloutInput(
        rollout_id="evaluate", task={"instruction": "fix", **task_fields}, metadata=metadata
    )
    result = asyncio.run(rollout.evaluate(input_data, tmp_path, AgentResult(), uploaded))
    assert result.reward == reward
    assert result.status == ("success" if reward == 1.0 else "task_failed")
    assert result.metrics == {"swebench_resolved": reward}


def test_missing_evaluation_asset_fails_before_installation(evaluation_paths):
    uploaded, destinations = evaluation_paths
    (uploaded / "tests/config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        rollout._install_evaluator_assets(uploaded, task_id="repo__name-1")
    assert all(not destination.exists() for destination in destinations.values())


def test_evaluation_asset_directory_symlink_is_rejected(evaluation_paths, tmp_path):
    uploaded, _ = evaluation_paths
    (uploaded / "tests/config.json").unlink()
    (uploaded / "tests").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("untrusted")
    (uploaded / "tests").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        rollout._install_evaluator_assets(uploaded, task_id="repo__name-1")


def test_cleanup_waits_until_agent_processes_have_exited(monkeypatch):
    from types import SimpleNamespace

    checks = iter([True, True, False])
    observed = []

    async def create_process(*args, **kwargs):
        return _ClaudeProcess(0)

    async def sleep(delay):
        observed.append(delay)

    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(rollout.asyncio, "sleep", sleep)
    monkeypatch.setattr(rollout.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=1234))
    monkeypatch.setattr(rollout, "_agent_processes_running", lambda uid: next(checks))

    asyncio.run(rollout._stop_agent_processes())
    assert observed == [0.05, 0.05]


def test_cleanup_failure_prevents_successful_agent_return(tmp_path, monkeypatch):
    from northrl.remote_rollout.contracts import RolloutEnvironmentError

    async def agent(*args, **kwargs):
        return None

    async def cleanup():
        raise RolloutEnvironmentError("agent still alive")

    monkeypatch.setattr(rollout, "_run_claude_code", agent)
    monkeypatch.setattr(rollout, "_stop_agent_processes", cleanup)
    input_data = RolloutInput(rollout_id="cleanup-failed", task={"instruction": "fix"})
    with pytest.raises(RolloutEnvironmentError, match="still alive"):
        asyncio.run(rollout.run_agent(input_data, tmp_path, {}))


@pytest.mark.parametrize(
    "asset", ["run-tests.sh", "tests/config.json", "log_parsers.py", "swe_constants.py"]
)
def test_prepare_allows_preinstalled_evaluator_assets(tmp_path, monkeypatch, asset):
    installed = tmp_path / "installed" / asset
    installed.parent.mkdir(parents=True)
    installed.write_text("preinstalled evaluator")
    monkeypatch.setattr(rollout, "_EVALUATOR_DESTINATIONS", {asset: installed})
    events = []
    monkeypatch.setattr(
        rollout,
        "_prepare_claude",
        lambda workspace: events.append(("claude", workspace)),
    )
    monkeypatch.setattr(
        rollout,
        "_ensure_agent_user",
        lambda workspace: events.append(("user", workspace)),
    )
    network = AsyncMock()
    monkeypatch.setattr(rollout, "prepare_network_policy", network)
    input_data = RolloutInput(rollout_id="preinstalled-template", task={"instruction": "fix"})

    asyncio.run(rollout.prepare(input_data, tmp_path))

    assert events == [("claude", tmp_path), ("user", tmp_path)]
    network.assert_awaited_once_with("northrl-agent")
    assert installed.read_text() == "preinstalled evaluator"


@pytest.mark.parametrize("instance_id", ["another__repo-2", None])
def test_evaluate_rejects_assets_for_a_different_task(
    evaluation_paths, tmp_path, monkeypatch, instance_id
):
    uploaded, destinations = evaluation_paths
    (uploaded / "tests/config.json").write_text(json.dumps({"instance_id": instance_id}))
    monkeypatch.setattr(
        rollout,
        "_run_evaluator",
        lambda workspace: pytest.fail("mismatched evaluator must never run"),
    )
    input_data = RolloutInput(
        rollout_id="mismatch", task={"instruction": "fix", "task_id": "repo__name-1"}
    )
    with pytest.raises(ValueError, match="instance_id does not match task_id"):
        asyncio.run(rollout.evaluate(input_data, tmp_path, AgentResult(), uploaded))
    assert all(not destination.exists() for destination in destinations.values())


@pytest.mark.parametrize("task_id", [None, "", "  ", 1])
@pytest.mark.parametrize("metadata_id", [False, True])
def test_evaluate_requires_a_task_id(evaluation_paths, tmp_path, monkeypatch, task_id, metadata_id):
    uploaded, destinations = evaluation_paths
    evaluator = AsyncMock()
    monkeypatch.setattr(rollout, "_run_evaluator", evaluator)
    input_data = RolloutInput(
        rollout_id="missing-id",
        task={"instruction": "fix", "task_id": "repo__name-1" if metadata_id else task_id},
        metadata={"task_id": task_id} if metadata_id else {},
    )
    with pytest.raises(ValueError, match="requires a non-empty metadata.task_id"):
        asyncio.run(rollout.evaluate(input_data, tmp_path, AgentResult(), uploaded))
    assert all(not destination.exists() for destination in destinations.values())
    evaluator.assert_not_awaited()


def test_swe_evaluation_entrypoint_writes_canonical_segments(
    evaluation_paths, tmp_path, monkeypatch
):
    """Preserve the main entrypoint's wire contract through the new task hooks."""
    from northrl.remote_rollout import e2b_runtime
    from northrl.remote_rollout.contracts import (
        AgentStageResult,
        RemoteRolloutTrace,
        RemoteSegment,
    )
    from northrl.remote_rollout.wire import (
        decode_result,
        dump_artifact,
        encode_agent_stage,
        load_result_gzip,
    )

    uploaded, _destinations = evaluation_paths
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    platform = tmp_path / "platform"
    platform.mkdir()
    input_data = RolloutInput(
        rollout_id="swe-segments", task={"instruction": "fix it", "task_id": "repo__name-1"}
    )
    input_path = platform / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "input_data": input_data.model_dump(),
                "config": {},
                "gateway": {
                    "base_url": "https://northgate.example",
                    "model": "weaver-sampler-1",
                    "sampling_session_id": "sampler-1",
                    "weight_version": "step-1",
                },
                "attempt_id": "attempt-1",
            }
        )
    )
    segment = RemoteSegment(
        tokens=[1, 2], loss_mask=[0, 1], log_probs=[0.0, -0.1], old_log_probs=[0.0, -0.1]
    )
    trace = RemoteRolloutTrace(rollout_id=input_data.rollout_id, segments=[segment])
    stage = AgentStageResult(
        rollout_id=input_data.rollout_id,
        attempt_id="attempt-1",
        workspace=str(workspace),
        agent_result=AgentResult(),
        trace=trace,
    )
    stage_path = platform / "agent-stage.json.gz"
    stage_path.write_bytes(dump_artifact(encode_agent_stage(stage)))
    output_path = platform / "result.json.gz"

    async def evaluate_score(actual_workspace):
        assert actual_workspace == workspace
        return 1.0

    monkeypatch.setattr(rollout, "_run_evaluator", evaluate_score)
    monkeypatch.setattr(e2b_runtime, "_load_adapter", lambda _path: rollout)
    asyncio.run(
        e2b_runtime._run_phase(
            phase="evaluate",
            input_path=input_path,
            output_path=output_path,
            trace_path=platform / "trace.json.gz",
            stage_path=stage_path,
            entrypoint=str(Path(rollout.__file__)),
            evaluation_assets=uploaded,
        )
    )

    result_bytes = output_path.read_bytes()
    assert result_bytes.startswith(b"\x1f\x8b")
    compact_result = load_result_gzip(result_bytes)
    result_json = compact_result.model_dump(mode="json")
    assert "segments" in result_json
    assert "trajectories" not in result_json
    assert result_json["schema_version"] == 2
    assert "tokens" not in result_json["segments"][0]
    result = decode_result(compact_result)
    assert result.segments == [segment]
    assert result.reward == 1.0
    assert result.metrics["swebench_resolved"] == 1.0
