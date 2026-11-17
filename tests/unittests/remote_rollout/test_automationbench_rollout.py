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

"""Check the three hooks against the installed template's public interfaces."""

import asyncio
import json
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout.contracts import AgentResult, RolloutEnvironmentError, RolloutInput
from northrl.remote_rollout.network_policy import network_policy_context
from northrl.rollout_entrypoints import automationbench as rollout


@pytest.fixture
def case(tmp_path, monkeypatch):
    task = dict(task_id="task-001", source_task_id="source-1", task_sha256="a" * 64)
    task["initial_state"] = {"keep": 1, "drop": None}
    runner = Mock()
    runner.load_task.return_value = task
    runner.strip_none_values.return_value = {"keep": 1}
    rubric = SimpleNamespace(partial_credit=Mock(), task_completed_correctly=Mock())
    package = SimpleNamespace(__path__=[])
    monkeypatch.setitem(sys.modules, "agent_runner", runner)
    monkeypatch.setitem(sys.modules, "automationbench", package)
    monkeypatch.setitem(sys.modules, "automationbench.rubric", rubric)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(rollout, "_BUNDLE_ROOT", tmp_path / "bundle")
    monkeypatch.setenv("AUTOMATIONBENCH_STRICT_ASSERTIONS", "0")
    return SimpleNamespace(
        workspace=tmp_path,
        task=task,
        runner=runner,
        rubric=rubric,
        input=RolloutInput(
            rollout_id="attempt-1",
            task={"task_sha256": task["task_sha256"]},
            metadata={"task_id": task["task_id"]},
        ),
    )


@pytest.mark.parametrize("change", [None, "hash", "content", "existing_output"])
def test_prepare_checks_template_task_and_fresh_workspace(case, change):
    if change == "hash":
        case.input.task["task_sha256"] = "b" * 64
    elif change == "content":
        case.runner.load_task.side_effect = ValueError("invalid template task content")
    elif change == "existing_output":
        (case.workspace / "automationbench-output").mkdir()
    if change:
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(rollout.prepare(case.input, case.workspace))
    else:
        asyncio.run(rollout.prepare(case.input, case.workspace))
    case.runner.load_task.assert_called_once_with(rollout._BUNDLE_ROOT, "task-001")


def test_configured_policy_prepares_only_nonroot_runner_workspace(case, monkeypatch):
    identity = SimpleNamespace(pw_uid=1000, pw_gid=1000, pw_dir="/home/user")
    monkeypatch.setattr(rollout.pwd, "getpwnam", Mock(return_value=identity))
    ownership = Mock()
    monkeypatch.setattr(rollout.os, "fchown", ownership)
    prepare = AsyncMock()
    monkeypatch.setattr(rollout, "prepare_network_policy", prepare)
    with network_policy_context({"deny_domains": ["example.org"]}, task_id="task-001"):
        asyncio.run(rollout.prepare(case.input, case.workspace))
    ownership.assert_called_once()
    assert ownership.call_args.args[1:] == (1000, 1000)
    assert case.workspace.stat().st_mode & 0o777 == 0o750
    prepare.assert_awaited_once_with("user")


@pytest.mark.parametrize("identity", [None, (0, 1000), (1000, 0)])
def test_configured_policy_rejects_missing_or_root_agent(case, monkeypatch, identity):
    lookup = Mock(
        side_effect=KeyError("user") if identity is None else None,
        return_value=SimpleNamespace(pw_uid=identity[0], pw_gid=identity[1]) if identity else None,
    )
    monkeypatch.setattr(rollout.pwd, "getpwnam", lookup)
    prepare = AsyncMock()
    monkeypatch.setattr(rollout, "prepare_network_policy", prepare)
    with network_policy_context({"deny_domains": ["example.org"]}):
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(rollout.prepare(case.input, case.workspace))
    prepare.assert_not_awaited()


def test_configured_policy_rejects_workspace_symlink(case, monkeypatch):
    identity = SimpleNamespace(pw_uid=1000, pw_gid=1000)
    monkeypatch.setattr(rollout.pwd, "getpwnam", Mock(return_value=identity))
    link = case.workspace / "workspace-link"
    link.symlink_to(case.workspace, target_is_directory=True)
    ownership = Mock()
    monkeypatch.setattr(rollout.os, "fchown", ownership)
    prepare = AsyncMock()
    monkeypatch.setattr(rollout, "prepare_network_policy", prepare)
    with network_policy_context({"deny_domains": ["example.org"]}):
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(rollout.prepare(case.input, link))
    ownership.assert_not_called()
    prepare.assert_not_awaited()


def test_disabled_policy_preserves_original_user_and_workspace(case, monkeypatch):
    monkeypatch.setattr(rollout.pwd, "getpwnam", Mock(side_effect=AssertionError("unused")))
    monkeypatch.setattr(rollout.os, "fchown", Mock(side_effect=AssertionError("unused")))
    prepare = AsyncMock()
    monkeypatch.setattr(rollout, "prepare_network_policy", prepare)
    with network_policy_context({"deny_domains": []}):
        asyncio.run(rollout.prepare(case.input, case.workspace))
    prepare.assert_not_awaited()


def test_policy_setup_failure_prevents_successful_preparation(case, monkeypatch):
    monkeypatch.setattr(rollout, "_agent_identity", Mock())
    monkeypatch.setattr(rollout, "_prepare_agent_workspace", Mock())
    monkeypatch.setattr(
        rollout, "prepare_network_policy", AsyncMock(side_effect=RolloutEnvironmentError("nft"))
    )
    with network_policy_context({"deny_domains": ["example.org"]}):
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(rollout.prepare(case.input, case.workspace))


def test_agent_uses_policy_environment_and_restricted_command(case, monkeypatch):
    identity = SimpleNamespace(pw_uid=1000, pw_gid=1000, pw_dir="/home/user")
    monkeypatch.setattr(rollout.pwd, "getpwnam", Mock(return_value=identity))
    environment = Mock(side_effect=lambda env, _: {**env, "HTTPS_PROXY": "http://127.0.0.1:8123"})
    command = Mock(side_effect=lambda argv, _: ["restricted-command", *argv[5:]])
    monkeypatch.setattr(rollout, "agent_network_environment", environment)
    monkeypatch.setattr(rollout, "agent_network_command", command)
    process = SimpleNamespace(pid=123, returncode=0, wait=AsyncMock(return_value=0))
    launch = AsyncMock(return_value=process)
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(rollout.os, "killpg", Mock())
    relay = {"OPENAI_API_KEY": "relay-token", "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1"}
    with network_policy_context({"deny_domains": ["example.org"]}, task_id="task-001"):
        result = asyncio.run(rollout.run_agent(case.input, case.workspace, relay))
    assert result.exit_code == 0
    assert command.call_args.args[0][:5] == [
        "runuser",
        "--user",
        "user",
        "--preserve-environment",
        "--",
    ]
    assert command.call_args.args[1] == environment.call_args.args[1] == "user"
    assert launch.call_args.args[:2] == ("restricted-command", "/bin/bash")
    assert launch.call_args.kwargs["env"] == {
        **relay,
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/home/user",
        "USER": "user",
        "LOGNAME": "user",
        "HTTPS_PROXY": "http://127.0.0.1:8123",
    }


def test_policy_readiness_failure_prevents_agent_launch(case, monkeypatch):
    monkeypatch.setattr(
        rollout, "_agent_identity", Mock(return_value=SimpleNamespace(pw_dir="/home/user"))
    )
    monkeypatch.setattr(
        rollout, "agent_network_environment", Mock(side_effect=RolloutEnvironmentError("not ready"))
    )
    launch = AsyncMock()
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", launch)
    relay = {"OPENAI_API_KEY": "relay-token", "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1"}
    with network_policy_context({"deny_domains": ["example.org"]}):
        with pytest.raises(RolloutEnvironmentError, match="not ready"):
            asyncio.run(rollout.run_agent(case.input, case.workspace, relay))
    launch.assert_not_awaited()


@pytest.mark.parametrize("stop", [None, asyncio.CancelledError])
def test_run_agent_relay_cli_and_process_group_cleanup(case, monkeypatch, stop):
    wait = AsyncMock(side_effect=[stop(), 130, 130] if stop else [0, 0])
    process = SimpleNamespace(pid=123, returncode=None if stop else 0, wait=wait)
    launch, kill = AsyncMock(return_value=process), Mock()
    monkeypatch.setattr(rollout.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(rollout.os, "killpg", kill)
    monkeypatch.setenv("E2B_API_KEY", "host-secret")
    relay = {"OPENAI_API_KEY": "relay-token", "OPENAI_BASE_URL": "http://relay/v1"}
    case.input.task.update(max_steps=7, request_timeout_sec=12)
    if stop:
        with pytest.raises(stop):
            asyncio.run(rollout.run_agent(case.input, case.workspace, relay))
        marker = json.loads((case.workspace / "agent-stop.json").read_text())
        assert marker == ["attempt-1", "task-001", case.task["task_sha256"]]
        kill.assert_any_call(process.pid, signal.SIGTERM)
    else:
        assert asyncio.run(rollout.run_agent(case.input, case.workspace, relay)).exit_code == 0
        assert not (case.workspace / "agent-stop.json").exists()
    kill.assert_any_call(process.pid, signal.SIGKILL)
    argv, options = launch.call_args
    assert argv[:2] == ("/bin/bash", str(rollout._BUNDLE_ROOT / "run_rollout.sh"))
    args = dict(zip(argv[2::2], argv[3::2]))
    assert args.pop("--output-dir") == str(case.workspace / "automationbench-output")
    assert args["--task-id"] == "task-001"
    assert args["--model"] == "northrl" and args["--max-steps"] == "7"
    assert "--request-timeout" not in args
    assert options["cwd"] == case.workspace and options["start_new_session"] is True
    assert options["env"] == dict(
        relay, PATH="/usr/local/bin:/usr/bin:/bin", PYTHONDONTWRITEBYTECODE="1"
    )


@pytest.mark.parametrize("phase", ["term", "reap"])
@pytest.mark.parametrize("cancellations", [1, 2])
def test_cancel_cleanup_survives_repeated_cancellation(case, monkeypatch, phase, cancellations):
    async def scenario():
        running = asyncio.Event()
        entered, release, reaped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(pid=123, returncode=None)
        waits = 0

        async def wait():
            nonlocal waits
            waits += 1
            call = waits
            if call == 1:
                running.set()
                await asyncio.Event().wait()
            if call == 2 and phase == "reap":
                raise TimeoutError("SIGTERM grace expired")
            if call == (2 if phase == "term" else 3):
                entered.set()
            await release.wait()
            process.returncode = -9
            if call == 3:
                reaped.set()
            return process.returncode

        process.wait = wait
        monkeypatch.setattr(
            rollout.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
        )
        kill = Mock()
        monkeypatch.setattr(rollout.os, "killpg", kill)
        relay = {"OPENAI_API_KEY": "relay-token", "OPENAI_BASE_URL": "http://relay/v1"}
        task = asyncio.create_task(rollout.run_agent(case.input, case.workspace, relay))
        marker = case.workspace / "agent-stop.json"
        try:
            await asyncio.wait_for(running.wait(), timeout=1)
            task.cancel()  # The runtime owns the agent deadline and context-limit stop.
            await asyncio.wait_for(entered.wait(), timeout=1)
            kill.assert_any_call(process.pid, signal.SIGTERM)
            for _ in range(cancellations):
                task.cancel()  # The runtime cancels the harness when its relay hits context limit.
                await asyncio.sleep(0.01)
                assert not task.done()
                assert not marker.exists()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert reaped.is_set()
            kill.assert_any_call(process.pid, signal.SIGKILL)
            assert json.loads(marker.read_text()) == ["attempt-1", "task-001", "a" * 64]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_legacy_agent_timeout_does_not_stop_process(case, monkeypatch):
    async def scenario():
        running, release = asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(pid=123, returncode=None)

        async def wait():
            running.set()
            await release.wait()
            process.returncode = 0
            return 0

        process.wait = wait
        monkeypatch.setattr(
            rollout.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
        )
        kill = Mock()
        monkeypatch.setattr(rollout.os, "killpg", kill)
        case.input.task["agent_timeout_sec"] = 0.001
        relay = {"OPENAI_API_KEY": "relay-token", "OPENAI_BASE_URL": "http://relay/v1"}
        task = asyncio.create_task(rollout.run_agent(case.input, case.workspace, relay))
        try:
            await asyncio.wait_for(running.wait(), timeout=1)
            await asyncio.sleep(0.02)
            assert not task.done()
            kill.assert_not_called()
            release.set()
            assert (await task).exit_code == 0
            assert not (case.workspace / "agent-stop.json").exists()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["partial", "cancelled", "model_error", "world_pair", "identity"]
)
def test_evaluate_uploaded_rubric_snapshot_identity_and_stop(case, change):
    reward = 0.5 if change in {"partial", "cancelled"} else None
    output, assets = case.workspace / "automationbench-output", case.workspace / "evaluation"
    output.mkdir()
    assets.mkdir()
    record = dict(case.task, status="completed", finished_at="2026-01-01T00:00:00Z")
    evaluation = dict(case.task, assertions=[{"id": "row"}, {"id": "ads"}])
    world, runtime = {"rows": ["updated"]}, {"world_sha256": "paired", "offline_jobs": ["job"]}
    if change == "identity":
        evaluation["task_sha256"] = "b" * 64
    if change in {"model_error", "cancelled"}:
        record.update(status="running" if change == "cancelled" else change, finished_at=None)
    if change == "cancelled":
        marker = ["attempt-1", "task-001", "a" * 64]
        (case.workspace / "agent-stop.json").write_text(json.dumps(marker))
    if change == "world_pair":
        case.runner.restore_world.side_effect = ValueError("runtime belongs to another snapshot")
    for path, data in (
        (output / "run.json", record),
        (output / "final_world.json", world),
        (output / "runtime_state.json", runtime),
        (assets / "evaluation.json", evaluation),
    ):
        path.write_text(json.dumps(data))
    case.rubric.partial_credit.return_value = reward
    case.rubric.task_completed_correctly.return_value = float(reward == 1.0)
    if reward is None:
        with pytest.raises(RolloutEnvironmentError):
            asyncio.run(rollout.evaluate(case.input, case.workspace, AgentResult(), assets))
        case.rubric.partial_credit.assert_not_called()
    else:
        result = asyncio.run(rollout.evaluate(case.input, case.workspace, AgentResult(), assets))
        assert result.reward == reward
        assert result.status == ("success" if reward > 0 else "task_failed")
        assert result.metrics == {"automationbench_task_completed_correctly": float(reward == 1)}
        case.runner.restore_world.assert_called_once_with(world, runtime)
        case.runner.strip_none_values.assert_called_once_with(case.task["initial_state"])
        state = case.rubric.partial_credit.call_args.args[0]
        assert state["world"] is case.runner.restore_world.return_value
        assert state["initial_state"] == {"keep": 1}
        assert state["info"]["assertions"] == evaluation["assertions"]
