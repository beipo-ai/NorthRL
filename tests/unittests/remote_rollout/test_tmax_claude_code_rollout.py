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

from __future__ import annotations

import asyncio
import json
import os
import pwd
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from northrl.remote_rollout import network_policy
from northrl.remote_rollout.contracts import AgentResult, RolloutEnvironmentError, RolloutInput
from northrl.rollout_entrypoints import tmax_claude_code_rollout as tmax

_MODEL_ENV = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8000", "ANTHROPIC_API_KEY": "northrl"}
_RESULT_BLOCK = "SWEBench results starts here\n{verdict}\nSWEBench results ends here\n"


def _input(**task) -> RolloutInput:
    return RolloutInput(
        rollout_id="rollout-1",
        task={"instruction": "Fix the service.", **task},
        metadata={"task_id": "tmax__task_000001_aa__bb"},
    )


def _identity(home: Path) -> pwd.struct_passwd:
    return pwd.struct_passwd(("user", "x", 1000, 1000, "", str(home), "/bin/bash"))


@pytest.fixture
def private_state(tmp_path, monkeypatch):
    """Treat files owned by the test user as root-owned and keep state under tmp_path."""
    monkeypatch.setattr(tmax, "_ROOT_UID", os.getuid())
    monkeypatch.setattr(tmax, "_STATE_ROOT", tmp_path / "state")
    return tmp_path


def test_agent_environment_requests_forced_thinking_and_drops_host_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("NORTHGATE_API_KEY", "secret")
    environment = tmax._agent_environment(_input(), _identity(tmp_path), _MODEL_ENV)

    assert json.loads(environment["CLAUDE_CODE_EXTRA_BODY"]) == {
        "output_config": {"effort": "high"}
    }
    assert environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "128000"
    assert environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16384"
    assert environment["ANTHROPIC_BASE_URL"] == _MODEL_ENV["ANTHROPIC_BASE_URL"]
    assert environment["HOME"] == str(tmp_path) and environment["USER"] == "user"
    assert "NORTHGATE_API_KEY" not in environment
    assert environment["PATH"].startswith(str(tmax._CLAUDE_ROOT) + ":")


def test_agent_environment_options(tmp_path):
    environment = tmax._agent_environment(
        _input(effort="", compact_window=96000), _identity(tmp_path), _MODEL_ENV
    )
    assert "CLAUDE_CODE_EXTRA_BODY" not in environment
    assert environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "96000"
    with pytest.raises(RolloutEnvironmentError, match="model gateway"):
        tmax._agent_environment(_input(), _identity(tmp_path), {})
    with pytest.raises(RolloutEnvironmentError, match="compact_window"):
        tmax._agent_environment(_input(compact_window=0), _identity(tmp_path), _MODEL_ENV)


def test_final_result_event_takes_the_last_result():
    stdout = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            "not json",
            json.dumps({"type": "result", "subtype": "error", "num_turns": 1}),
            json.dumps({"type": "assistant"}),
            json.dumps({"type": "result", "subtype": "success", "num_turns": 7}),
        ]
    )
    assert tmax._final_result_event(stdout)["num_turns"] == 7
    assert tmax._final_result_event("") == {}


def test_run_agent_runs_claude_as_template_user(monkeypatch, private_state):
    calls = []

    async def fake_run(argv, *, timeout, cwd=None, env=None, stdin=None):
        calls.append({"argv": [str(a) for a in argv], "timeout": timeout, "cwd": cwd, "env": env})
        stdout = json.dumps({"type": "result", "subtype": "success", "num_turns": 12})
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(tmax, "_agent_identity", lambda: _identity(private_state))
    monkeypatch.setattr(tmax, "_run_command", fake_run)
    result = asyncio.run(tmax.run_agent(_input(), tmax._WORKSPACE, _MODEL_ENV))

    assert result == AgentResult(exit_code=0)
    argv = calls[0]["argv"]
    assert argv[:5] == ["runuser", "--user", "user", "--preserve-environment", "--"]
    assert argv[argv.index("--model") + 1] == "northrl"
    assert argv[-2:] == ["--", "Fix the service."]
    # The platform agent budget, not the entrypoint, limits Claude Code's run time.
    assert calls[0]["timeout"] is None and calls[0]["cwd"] == tmax._WORKSPACE
    summary = json.loads((private_state / "state" / "rollout-1" / "agent.json").read_text())
    assert summary["num_turns"] == 12
    assert tmax._agent_metrics(_input()) == {"claude_num_turns": 12.0}


@pytest.mark.parametrize("fails", [False, True])
def test_prepare_network_policy_follows_install_and_integrity(monkeypatch, private_state, fails):
    identity = _identity(private_state)
    monkeypatch.setattr(tmax, "_WORKSPACE", private_state)
    monkeypatch.setattr(tmax, "_agent_identity", lambda: identity)
    monkeypatch.setattr(tmax, "_remove_baked_grader", Mock())
    monkeypatch.setattr(tmax, "_prepare_agent_dirs", Mock())
    install_claude, record_integrity = AsyncMock(), AsyncMock()
    monkeypatch.setattr(tmax, "_install_claude", install_claude)
    monkeypatch.setattr(tmax, "_record_integrity", record_integrity)

    def install(user, domains, task_id):
        assert user == "user" and domains == ("github.com",)
        assert task_id == _input().metadata["task_id"]
        install_claude.assert_awaited_once()
        record_integrity.assert_awaited_once()
        assert (private_state / "state/rollout-1").is_dir()
        if fails:
            raise PermissionError("firewall unavailable")

    installation = Mock(side_effect=install)
    monkeypatch.setattr(network_policy, "_prepare", installation)
    with network_policy.network_policy_context(
        {"deny_domains": ["github.com"]}, task_id=_input().metadata["task_id"]
    ):
        if fails:
            with pytest.raises(RolloutEnvironmentError, match="firewall unavailable"):
                asyncio.run(tmax.prepare(_input(), private_state))
        else:
            asyncio.run(tmax.prepare(_input(), private_state))
            network_policy.assert_network_policy_prepared()
    installation.assert_called_once()


def test_network_launch_disables_user_sudo_and_preserves_gateway(monkeypatch, private_state):
    monkeypatch.setattr(tmax, "_agent_identity", lambda: _identity(private_state))
    run = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(tmax, "_run_command", run)

    def installed(user):
        assert user == "user"
        return {"uid": 1000, "gid": 1000, "port": 18080}

    monkeypatch.setattr(network_policy, "_load_policy", installed)
    monkeypatch.setattr(network_policy, "_binary", lambda name: "/usr/bin/" + name)
    asyncio.run(tmax.run_agent(_input(), private_state, _MODEL_ENV))
    command = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    assert command[:5] == ["/usr/bin/setpriv", "--reuid", "1000", "--regid", "1000"]
    assert "--no-new-privs" in command and "--clear-groups" in command
    assert "--bounding-set=-all" in command
    assert command[-2:] == ["--", "Fix the service."]
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert environment["ANTHROPIC_BASE_URL"] == _MODEL_ENV["ANTHROPIC_BASE_URL"]


def test_network_readiness_failure_prevents_claude_launch(monkeypatch, private_state):
    monkeypatch.setattr(tmax, "_agent_identity", lambda: _identity(private_state))
    run = AsyncMock()
    monkeypatch.setattr(tmax, "_run_command", run)
    monkeypatch.setattr(
        network_policy,
        "_load_policy",
        Mock(side_effect=RolloutEnvironmentError("network policy is not ready")),
    )
    with pytest.raises(RolloutEnvironmentError, match="not ready"):
        asyncio.run(tmax.run_agent(_input(), private_state, _MODEL_ENV))
    run.assert_not_awaited()


def test_cancelled_command_kills_its_process_group(tmp_path):
    pid_file = tmp_path / "pid"

    async def scenario() -> int:
        script = f"echo $$ > {pid_file}; sleep 30 & wait"
        task = asyncio.create_task(tmax._run_command(["/bin/sh", "-c", script], timeout=None))
        while not pid_file.exists() or not pid_file.read_text().strip():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(pid_file.read_text())

    group = asyncio.run(scenario())
    # The shell and its background sleep share the group; both must be gone.
    for _ in range(200):
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    pytest.fail("process group survived cancellation")


def _evaluation_dir(root: Path) -> Path:
    assets = root / "assets"
    assets.mkdir(mode=0o700)
    (assets / "run-tests.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (assets / "run-tests.sh").chmod(0o700)
    return assets


@pytest.mark.parametrize(
    ("output", "reward", "status", "mismatch"),
    [
        (_RESULT_BLOCK.format(verdict="PASSED"), 1.0, "success", 0.0),
        (_RESULT_BLOCK.format(verdict="FAILED"), 0.0, "task_failed", 0.0),
        (
            "[tmax-verifier] INTEGRITY MISMATCH: toolchain changed\n"
            + _RESULT_BLOCK.format(verdict="FAILED"),
            0.0,
            "task_failed",
            1.0,
        ),
    ],
)
def test_evaluate_reads_the_result_block(
    monkeypatch, private_state, output, reward, status, mismatch
):
    calls = []

    async def fake_run(argv, *, timeout, cwd=None, env=None, stdin=None):
        calls.append({"argv": [str(a) for a in argv], "timeout": timeout, "env": env})
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(tmax, "_run_command", fake_run)
    assets = _evaluation_dir(private_state)
    result = asyncio.run(tmax.evaluate(_input(), tmax._WORKSPACE, AgentResult(), assets))

    assert result.reward == reward and result.status == status
    assert result.metrics["tmax_passed"] == reward
    assert result.metrics["tmax_integrity_mismatch"] == mismatch
    assert calls[0]["argv"] == ["/bin/bash", str(assets / "run-tests.sh")]
    assert calls[0]["timeout"] == 1200.0
    assert set(calls[0]["env"]) >= {"PATH", "HOME"} and calls[0]["env"]["HOME"] == "/root"


@pytest.mark.parametrize("output", ["no markers here", _RESULT_BLOCK.format(verdict="PASSED") * 2])
def test_evaluate_rejects_missing_or_repeated_markers(monkeypatch, private_state, output):
    async def fake_run(argv, *, timeout, cwd=None, env=None, stdin=None):
        return subprocess.CompletedProcess(argv, 1, output, "")

    monkeypatch.setattr(tmax, "_run_command", fake_run)
    with pytest.raises(RolloutEnvironmentError, match="result markers"):
        asyncio.run(
            tmax.evaluate(_input(), tmax._WORKSPACE, AgentResult(), _evaluation_dir(private_state))
        )


def test_evaluate_timeout_is_an_environment_error(monkeypatch, private_state):
    async def fake_run(argv, *, timeout, cwd=None, env=None, stdin=None):
        raise tmax.CommandTimeoutError("slow grader")

    monkeypatch.setattr(tmax, "_run_command", fake_run)
    with pytest.raises(RolloutEnvironmentError, match="timeout"):
        asyncio.run(
            tmax.evaluate(_input(), tmax._WORKSPACE, AgentResult(), _evaluation_dir(private_state))
        )


def test_evaluate_requires_run_tests(private_state):
    assets = private_state / "empty"
    assets.mkdir(mode=0o700)
    with pytest.raises(RolloutEnvironmentError, match="run-tests.sh"):
        asyncio.run(tmax.evaluate(_input(), tmax._WORKSPACE, AgentResult(), assets))


def test_remove_baked_grader(tmp_path, monkeypatch):
    tests = tmp_path / "tests"
    (tests / "nested").mkdir(parents=True)
    (tests / "nested" / "test_final_state.py").write_text("assert True\n")
    verifier = tmp_path / "logs" / "verifier"
    verifier.mkdir(parents=True)
    monkeypatch.setattr(tmax, "_LEGACY_GRADER_PATHS", (tests, verifier, tmp_path / "absent"))

    tmax._remove_baked_grader()
    assert not tests.exists() and not verifier.exists()
