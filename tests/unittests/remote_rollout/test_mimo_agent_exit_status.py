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

"""MiMo verifier inputs preserve the CLI outcome without requiring a success event."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from northrl.remote_rollout import network_policy
from northrl.remote_rollout.contracts import AgentResult, RolloutEnvironmentError, RolloutInput
from northrl.rollout_entrypoints.mimo import arvo, code, general, terminal


@pytest.fixture(params=[code, arvo, terminal, general], ids=["code", "arvo", "terminal", "general"])
def mimo_agent(request, tmp_path, monkeypatch):
    adapter = request.param
    monkeypatch.setattr(network_policy, "_ROOT", tmp_path / "network-policy")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    identity = SimpleNamespace(pw_name="agent", pw_uid=1234, pw_dir="/home/agent")
    monkeypatch.setattr(adapter, "_agent_identity", lambda _: identity)
    monkeypatch.setattr(adapter, "_require_private_path", lambda _: None)
    monkeypatch.setattr(adapter, "resolve_workspace", lambda _: workspace)
    monkeypatch.setattr(adapter.shutil, "which", lambda name, **_: f"/usr/bin/{name}")
    if hasattr(adapter, "_agent_binary"):
        monkeypatch.setattr(adapter, "_agent_binary", lambda _: binary)
    if hasattr(adapter, "_runtime_environment"):
        monkeypatch.setattr(adapter, "_runtime_environment", lambda _: {"PATH": "/usr/bin"})
    if hasattr(adapter, "_IMAGE_PATH"):
        monkeypatch.setattr(adapter, "_IMAGE_PATH", tmp_path / "image-path")
    if hasattr(adapter, "_RUNTIME_METADATA"):
        monkeypatch.setattr(adapter, "_RUNTIME_METADATA", tmp_path / "metadata")
    if adapter is general:
        monkeypatch.setattr(adapter, "_servers", lambda: [])

    resolve = Path.resolve

    def resolve_binary(path, *args, **kwargs):
        if path == Path("/opt/mimo-agents/claude"):
            return binary
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_binary)
    run_command = adapter._run_command
    command = AsyncMock()
    cleanup = AsyncMock()
    monkeypatch.setattr(adapter, "_run_command", command)
    monkeypatch.setattr(adapter, "_stop_agent_processes", cleanup)
    return SimpleNamespace(
        adapter=adapter,
        workspace=workspace,
        command=command,
        run_command=run_command,
        cleanup=cleanup,
        input=RolloutInput(rollout_id="mimo-exit", task={"instruction": "repair the task"}),
        environment={
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
            "ANTHROPIC_API_KEY": "test",
        },
    )


@pytest.mark.parametrize(
    ("exit_code", "output"),
    [
        (0, ""),
        (0, "non-JSON output\n"),
        (0, '{"type":"result","subtype":"error_max_turns","is_error":true}\n'),
        (0, '{"type":"result","subtype":"success","is_error":true}\n'),
        (1, '{"type":"result","subtype":"success","is_error":false}\n'),
        (124, ""),
        (-9, ""),
    ],
    ids=["no-final", "malformed-output", "max-turns", "error-flag", "cli-error", "124", "signal"],
)
def test_mimo_preserves_cli_exit_without_a_success_marker(mimo_agent, exit_code, output):
    mimo_agent.command.return_value = subprocess.CompletedProcess(["claude"], exit_code, output, "")

    result = asyncio.run(
        mimo_agent.adapter.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
    )

    assert result.exit_code == exit_code
    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)
    arguments, options = mimo_agent.command.await_args
    assert arguments[0][0] == "runuser"
    assert "HTTPS_PROXY" not in options["env"]


def test_mimo_timeout_preserves_timeout_code_after_cleanup(mimo_agent):
    mimo_agent.command.side_effect = mimo_agent.adapter.CommandTimeoutError("agent timed out")

    result = asyncio.run(
        mimo_agent.adapter.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
    )

    assert result.exit_code == 124
    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)


@pytest.fixture
def prepared_network_policy(monkeypatch):
    def load(user):
        assert user == "agent"
        return {"uid": 1234, "gid": 1234, "port": 18080}

    monkeypatch.setattr(network_policy, "_load_policy", load)
    monkeypatch.setattr(network_policy, "_binary", lambda name: "/usr/bin/" + name)


def test_mimo_network_policy_wraps_agent_launch(mimo_agent, prepared_network_policy):
    mimo_agent.command.return_value = subprocess.CompletedProcess(["claude"], 0, "", "")

    asyncio.run(
        mimo_agent.adapter.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
    )

    arguments, options = mimo_agent.command.await_args
    assert arguments[0][:3] == ["/usr/bin/setpriv", "--reuid", "1234"]
    assert "--no-new-privs" in arguments[0]
    assert "--clear-groups" in arguments[0]
    assert "runuser" not in arguments[0]
    assert options["env"]["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert options["env"]["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert options["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"


def test_mimo_configured_but_unprepared_policy_prevents_launch(mimo_agent):
    with network_policy.network_policy_context({"deny_domains": ["github.com"]}):
        with pytest.raises(RolloutEnvironmentError, match="network policy is not ready"):
            asyncio.run(
                mimo_agent.adapter.run_agent(
                    mimo_agent.input, mimo_agent.workspace, mimo_agent.environment
                )
            )

    mimo_agent.command.assert_not_awaited()


def test_code_verifier_keeps_agent_network_policy(tmp_path, monkeypatch, prepared_network_policy):
    workspace, private, assets = (tmp_path / name for name in ("workspace", "private", "assets"))
    for directory in (workspace, private, assets):
        directory.mkdir()
    (private / "baseline.git").mkdir()
    (private / "state.json").write_text(json.dumps({"rollout_id": "code-policy", "base": "a" * 40}))
    (assets / "evaluation.json").write_text(
        json.dumps({"test_command": "pytest", "verifier_timeout_sec": 10})
    )
    (assets / "test.patch").write_text("")
    monkeypatch.setattr(code, "_private_directory", lambda _: private)
    monkeypatch.setattr(code, "_require_private_path", lambda _: None)
    monkeypatch.setattr(code, "_evaluation_file", lambda directory, name: directory / name)
    monkeypatch.setattr(code, "_agent_identity", lambda _: SimpleNamespace(pw_name="agent"))
    monkeypatch.setattr(
        code, "_runtime_environment", lambda _, user=None: {"USER": user or "agent"}
    )
    command = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(code, "_run_command", command)

    result = asyncio.run(
        code.evaluate(
            RolloutInput(rollout_id="code-policy", task={}), workspace, AgentResult(), assets
        )
    )

    assert result.reward == 1.0
    *trusted_calls, verifier = command.await_args_list
    assert all("HTTPS_PROXY" not in call.kwargs["env"] for call in trusted_calls)
    assert verifier.args[0][:3] == ["/usr/bin/setpriv", "--reuid", "1234"]
    assert verifier.kwargs["env"]["HTTPS_PROXY"] == "http://127.0.0.1:18080"


@pytest.mark.parametrize("mimo_agent", [general], indirect=True, ids=["general"])
@pytest.mark.parametrize("final_event", [False, True])
@pytest.mark.parametrize("initial_event", [False, True])
def test_general_keeps_available_answer_after_unsuccessful_cli(
    mimo_agent, monkeypatch, final_event, initial_event
):
    events = []
    if initial_event:
        events.append(
            {
                "type": "system",
                "subtype": "init",
                "mcp_servers": [{"name": "tools", "status": "connected"}],
            }
        )
    events.append(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "partial answer"}]}}
    )
    if final_event:
        events.append(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "result": "partial answer",
            }
        )
    monkeypatch.setattr(general, "_servers", lambda: [{"name": "tools"}])
    mimo_agent.command.return_value = subprocess.CompletedProcess(
        ["claude"], 1, "\n".join(json.dumps(event) for event in events), ""
    )

    result = asyncio.run(
        general.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
    )

    assert result.exit_code == 1
    assert result.final_answer == "partial answer"
    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)


@pytest.mark.parametrize("mimo_agent", [general], indirect=True, ids=["general"])
def test_general_still_rejects_disconnected_mcp_server(mimo_agent, monkeypatch):
    monkeypatch.setattr(general, "_servers", lambda: [{"name": "tools"}])
    mimo_agent.command.return_value = subprocess.CompletedProcess(
        ["claude"],
        1,
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "mcp_servers": [{"name": "tools", "status": "failed"}],
            }
        ),
        "",
    )

    with pytest.raises(RolloutEnvironmentError, match="MCP servers did not connect"):
        asyncio.run(
            general.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
        )

    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)


@pytest.fixture
def real_mimo_command(mimo_agent, monkeypatch):
    """Replace the sandbox CLI with a local process, preserving command execution."""

    def install(script):
        async def run_command(_argv, **kwargs):
            return await mimo_agent.run_command([sys.executable, "-c", script], **kwargs)

        monkeypatch.setattr(mimo_agent.adapter, "_run_command", run_command)

    return install


@pytest.mark.parametrize("legacy_timeout", [0.001, None], ids=["short-budget", "invalid-budget"])
def test_mimo_ignores_agent_timeout_in_task_data(mimo_agent, real_mimo_command, legacy_timeout):
    # This command exceeds the old sample budget and must still complete normally.
    real_mimo_command("import time; time.sleep(0.03)")
    mimo_agent.input.task["agent_timeout_sec"] = legacy_timeout

    result = asyncio.run(
        mimo_agent.adapter.run_agent(mimo_agent.input, mimo_agent.workspace, mimo_agent.environment)
    )

    assert result.exit_code == 0
    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)


def _process_is_running(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return False
    state = next(line for line in status.splitlines() if line.startswith("State:"))
    # An orphan can remain a zombie briefly until the host's init reaps it.
    return state.split()[1] != "Z"


def test_mimo_framework_cancellation_kills_agent_and_child(mimo_agent, real_mimo_command, tmp_path):
    pids_path = tmp_path / "agent-pids.json"
    real_mimo_command(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pids_path)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
        "time.sleep(30)\n"
    )

    async def cancel_after_start():
        agent = asyncio.create_task(
            mimo_agent.adapter.run_agent(
                mimo_agent.input, mimo_agent.workspace, mimo_agent.environment
            )
        )
        pids = []
        try:

            async def wait_for_pids():
                while True:
                    try:
                        return json.loads(pids_path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        await asyncio.sleep(0.01)

            pids = await asyncio.wait_for(wait_for_pids(), timeout=5)
            assert all(_process_is_running(pid) for pid in pids)
            agent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(agent, timeout=5)

            async def wait_for_processes_to_stop():
                while any(_process_is_running(pid) for pid in pids):
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_processes_to_stop(), timeout=5)
        finally:
            # Keep the test bounded and clean if a process-cleanup regression occurs.
            if pids:
                try:
                    os.killpg(pids[0], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if not agent.done():
                agent.cancel()
            await asyncio.gather(agent, return_exceptions=True)

    asyncio.run(cancel_after_start())
    mimo_agent.cleanup.assert_awaited_once_with(mimo_agent.input)


def test_mimo_command_still_enforces_explicit_verifier_timeout(mimo_agent):
    async def run_command():
        with pytest.raises(mimo_agent.adapter.CommandTimeoutError):
            await mimo_agent.run_command(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.03
            )

    asyncio.run(run_command())
