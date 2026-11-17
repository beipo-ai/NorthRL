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

"""Keep the ARVO submission service alive across independent runtime phases."""

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path


def test_verifier_survives_parent_exit_and_evaluates_saved_submission(tmp_path):
    helper = tmp_path / "phase.py"
    helper.write_text(
        textwrap.dedent(
            """
            import asyncio, json, os, subprocess, sys, time
            from http.server import BaseHTTPRequestHandler, HTTPServer
            from pathlib import Path
            from types import SimpleNamespace
            from northrl.remote_rollout.contracts import AgentResult, RolloutInput
            from northrl.rollout_entrypoints.mimo import arvo

            root = Path.cwd()
            arvo._ROOT, arvo._STATE_ROOT, arvo._ROOT_UID = root, root / "state", os.getuid()
            arvo._PORT = int((root / "port").read_text())
            task = RolloutInput(rollout_id="lifetime", task={
                "description": "AddressSanitizer: heap-buffer-overflow in function `target`"
            })
            if sys.argv[1] == "--serve-verifier":
                # Only health checking is needed; evaluator reads the saved submission.
                with HTTPServer(("127.0.0.1", arvo._PORT), BaseHTTPRequestHandler) as server:
                    server.timeout = 0.1
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        server.handle_request()
            elif sys.argv[1] == "evaluate":
                result = asyncio.run(arvo.evaluate(task, root, AgentResult(), root))
                print(result.model_dump_json())
            else:
                arvo.__file__ = __file__
                arvo._agent_identity = lambda _: SimpleNamespace(pw_name="agent", pw_uid=1234)
                arvo._verifier_identity = lambda: SimpleNamespace(pw_uid=1235)
                arvo._agent_binary = lambda _: None
                environment = {"PATH": os.defpath, "PYTHONPATH": os.environ["PYTHONPATH"]}
                arvo._runtime_environment = lambda _, user: environment
                async def prepare_network(user):
                    assert user == "agent"
                    state = arvo._read_state(task)
                    os.kill(state["verifier_pid"], 0)
                    await arvo._wait_for_verifier()
                    (root / "network-prepared").touch()
                arvo.prepare_network_policy = prepare_network
                popen, async_spawn = subprocess.Popen, asyncio.create_subprocess_exec
                transports = []

                def launch(argv, **kwargs):
                    process = popen(argv, **kwargs)
                    (root / "launch.json").write_text(json.dumps({
                        "pid": process.pid,
                        "argv": [str(arg) for arg in argv],
                        "detached": kwargs["start_new_session"],
                        "cwd": str(kwargs["cwd"]),
                        "environment": kwargs["env"],
                        "stdin_closed": kwargs["stdin"] == subprocess.DEVNULL,
                        "shared_log": kwargs["stdout"] is kwargs["stderr"],
                        "log": kwargs["stdout"].name,
                    }))
                    return process

                async def track_transport(*args, **kwargs):
                    process = await async_spawn(*args, **kwargs)
                    transports.append(process._transport)
                    return process

                subprocess.Popen = launch
                asyncio.create_subprocess_exec = track_transport

                async def prepare_phase():
                    await arvo.prepare(task, Path("/home/agent"))
                    # Reproduce sandbox interpreter teardown even on Python builds
                    # that otherwise retain the asyncio transport until process exit.
                    for transport in transports:
                        transport.close()
                    await asyncio.sleep(0.02)

                asyncio.run(prepare_phase())
            """
        ),
        encoding="utf-8",
    )
    tmp_path.chmod(0o700)
    (tmp_path / "binary").mkdir()
    (tmp_path / "binary/run.sh").touch()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        (tmp_path / "port").write_text(str(listener.getsockname()[1]))
    environment = {
        "PATH": os.defpath,
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch_file = tmp_path / "launch.json"
    try:
        prepared = subprocess.run(
            [sys.executable, "-B", str(helper), "prepare"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert prepared.returncode == 0, prepared.stderr
        launch = json.loads(launch_file.read_text())
        state = json.loads((tmp_path / "state/lifetime/state.json").read_text())
        assert (tmp_path / "network-prepared").exists()
        assert state["verifier_pid"] == launch["pid"]
        assert launch["argv"] == [sys.executable, "-u", str(helper), "--serve-verifier"]
        assert launch["detached"] and launch["stdin_closed"] and launch["shared_log"]
        assert launch["cwd"] == str(tmp_path)
        assert launch["environment"] == {
            "PATH": os.defpath,
            "PYTHONPATH": environment["PYTHONPATH"],
        }
        assert launch["log"] == str(tmp_path / "arvo-server.log")
        assert os.getsid(launch["pid"]) == launch["pid"]
        (tmp_path / "last_result.json").write_text(
            json.dumps({"crash": True, "match": True, "timeout": False, "submit_count": 1})
        )
        evaluated = subprocess.run(
            [sys.executable, "-B", str(helper), "evaluate"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert evaluated.returncode == 0, evaluated.stderr
        result = json.loads(evaluated.stdout)
        assert result["reward"] == 1.0 and result["status"] == "success"
        assert result["metrics"]["mimo_arvo_submit_count"] == 1.0
    finally:
        if launch_file.exists():
            pid = json.loads(launch_file.read_text())["pid"]
            command = Path(f"/proc/{pid}/cmdline")
            if command.exists() and str(helper).encode() in command.read_bytes():
                os.kill(pid, signal.SIGKILL)
