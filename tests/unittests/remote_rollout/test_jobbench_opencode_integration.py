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

"""Opt-in native CLI smoke test against a local, fake OpenAI endpoint.

Set NORTHRL_TEST_OPENCODE_BIN to an independently verified OpenCode V2 2.2.0 binary.
The test supplies no real model or search credentials and installs no software.
"""

import json
import os
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from northrl.rollout_entrypoints.jobbench import opencode


@pytest.mark.integration
def test_native_cli_uses_local_relay_and_emits_parseable_final_answer(tmp_path):
    configured = os.environ.get("NORTHRL_TEST_OPENCODE_BIN")
    if not configured:
        pytest.skip("Set NORTHRL_TEST_OPENCODE_BIN to a verified OpenCode V2 2.2.0 binary")
    binary = Path(configured).resolve(strict=True)
    assert binary.is_file() and os.access(binary, os.X_OK)
    calls = []
    answer = "JOBBENCH_OPENCODE_SMOKE_OK"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # pylint: disable=invalid-name
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            calls.append((self.path, body, self.headers.get("Authorization")))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            common = {"id": "chatcmpl-smoke", "created": 1, "model": "northrl"}
            if body.get("stream"):
                chunks = [
                    {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": answer},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                ]
                payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                data = (payload + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                data = json.dumps(
                    {
                        **common,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "opencode.json").write_text(json.dumps(opencode.configuration(3)))
    global_config = home / ".config/opencode"
    global_config.mkdir(parents=True)
    (global_config / "opencode.json").write_text(json.dumps({"update": "disable"}))
    # Use an allowlisted environment so host credentials cannot reach the CLI.
    environment = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "XDG_STATE_HOME": str(home / ".local/state"),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "OPENAI_API_KEY": "jobbench-local-smoke-key",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    # The adapter's first five arguments select runuser; this test uses its own UID.
    argv = [*opencode.command(binary, "unused")[5:], "Reply with a short confirmation."]
    process = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"OpenCode exceeded 45 seconds: {stdout[-2000:]}\n{stderr[-2000:]}")
        assert process.returncode == 0, f"{stdout[-4000:]}\n{stderr[-4000:]}"
        assert not opencode.has_error(stdout), stdout[-4000:]
        assert opencode.final_answer(stdout) == answer, stdout[-4000:]
        assert calls
        assert all(path == "/v1/chat/completions" for path, _, _ in calls)
        assert all(body.get("model") == "northrl" for _, body, _ in calls)
        assert all(auth == "Bearer jobbench-local-smoke-key" for _, _, auth in calls)
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
