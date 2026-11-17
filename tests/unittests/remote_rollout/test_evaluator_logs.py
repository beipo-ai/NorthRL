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

"""Exercise real Bash/Python processes, including the unchanged SWE entrypoint."""

import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from northrl.remote_rollout import evaluator_logs
from northrl.rollout_entrypoints import swerebench_claude_code_rollout as swe


def _wrap(tmp_path, source):
    script = tmp_path / "evaluation assets" / "run-tests.sh"
    script.parent.mkdir(exist_ok=True)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"input_data": {"rollout_id": "rollout-1"}, "attempt_id": "attempt-1"})
    )
    logs = tmp_path / "logs"
    script.write_bytes(
        evaluator_logs.wrap_evaluator_script(
            source,
            interpreter=sys.executable,
            helper=str(Path(evaluator_logs.__file__).resolve()),
            input_path=str(input_path),
            log_dir=str(logs),
        )
    )
    script.chmod(0o700)
    return script, logs


@pytest.mark.parametrize(
    "source",
    [
        b'printf "stdout\\n"; printf "stderr\\n" >&2; exit 0\n',
        b'printf "failed\\n" >&2; exit 7\n',
        b'set -euo pipefail\ntrap \'printf "EXIT:%s\\n" "$?" >&2\' EXIT\nprintf before\nfalse\nprintf never\n',
        b'printf "%s\\n" "$0" "${BASH_SOURCE[0]}" "$PWD" "$1" "$2" "$HOME" "$SHLVL"\n',
        b'. "$(dirname "$0")/sibling.sh"\n',
        b"kill -TERM $$\n",
        b"kill -KILL $$\n",
    ],
)
def test_wrapper_preserves_original_output_exit_status_path_args_and_traps(tmp_path, source):
    script = tmp_path / "evaluation assets" / "run-tests.sh"
    script.parent.mkdir()
    (script.parent / "sibling.sh").write_text('printf "relative include\\n"\n')
    script.write_bytes(source)
    command = ["/bin/bash", str(script), "argument with spaces", "--flag"]
    baseline = subprocess.run(command, cwd=tmp_path, capture_output=True, check=False, timeout=10)
    _, logs = _wrap(tmp_path, source)
    actual = subprocess.run(command, cwd=tmp_path, capture_output=True, check=False, timeout=10)
    assert (actual.returncode, actual.stdout, actual.stderr) == (
        baseline.returncode,
        baseline.stdout,
        baseline.stderr,
    )
    assert (logs / "stdout.log").read_bytes() == baseline.stdout
    assert (logs / "stderr.log").read_bytes() == baseline.stderr
    assert script.read_bytes() == source
    assert script.stat().st_mode & 0o777 == 0o700
    status = json.loads((logs / "status.json").read_text())
    assert status["returncode"] == baseline.returncode
    assert status["finished"] is True
    assert status["rollout_id"] == "rollout-1"
    assert status["attempt_id"] == "attempt-1"
    assert status["bytes"] == {"stdout": len(baseline.stdout), "stderr": len(baseline.stderr)}
    assert status["capture_errors"] == {}
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in logs.iterdir())


def test_large_binary_output_is_not_truncated_or_deadlocked(tmp_path):
    source = (
        f"'{sys.executable}' - <<'PY'\n"
        "import os\n"
        "for i in range(128):\n"
        "    os.write(1, bytes(range(256)) * 256)\n"
        "    os.write(2, bytes(reversed(range(256))) * 256)\n"
        "PY\n"
    ).encode()
    script, logs = _wrap(tmp_path, source)
    actual = subprocess.run(
        ["/bin/bash", str(script)], capture_output=True, check=False, timeout=15
    )
    assert actual.returncode == 0, actual.stderr[-1000:]
    assert actual.stdout == bytes(range(256)) * 256 * 128
    assert actual.stderr == bytes(reversed(range(256))) * 256 * 128
    assert (logs / "stdout.log").read_bytes() == actual.stdout
    assert (logs / "stderr.log").read_bytes() == actual.stderr


@pytest.mark.parametrize("termination", [signal.SIGTERM, signal.SIGKILL])
def test_output_is_saved_before_exit_and_survives_interruption(tmp_path, termination):
    script, logs = _wrap(
        tmp_path, b"printf ready; printf error >&2; while true; do sleep 10; done\n"
    )
    process = subprocess.Popen(
        ["/bin/bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5
        while not (logs / "stderr.log").exists() or (logs / "stderr.log").read_bytes() != b"error":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (logs / "stdout.log").read_bytes() == b"ready"
        assert json.loads((logs / "status.json").read_text())["finished"] is False
        # SIGKILL intentionally leaves only the initial status, as a lost sandbox would.
        if termination == signal.SIGKILL:
            # Terminate the evaluator's process group too, to avoid leaking the test child.
            process.send_signal(signal.SIGSTOP)
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
            assert len(children) == 1
            os.killpg(int(children[0]), signal.SIGKILL)
        process.send_signal(termination)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == -termination
        assert stdout == b"ready" and stderr == b"error"
        status = json.loads((logs / "status.json").read_text())
        if termination == signal.SIGTERM:
            assert status["finished"] is True and status["returncode"] == -signal.SIGTERM
        assert (logs / "stdout.log").read_bytes() == stdout
        assert (logs / "stderr.log").read_bytes() == stderr
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)


def test_log_initialization_failure_still_runs_original_script(tmp_path):
    script, logs = _wrap(tmp_path, b"printf still-runs; exit 9\n")
    logs.write_text("not a directory")
    actual = subprocess.run(["/bin/bash", str(script)], capture_output=True, check=False, timeout=5)
    assert actual.returncode == 9
    assert actual.stdout == b"still-runs"
    assert b"capture unavailable" in actual.stderr


def test_capture_write_failure_preserves_forwarded_output_and_exit_code(
    tmp_path, monkeypatch, capfdbinary
):
    script, logs = _wrap(tmp_path, b"printf out; printf err >&2; exit 7\n")
    original_open = Path.open

    class FullDisk(io.BytesIO):
        def write(self, _data):
            raise OSError("disk full")

    def open_file(path, *args, **kwargs):
        if path == logs / "stdout.log":
            original_open(path, *args, **kwargs).close()
            return FullDisk()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    assert (
        evaluator_logs.capture_evaluator(
            script, [], input_path=tmp_path / "input.json", log_dir=logs
        )
        == 7
    )
    output = capfdbinary.readouterr()
    assert output.out == b"out"
    assert b"err" in output.err
    status = json.loads((logs / "status.json").read_text())
    assert status["capture_errors"]["stdout"] == "OSError: disk full"
    assert status["returncode"] == 7


@pytest.mark.parametrize("verdict", ["PASSED", "FAILED", "crash"])
def test_unchanged_swe_entrypoint_parses_reward_and_keeps_full_logs(tmp_path, monkeypatch, verdict):
    source = b'printf "diagnostic\\n" >&2\n'
    if verdict == "crash":
        source += b'head -c 10000 /dev/zero | tr "\\0" x >&2; exit 4\n'
    else:
        source += f'printf "SWEBench results starts here\\n{verdict}\\nSWEBench results ends here\\n"\n'.encode()
    script, logs = _wrap(tmp_path, source)
    monkeypatch.setattr(swe, "_EVALUATOR", str(script))
    monkeypatch.setattr(swe, "_trusted_runtime_path", lambda *_args: "/usr/bin:/bin")
    if verdict == "crash":
        with pytest.raises(RuntimeError, match="exited with status 4") as error:
            asyncio.run(swe._run_evaluator(tmp_path))
        assert len(str(error.value)) < 2100
        assert len((logs / "stderr.log").read_bytes()) > 10000
    else:
        assert asyncio.run(swe._run_evaluator(tmp_path)) == float(verdict == "PASSED")
        assert verdict.encode() in (logs / "stdout.log").read_bytes()
    assert json.loads((logs / "status.json").read_text())["finished"] is True
