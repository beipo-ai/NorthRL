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

"""Capture a packaged evaluator's output without changing task entrypoint hooks.

This file also runs directly with the sandbox's isolated Python interpreter;
keep it standard-library-only. The wrapper contains the original script and
restores it at the same path before invoking Bash, preserving $0/BASH_SOURCE,
relative includes, arguments and the evaluator's own traps and shell options.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

_SOURCE_MARKER = b"\n# NORTHRL_ORIGINAL_EVALUATOR_BASE64\n# "
_CHUNK_SIZE = 64 * 1024


def wrap_evaluator_script(
    source: bytes, *, interpreter: str, helper: str, input_path: str, log_dir: str
) -> bytes:
    """Wrap only the uploaded copy; the host's evaluation assets stay untouched."""
    command = shlex.join(
        [interpreter, "-I", helper, "--input", input_path, "--log-dir", log_dir, "--"]
    )
    return (
        f'#!/bin/bash\nexec {command} "$0" "$@"\n'.encode()
        + _SOURCE_MARKER
        + base64.b64encode(source)
        + b"\n"
    )


def _restore_script(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evaluator wrapper must be a regular file")
    wrapper = path.read_bytes()
    _prefix, marker, encoded = wrapper.rpartition(_SOURCE_MARKER)
    if not marker:
        raise ValueError("Evaluator wrapper has no embedded source")
    original = base64.b64decode(encoded.strip(), validate=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(original)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            temporary.chmod(0o600)
            json.dump(status, stream)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _warn(error: Exception) -> None:
    print(
        f"[E2B evaluator logs] capture unavailable: {type(error).__name__}: {error}",
        file=sys.stderr,
        flush=True,
    )


def capture_evaluator(script: Path, args: list[str], *, input_path: Path, log_dir: Path) -> int:
    """Tee both pipes incrementally; log failures do not change the evaluator result."""
    _restore_script(script)
    command = ["/bin/bash", str(script), *args]
    with ExitStack() as stack:
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            status: dict[str, Any] = {
                "rollout_id": payload["input_data"]["rollout_id"],
                "attempt_id": payload["attempt_id"],
                "started_at": time.time(),
                "finished": False,
                "returncode": None,
                "capture_errors": {},
            }
            log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            log_dir.chmod(0o700)
            outputs = {}
            for name in ("stdout", "stderr"):
                target = log_dir / f"{name}.log"
                outputs[name] = stack.enter_context(target.open("wb", buffering=0))
                target.chmod(0o600)
            _write_status(log_dir / "status.json", status)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _warn(exc)
            # Logging initialization is optional; the restored script still runs normally.
            os.execv(command[0], command)

        with (
            subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
            ) as process,
            selectors.DefaultSelector() as selector,
        ):
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, ("stdout", 1))
            selector.register(process.stderr, selectors.EVENT_READ, ("stderr", 2))
            signal_received_at: float | None = None

            def forward_signal(signum: int, _frame: Any) -> None:
                nonlocal signal_received_at
                signal_received_at = time.monotonic()
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous = signal.signal(signum, forward_signal)
                stack.callback(signal.signal, signum, previous)

            sizes = {"stdout": 0, "stderr": 0}
            forwarding_failed: set[int] = set()
            while selector.get_map() or process.poll() is None:
                if signal_received_at is not None and time.monotonic() - signal_received_at > 2:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    signal_received_at = None
                for key, _events in selector.select(timeout=0.1):
                    name, descriptor = key.data
                    chunk = os.read(key.fd, _CHUNK_SIZE)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if name not in status["capture_errors"]:
                        try:
                            pending = memoryview(chunk)
                            while pending:
                                written = outputs[name].write(pending)
                                sizes[name] += written
                                pending = pending[written:]
                        except OSError as exc:
                            status["capture_errors"][name] = f"{type(exc).__name__}: {exc}"
                            _warn(exc)
                    # os.write can be short even when the original pipe is blocking.
                    if descriptor not in forwarding_failed:
                        try:
                            pending = memoryview(chunk)
                            while pending:
                                pending = pending[os.write(descriptor, pending) :]
                        except BrokenPipeError:
                            forwarding_failed.add(descriptor)
                            forward_signal(signal.SIGPIPE, None)
            returncode = process.wait()
            status.update(
                finished=True, returncode=returncode, finished_at=time.time(), bytes=sizes
            )
            try:
                _write_status(log_dir / "status.json", status)
            except OSError as exc:
                _warn(exc)
            return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    returncode = capture_evaluator(
        args.script, args.script_args, input_path=args.input, log_dir=args.log_dir
    )
    if returncode < 0:
        signum = -returncode
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
