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

"""MiMo open-source code tasks with delayed hidden-test injection."""

from __future__ import annotations

import asyncio
import json
import math
import os
import pwd
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)
from northrl.remote_rollout.network_policy import (
    agent_network_command,
    agent_network_environment,
    prepare_network_policy,
)

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_STATE_ROOT = Path("/root/northrl-mimo")
_IMAGE_PATH = Path("/etc/mimo-e2b/image-path")
_ROOT_UID = 0


class CommandTimeoutError(RolloutEnvironmentError):
    """A subprocess exceeded the task's configured execution deadline."""


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RolloutEnvironmentError(f"MiMo {name} must be a positive finite number")
    return float(value)


def _agent_identity(input_data: RolloutInput) -> pwd.struct_passwd:
    user = input_data.task.get("agent_user")
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]{0,30}", user):
        raise RolloutEnvironmentError("MiMo requires a validated agent_user")
    try:
        identity = pwd.getpwnam(user)
    except KeyError as exc:
        raise RolloutEnvironmentError("MiMo template agent user is missing") from exc
    if identity.pw_uid == 0 or identity.pw_dir != f"/home/{user}":
        raise RolloutEnvironmentError("MiMo agent must use its non-root template account")
    return identity


def resolve_workspace(input_data: RolloutInput) -> Path:
    raw = input_data.task.get("cwd")
    if not isinstance(raw, str) or not raw.startswith("/") or ".." in Path(raw).parts:
        raise RolloutEnvironmentError("MiMo task.cwd must be an absolute template directory")
    path = Path(raw)
    if not path.is_dir() or path.resolve() != path:
        raise RolloutEnvironmentError("MiMo workspace is missing or traverses a symlink")
    return path


def _require_private_path(path: Path) -> None:
    if path.is_symlink():
        raise RolloutEnvironmentError("MiMo trusted path must not be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo trusted path is missing") from exc
    if metadata.st_uid != _ROOT_UID or metadata.st_mode & 0o022:
        raise RolloutEnvironmentError("MiMo trusted path must be root-owned and non-writable")


def _agent_binary(input_data: RolloutInput) -> Path:
    try:
        path = Path("/opt/mimo-agents/claude").resolve(strict=True)
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo template Claude binary is missing") from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RolloutEnvironmentError("MiMo template Claude binary is not executable")
    _require_private_path(path)
    if path.is_relative_to(resolve_workspace(input_data)):
        raise RolloutEnvironmentError("MiMo Claude binary must be outside the task workspace")
    return path


def _runtime_environment(input_data: RolloutInput, *, user: str | None = None) -> dict[str, str]:
    identity = _agent_identity(input_data)
    selected_user = identity.pw_name if user is None else user
    home = "/root" if selected_user == "root" else identity.pw_dir
    image_path = _IMAGE_PATH.read_text(encoding="utf-8") if _IMAGE_PATH.is_file() else _SYSTEM_PATH
    # Preserve the source image's toolchain paths without inheriting model/judge credentials.
    paths = [*image_path.split(":"), *_SYSTEM_PATH.split(":")]
    if selected_user != "root":
        paths.insert(0, f"{identity.pw_dir}/.local/bin")
    else:
        trusted_paths = []
        for raw_path in paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                continue
            try:
                candidate = candidate.resolve(strict=True)
                for parent in (candidate, *candidate.parents):
                    _require_private_path(parent)
            except (OSError, RolloutEnvironmentError):
                continue
            if candidate.is_dir():
                trusted_paths.append(str(candidate))
        paths = trusted_paths
    environment = {
        "HOME": home,
        "USER": selected_user,
        "LOGNAME": selected_user,
        "PATH": ":".join(dict.fromkeys(path for path in paths if path.startswith("/"))),
    }
    for name in ("LANG", "LC_ALL", "TZ"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def _private_directory(input_data: RolloutInput) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", input_data.rollout_id) or input_data.rollout_id in {
        ".",
        "..",
    }:
        raise RolloutEnvironmentError("Invalid MiMo rollout state identity")
    for path in (_STATE_ROOT, _STATE_ROOT / input_data.rollout_id):
        if path.is_symlink():
            raise RolloutEnvironmentError("MiMo private state must not traverse a symlink")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_path(path)
        path.chmod(0o700)
    return _STATE_ROOT / input_data.rollout_id


def _evaluation_file(evaluation_assets: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RolloutEnvironmentError("Invalid MiMo evaluation asset path")
    _require_private_path(evaluation_assets)
    evaluation_assets.chmod(0o700)
    path = evaluation_assets / relative
    for parent in [*path.parents]:
        if parent == evaluation_assets:
            break
        if parent.is_symlink():
            raise RolloutEnvironmentError("MiMo evaluation asset traverses a symlink")
    _require_private_path(path)
    if not path.is_file():
        raise RolloutEnvironmentError("MiMo evaluation asset must be a file")
    return path


async def _finish_cleanup(cleanup: asyncio.Task) -> None:
    """Complete cleanup even if the caller is cancelled repeatedly."""
    cancelled = False
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            if cleanup.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def _run_command(
    argv: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Capture output privately; errors deliberately omit possibly secret output."""
    arguments = [str(arg) for arg in argv]
    if not arguments:
        raise RolloutEnvironmentError("MiMo subprocess command is empty")
    # These hooks run as root: never resolve a utility via the agent's PATH.
    if "/" not in arguments[0]:
        executable = shutil.which(arguments[0], path=_SYSTEM_PATH)
        if executable is None:
            raise RolloutEnvironmentError("MiMo subprocess utility is unavailable")
        arguments[0] = executable
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            env={"PATH": _SYSTEM_PATH} if env is None else env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RolloutEnvironmentError("MiMo subprocess could not be started") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (asyncio.CancelledError, TimeoutError) as exc:
        await _finish_cleanup(asyncio.create_task(_kill_process_group(process)))
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise CommandTimeoutError("MiMo subprocess exceeded its timeout") from exc
    returncode = process.returncode
    if returncode is None:
        raise RolloutEnvironmentError("MiMo subprocess did not terminate")
    result = subprocess.CompletedProcess(
        list(argv),
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
    if check and result.returncode:
        raise RolloutEnvironmentError(f"MiMo subprocess exited with status {result.returncode}")
    return result


def _agent_processes_running(uid: int) -> bool:
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = dict(
                line.split(":", 1)
                for line in (directory / "status").read_text().splitlines()
                if ":" in line
            )
        except FileNotFoundError:
            continue
        if int(fields["Uid"].split()[0]) == uid and fields["State"].split()[0] != "Z":
            return True
    return False


async def _stop_agent_processes(input_data: RolloutInput) -> None:
    identity = _agent_identity(input_data)
    result = await _run_command(
        ["pkill", "-KILL", "-u", str(identity.pw_uid)], check=False, timeout=10
    )
    if result.returncode not in (0, 1):
        raise RolloutEnvironmentError("MiMo agent process cleanup failed")
    for _ in range(100):
        if not _agent_processes_running(identity.pw_uid):
            return
        await asyncio.sleep(0.05)
    raise RolloutEnvironmentError("MiMo agent processes remained alive after cleanup")


def _test_paths(numstat: str) -> list[str]:
    """Read Git's NUL-delimited patch statistics, including rename endpoints."""
    paths: set[str] = set()
    fields = iter(numstat.split("\0"))
    for entry in fields:
        if not entry:
            continue
        columns = entry.split("\t", 2)
        if len(columns) != 3:
            raise RolloutEnvironmentError("Invalid hidden test patch statistics")
        if columns[2]:
            paths.add(columns[2])
        else:
            try:
                paths.update((next(fields), next(fields)))
            except StopIteration as exc:
                raise RolloutEnvironmentError("Invalid renamed hidden test paths") from exc
    for path in paths:
        parts = PurePosixPath(path).parts
        if not parts or path.startswith("/") or ".." in parts or ".git" in parts:
            raise RolloutEnvironmentError("Unsafe hidden test patch path")
    return sorted(paths)


def _reset_tests(base: str, paths: list[str]) -> str:
    """Restore exactly the hidden test paths, keeping the submitted implementation."""
    lines = ["set -eu"]
    for path in paths:
        lines.extend(
            [
                f"p={shlex.quote(path)}",
                'parent=$(dirname -- "$p"); while [ "$parent" != . ]; do',
                '  test ! -L "$parent" || { echo "Symlink test parent" >&2; exit 1; }',
                '  parent=$(dirname -- "$parent")',
                "done",
                f'git --literal-pathspecs reset -q {shlex.quote(base)} -- "$p"',
                'rm -rf -- "$p"',
                f"if git cat-file -e {shlex.quote(base + ':' + path)} 2>/dev/null; then",
                f'  git --literal-pathspecs restore --source={shlex.quote(base)} --worktree -- "$p"',
                "fi",
            ]
        )
    return "\n".join(lines)


async def prepare(input_data: RolloutInput, workspace: Path) -> None:
    """Keep the public baseline history while removing every other Git object."""
    _agent_binary(input_data)
    git_directory = workspace / ".git"
    if not git_directory.is_dir() or git_directory.is_symlink():
        raise RolloutEnvironmentError("MiMo code template requires a real Git directory")
    # Shared/partial repositories and active operations need a template-specific
    # migration; never silently change their index or external object stores.
    for name in (
        "commondir shallow modules info/grafts objects/info/alternates objects/info/http-alternates "
        "index.lock MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply BISECT_LOG"
    ).split():
        if (git_directory / name).exists() or (git_directory / name).is_symlink():
            raise RolloutEnvironmentError(
                "MiMo Git cleanup requires a standalone, complete repository"
            )
    if (
        any(
            (git_directory / name).is_symlink()
            for name in ("objects", "objects/pack", "objects/info", "info")
        )
        or any(path.is_symlink() for path in (git_directory / "objects").rglob("*"))
        or list((git_directory / "objects/pack").glob("*.promisor"))
    ):
        raise RolloutEnvironmentError("MiMo Git cleanup does not support shared or partial objects")
    environment = _runtime_environment(input_data, user="root")
    environment.update(
        {
            "PATH": _SYSTEM_PATH,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )

    async def git(*args: str | Path, directory: Path = git_directory, check: bool = True) -> str:
        return (
            await _run_command(
                [
                    "git",
                    "-c",
                    f"safe.directory={workspace}",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "core.fsmonitor=false",
                    "-c",
                    "diff.autoRefreshIndex=false",
                    "-c",
                    "core.logAllRefUpdates=false",
                    "-c",
                    "gc.auto=0",
                    *args,
                ],
                cwd=workspace,
                env={**environment, "GIT_DIR": str(directory), "GIT_WORK_TREE": str(workspace)},
                check=check,
            )
        ).stdout

    base = (await git("rev-parse", "--verify", "HEAD^{commit}")).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", base):
        raise RolloutEnvironmentError("MiMo code template has an invalid baseline commit")
    if (await git("rev-parse", "--shared-index-path")).strip():
        raise RolloutEnvironmentError("MiMo Git cleanup does not support a shared index")
    if await git("diff", "--cached", "--name-only", "-z"):
        raise RolloutEnvironmentError(
            "MiMo Git cleanup requires an index matching HEAD without conflicts"
        )
    branch = (await git("symbolic-ref", "-q", "HEAD", check=False)).strip()
    index = git_directory / "index"
    if index.is_symlink():
        raise RolloutEnvironmentError("MiMo Git cleanup does not support a linked index")
    index_bytes = index.read_bytes() if index.is_file() else None
    snapshot_commands = [
        ("diff", "--no-ext-diff", "--no-textconv", "--binary", "--full-index"),
        ("ls-files", "--stage", "-z"),
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    ]
    before = [(await git(*command)) for command in snapshot_commands]
    ancestors = set((await git("rev-list", base)).splitlines())
    records = (
        await git("for-each-ref", "--format=%(refname) %(objectname) %(objecttype) %(symref)")
    ).splitlines()
    refs: dict[str, tuple[str, str]] = {}
    for record in records:
        name, oid, kind, *symbolic = record.split()
        if name.startswith("refs/replace/"):
            continue
        target = oid
        if kind == "tag":
            target = (
                await git("rev-parse", "--verify", "--quiet", f"{oid}^{{commit}}", check=False)
            ).strip()
        if target in ancestors:
            refs[name] = (oid, symbolic[0] if symbolic else "")
    if any(target and target not in refs for _, target in refs.values()):
        raise RolloutEnvironmentError("MiMo Git cleanup encountered an unsupported symbolic ref")
    allowed = set(
        (await git("rev-list", "--objects", "--no-object-names", base, *refs)).splitlines()
    )
    for entry in before[1].split("\0"):
        if not entry:
            continue
        mode, oid, stage = entry.split("\t", 1)[0].split()
        # Gitlinks name commits in another repository, not superproject objects.
        if stage != "0" or (mode != "160000" and oid not in allowed):
            raise RolloutEnvironmentError(
                "MiMo Git cleanup index references objects outside the baseline"
            )
    original_objects = set(
        (await git("cat-file", "--batch-all-objects", "--batch-check=%(objectname)")).splitlines()
    )
    private = _private_directory(input_data)
    bundle, trusted_git, agent_git, original_git = (
        private / name for name in ("baseline.bundle", "baseline.git", "agent.git", "original.git")
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (bundle, trusted_git, agent_git, original_git)
    ):
        raise RolloutEnvironmentError("MiMo baseline state already exists")
    await git("bundle", "create", bundle, "HEAD", *refs)
    object_format = (await git("rev-parse", "--show-object-format")).strip()
    for destination in (trusted_git, agent_git):
        await _run_command(
            ["git", "init", "--bare", f"--object-format={object_format}", destination],
            env=environment,
        )
    await git(
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        bundle,
        "HEAD:refs/heads/baseline",
        directory=trusted_git,
    )
    bundle.unlink()
    # Drop the transfer copy before creating the agent's independent store.
    # Hard links would let agent writes corrupt the private verifier baseline.
    shutil.copytree(trusted_git / "objects", agent_git / "objects", dirs_exist_ok=True)
    for name, (oid, target) in refs.items():
        if target:
            await git("symbolic-ref", name, target, directory=agent_git)
        else:
            await git("update-ref", name, oid, directory=agent_git)
    if branch:
        await git("symbolic-ref", "HEAD", branch, directory=agent_git)
    else:
        await git("update-ref", "--no-deref", "HEAD", base, directory=agent_git)
    # Preserve the index and project configuration, not old logs, hooks, pseudo
    # refs or arbitrary metadata that could contain copies of future answers.
    for name in (
        "config",
        "config.worktree",
        "index",
        "info/exclude",
        "info/attributes",
        "info/sparse-checkout",
    ):
        source = git_directory / name
        if source.is_symlink():
            raise RolloutEnvironmentError("MiMo Git cleanup does not support linked metadata")
        if source.is_file():
            (agent_git / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, agent_git / name)
    actual = set(
        (
            await git(
                "cat-file",
                "--batch-all-objects",
                "--batch-check=%(objectname)",
                directory=agent_git,
            )
        ).splitlines()
    )
    if actual != allowed:
        raise RolloutEnvironmentError(
            "MiMo Git cleanup did not produce exactly the allowed objects"
        )
    shutil.move(str(git_directory), original_git)
    shutil.move(str(agent_git), git_directory)
    after = [(await git(*command)) for command in snapshot_commands]
    if (
        (await git("rev-parse", "HEAD")).strip() != base
        or (await git("symbolic-ref", "-q", "HEAD", check=False)).strip() != branch
        or (index.read_bytes() if index.is_file() else None) != index_bytes
        or after != before
    ):
        raise RolloutEnvironmentError("MiMo Git cleanup changed HEAD, index or workspace status")
    identity = _agent_identity(input_data)
    for directory, directories, files in os.walk(git_directory):
        for name in directories + files:
            os.chown(
                Path(directory) / name, identity.pw_uid, identity.pw_gid, follow_symlinks=False
            )
    os.chown(git_directory, identity.pw_uid, identity.pw_gid, follow_symlinks=False)
    shutil.rmtree(original_git)
    state = {
        "base": base,
        "rollout_id": input_data.rollout_id,
        "git_cleanup": {
            "removed_refs": len(records) - len(refs),
            "removed_objects": len(original_objects - actual),
            "object_count": len(actual),
            "head_preserved": True,
            "index_preserved": True,
            "diff_preserved": True,
        },
    }
    path = private / "state.json"
    if path.exists() or path.is_symlink():
        _require_private_path(path)
        path.unlink()
    path.write_text(json.dumps(state), encoding="utf-8")
    path.chmod(0o600)
    await prepare_network_policy(identity.pw_name)


async def run_agent(
    input_data: RolloutInput, workspace: Path, model_environment: dict[str, str]
) -> AgentResult:
    """Run the template CLI, preserve its exit status and finish agent cleanup."""
    instruction = input_data.task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RolloutEnvironmentError("MiMo task.instruction must be non-empty text")
    for name in ("runuser", "pkill"):
        if shutil.which(name, path=_SYSTEM_PATH) is None:
            raise RolloutEnvironmentError(f"MiMo template requires the {name} utility")
    identity = _agent_identity(input_data)
    binary = _agent_binary(input_data)
    environment = _runtime_environment(input_data)
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"):
        value = model_environment.get(name)
        if not isinstance(value, str) or not value:
            raise RolloutEnvironmentError("MiMo Claude requires the rollout model gateway")
        environment[name] = value
    environment.update(
        {
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_UPDATES": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        }
    )
    command: list[str] = [
        "runuser",
        "--user",
        identity.pw_name,
        "--preserve-environment",
        "--",
        str(binary),
        "--bare",
        "--print",
        "--model",
        "northrl",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Bash,Read,Edit,Write,Glob,Grep",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--",
        instruction,
    ]
    environment = agent_network_environment(environment, identity.pw_name)
    command = agent_network_command(command, identity.pw_name)
    try:
        result = await _run_command(
            command,
            cwd=workspace,
            env=environment,
            check=False,
        )
        return AgentResult(exit_code=result.returncode)
    except CommandTimeoutError:
        return AgentResult(exit_code=124)
    finally:
        await _finish_cleanup(asyncio.create_task(_stop_agent_processes(input_data)))


async def evaluate(
    input_data: RolloutInput, workspace: Path, agent_result: AgentResult, evaluation_assets: Path
) -> EvaluationResult:
    del agent_result
    state_path = _private_directory(input_data) / "state.json"
    _require_private_path(state_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo private rollout state") from exc
    if not isinstance(state, dict) or state.get("rollout_id") != input_data.rollout_id:
        raise RolloutEnvironmentError("MiMo private state belongs to another rollout")
    base = state.get("base")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40,64}", base):
        raise RolloutEnvironmentError("MiMo code evaluation has no valid initial commit")
    try:
        config = json.loads(_evaluation_file(evaluation_assets, "evaluation.json").read_text())
    except (OSError, ValueError) as exc:
        raise RolloutEnvironmentError("Invalid MiMo evaluation config") from exc
    if not isinstance(config, dict):
        raise RolloutEnvironmentError("MiMo evaluation config must be an object")
    if not isinstance(config.get("test_command"), str) or not config["test_command"].strip():
        raise RolloutEnvironmentError("MiMo code evaluation requires its original test_command")
    timeout = _positive_number(config.get("verifier_timeout_sec"), "verifier timeout")
    patch = _evaluation_file(evaluation_assets, "test.patch")
    root_environment = _runtime_environment(input_data, user="root")
    # Git never uses an executable path from the writable task checkout.
    root_environment["PATH"] = _SYSTEM_PATH
    trusted_git = _private_directory(input_data) / "baseline.git"
    _require_private_path(trusted_git)
    root_environment.update(
        {
            "GIT_DIR": str(trusted_git),
            "GIT_WORK_TREE": str(workspace),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
    )
    stats = []
    for reverse in (False, True):
        args = ["git", "apply", *(["--reverse"] if reverse else []), "--numstat", "-z", str(patch)]
        stats.append((await _run_command(args, cwd=workspace, env=root_environment)).stdout)
    paths = _test_paths("".join(stats))
    parents = {parent for path in paths for parent in PurePosixPath(path).parents if parent.parts}
    new_parents = {parent for parent in parents if not (workspace / parent).is_dir()}
    await _run_command(
        ["bash", "-c", _reset_tests(base, paths)],
        cwd=workspace,
        env=root_environment,
    )
    await _run_command(["git", "apply", patch], cwd=workspace, env=root_environment)
    identity = _agent_identity(input_data)
    # The original verifier can update its fixtures. Return ownership only for
    # the injected paths and newly created directories after root applies them.
    for path in [*(PurePosixPath(path) for path in paths), *sorted(new_parents)]:
        if any((workspace / parent).is_symlink() for parent in path.parents):
            raise RolloutEnvironmentError("Symlink hidden test parent")
        try:
            os.chown(workspace / path, identity.pw_uid, identity.pw_gid, follow_symlinks=False)
        except FileNotFoundError:
            pass  # Deleted test paths do not need an owner.
    tested = await _run_command(
        agent_network_command(
            [
                "runuser",
                "--user",
                identity.pw_name,
                "--preserve-environment",
                "--",
                "bash",
                "-c",
                config["test_command"],
            ],
            identity.pw_name,
        ),
        cwd=workspace,
        env=agent_network_environment(_runtime_environment(input_data), identity.pw_name),
        timeout=timeout,
        check=False,
    )
    for name, output in (("verifier.stdout", tested.stdout), ("verifier.stderr", tested.stderr)):
        log_path = _private_directory(input_data) / name
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(output)
    if tested.returncode in (126, 127):
        raise RolloutEnvironmentError("MiMo code test command or interpreter is unavailable")
    reward = float(tested.returncode == 0)
    return EvaluationResult(
        reward=reward,
        status="success" if reward else "task_failed",
        metrics={"mimo_code_resolved": reward, "verifier_exit_code": tested.returncode},
    )
