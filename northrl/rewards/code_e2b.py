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

"""Judge prepared code labels in disposable E2B SandboxFusion sandboxes.

The dataset supplies the judge request in ``label``. Legacy suites use sequential
upload batches. A published ``metadata.cases_ref`` selects one shared-storage
sandbox for the complete task, without uploading label.cases. Reward is 1 only
if every case passes; missing verdicts are never treated as accepted.
Raw judge scores are retained for diagnostics, independently of the reward.
Python tasks without an explicit code_template receive their original publisher's
prelude when their metadata identifies PrimeIntellect call-based tasks, TACO, or
LiveCodeBench. Other sources retain their complete-program execution contract.
Empty model source gets zero reward without creating a sandbox. Timed-out judge
commands retry twice in fresh sandboxes; an exhausted run timeout gets zero reward
for both legacy uploaded batches and shared tasks. Other exhausted shared run
failures still raise infrastructure errors.
Cleanup failures never discard validated verdicts or block otherwise eligible
retries. After a valid verdict, failed sandbox termination is retried once;
unconfirmed cleanup is logged and left to the server-side sandbox lifetime.
Shared evaluator exits retry within the same judging budget unless they report
a known configuration/data error. Failed commands log bounded, redacted tails.
Transient creation and command warmup failures each retry twice with jitter.
Uploads have a separate local deadline and retry four times with capped
exponential jitter in fresh sandboxes. An expired whole-attempt deadline never retries or
supplies a zero reward; exhausted setup and other infrastructure failures propagate.
Confirmed internal warmup cancellation uses that same warmup budget; caller
cancellation and graceful worker stop never start another sandbox.
"""

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import shlex
import ssl
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..utils.e2b_transport import set_e2b_tcp_user_timeout
from ._runtime_context import reward_stop_requested

logger = logging.getLogger(__name__)

_REQUEST_PATH = "/tmp/code-verifier-request.json"
# Legacy source and checker text travel in uploaded JSON. Shared requests use
# a quoted, base64 descriptor or a small JSON file, never executable source text.
_JUDGE_COMMAND = (
    "/root/miniconda3/bin/python /opt/e2b-sandboxfusion/judge.py --stdio"
    " < /tmp/code-verifier-request.json"
)
_MAX_REQUEST_BYTES = 1024 * 1024 * 1024
_MAX_RESULT_BYTES = 1024 * 1024 * 1024
_BATCH_BYTES = 10 * 1024 * 1024
_BATCH_CASES = 100
_SANDBOX_PROBE_TIMEOUT_S = 3.0
_SANDBOX_ENVD_PORT = 49983
_COMMAND_WARMUP_TIMEOUT_S = 30.0
_WARMUP_CLEANUP_TIMEOUT_S = 3.0
_MAX_CREATE_FAILURES = 3
_MAX_UPLOAD_FAILURES = 5
_UPLOAD_REQUEST_TIMEOUT_S = 300.0
_UPLOAD_RETRY_TIMEOUT_STEP = 0.5
_UPLOAD_RETRY_TIMEOUT_MAX_MULTIPLIER = 3.0
_UPLOAD_RETRY_INITIAL_DELAY_S = 10.0
_UPLOAD_RETRY_MAX_DELAY_S = 60.0
_MIN_ATTEMPT_TIMEOUT_S = 20 * 60
_MAX_WARMUP_FAILURES = 3
_MAX_RUN_TIMEOUTS = 3
_SANDBOX_CLEANUP_TIMEOUT_S = 10.0
_ERROR_CONTEXT_FIELDS = (
    "task_id",
    "group_id",
    "sample_id",
    "batch_index",
    "attempt",
    "payload_bytes",
    "sandbox_id",
)
_FENCED_CODE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_VEPFS_ROOT = "/mnt/data/mixrl-eval/v1"
_VEPFS_MOUNT_PATH = "/mnt/data"
_VEPFS_MAX_INLINE_COMMAND_BYTES = 64 * 1024
_VEPFS_MAX_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_VEPFS_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VEPFS_BOOTSTRAP_CONFIG_EXIT = 78
_RETRYABLE_JUDGE_EXIT_CODES = frozenset({1, 134, 137, 139})
_JUDGE_ERROR_JSON_MAX_CHARS = 64 * 1024

# Preserve the publishers' bindings and order, rather than merging their imports.
# PrimeIntellect-ai/genesys @ 32e8ccc14ab5e94b22d37c14f8737bb5983d1e10:
# src/genesys/verifiers/code_test_verifier.py, call-based injection only.
_PRIME_PYTHON_PRELUDE = "from typing import List, Dict, Set, Tuple, Union, Optional\n"
# FlagOpen/TACO @ 245eba3beb2d23a07082307de303de2589e4321a:
# metrics/testing_util.py, synthesize_cb_code / synthesize_std_code.
_TACO_PYTHON_PRELUDE = """import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *
"""
# synthesize_std_code also binds these aliases before the candidate's body.
# Resolve the streams in the sandbox on each execution, not in this process.
_TACO_STDIO_PYTHON_PRELUDE = _TACO_PYTHON_PRELUDE + "stdin = sys.stdin\nstdout = sys.stdout\n"
# LiveCodeBench/LiveCodeBench @ 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24:
# lcb_runner/evaluation/testing_util.py, import_string.
_LCB_PYTHON_PRELUDE = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


class _EmptyCodeError(ValueError):
    """The model response has no non-empty source after code extraction."""


class _UnexpectedWarmupCancellation(ConnectionError):
    """The owned warmup task was cancelled while its caller remained active."""


class CodeE2BError(RuntimeError):
    """An unscored judge failure, with a stage and redacted diagnostic."""

    def __init__(
        self,
        stage: str,
        error: Exception,
        *,
        is_timeout: bool = False,
        attempt_deadline_expired: bool = False,
        retryable_create: bool = False,
        retryable_upload: bool = False,
        retryable_warmup: bool = False,
        retryable_run: bool = False,
        warmup_cleanup_incomplete: bool = False,
        sandbox_cleanup_incomplete: bool = False,
        diagnostic_context: Mapping[str, Any] | None = None,
    ):
        self.stage = stage
        self.is_timeout = is_timeout
        self.attempt_deadline_expired = attempt_deadline_expired
        self.retryable_create = retryable_create
        self.retryable_upload = retryable_upload
        self.retryable_warmup = retryable_warmup
        self.retryable_run = retryable_run
        self.warmup_cleanup_incomplete = warmup_cleanup_incomplete
        self.sandbox_cleanup_incomplete = sandbox_cleanup_incomplete
        self.error_type = type(error).__name__
        diagnostic_context = diagnostic_context or {}
        context = "; ".join(
            f"{key}={diagnostic_context[key]!r}"
            for key in _ERROR_CONTEXT_FIELDS
            if isinstance(diagnostic_context.get(key), (str, int, float, bool))
        )
        super().__init__(
            f"Code E2B {stage} failed: {type(error).__name__}: {_error_message(error)}"
            + (" (attempt deadline expired)" if attempt_deadline_expired else "")
            + (f" [{_error_message(context)}]" if context else "")
        )


@dataclass
class _BatchRetryCounts:
    """Separate bounded setup recovery from the existing judge timeout policy."""

    run_timeouts: int = 0
    create_failures: int = 0
    create_retries: int = 0
    upload_failures: int = 0
    upload_retries: int = 0
    warmup_failures: int = 0
    warmup_retries: int = 0
    run_failures: int = 0
    run_retries: int = 0


def _e2b_http_status(error: BaseException) -> int | None:
    """Read structured status, or an explicit status prefix on the SDK base error."""
    from e2b.exceptions import SandboxException  # pylint: disable=import-outside-toplevel

    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None:
        status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 600:
        return status
    # Do not search arbitrary messages or treat auth/template subclasses as 5xx.
    # pylint: disable-next=unidiomatic-typecheck
    if type(error) is SandboxException:
        match = re.match(r"\s*([45][0-9]{2}):", str(error))
        if match:
            return int(match.group(1))
    return None


def _judge_error_type(stdout: str) -> str | None:
    """Read an error type for retry classification only, not for diagnostics."""
    if not isinstance(stdout, str) or len(stdout) > _JUDGE_ERROR_JSON_MAX_CHARS:
        return None
    try:
        result = json.loads(stdout)
    except (ValueError, RecursionError):
        return None
    error = result.get("error") if isinstance(result, dict) else None
    if not isinstance(error, dict):
        return None
    error_type = error.get("type")
    return error_type if isinstance(error_type, str) else None


def _e2b_retry_reason(error: Exception, stage: str) -> str | None:
    """Classify bounded infrastructure and shared-evaluator exit recovery."""
    if stage not in {"create", "upload", "warmup", "run"}:
        return None
    from e2b import CommandExitException  # pylint: disable=import-outside-toplevel
    from e2b import exceptions as sdk  # pylint: disable=import-outside-toplevel

    if isinstance(error, CommandExitException):
        if (
            stage != "run"
            or not isinstance(error.exit_code, int)
            or isinstance(error.exit_code, bool)
            or error.exit_code not in _RETRYABLE_JUDGE_EXIT_CODES
        ):
            return None
        error_type = _judge_error_type(error.stdout)
        if error_type in {
            "AssertionError",
            "FileNotFoundError",
            "PermissionError",
            "ValueError",
            "TypeError",
            "JSONDecodeError",
        }:
            return None

    chain: list[BaseException] = []
    cause: BaseException | None = error
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append(cause)
        # asyncio.timeout converts its own cancellation into TimeoutError.
        # That chained CancelledError is not an external cancellation request.
        if isinstance(
            cause, (TimeoutError, sdk.TimeoutException, httpx.TimeoutException)
        ) and isinstance(cause.__cause__, asyncio.CancelledError):
            break
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    permanent = (
        asyncio.CancelledError,
        ssl.SSLCertVerificationError,
        PermissionError,
        FileNotFoundError,
        sdk.AuthenticationException,
        sdk.InvalidArgumentException,
        sdk.TemplateException,
        sdk.NotEnoughSpaceException,
    )
    if any(isinstance(item, permanent) for item in chain):
        return None
    # An unrelated handled caller exception can appear in __context__. It
    # must not invent a retry reason or turn a real network timeout into a
    # local ValueError. Follow explicit causes for positive classification.
    recovery_chain: list[BaseException] = []
    cause = error
    seen.clear()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        recovery_chain.append(cause)
        if isinstance(
            cause, (TimeoutError, sdk.TimeoutException, httpx.TimeoutException)
        ) and isinstance(cause.__cause__, asyncio.CancelledError):
            break
        cause = cause.__cause__
    if any(isinstance(item, (ValueError, TypeError)) for item in recovery_chain):
        return None
    for item in chain:
        if isinstance(item, sdk.NotFoundException) and (
            stage == "create" or not isinstance(item, sdk.SandboxNotFoundException)
        ):
            return None
    statuses = [_e2b_http_status(item) for item in recovery_chain]
    if any(status is not None and status not in (408, 429) and status < 500 for status in statuses):
        return None

    # Newer SDKs expose Connect/pyqwest errors, sometimes without __cause__.
    rpc_reason = None
    pyqwest_types: tuple[type[Exception], ...] = ()
    try:
        from connectrpc.code import Code
        from connectrpc.errors import ConnectError
        from pyqwest import ReadError, StreamError, WriteError
    except ImportError:
        pass
    else:
        pyqwest_types = (ReadError, StreamError, WriteError)
        retryable_codes = {
            Code.UNKNOWN,
            Code.UNAVAILABLE,
            Code.INTERNAL,
            Code.DEADLINE_EXCEEDED,
            Code.RESOURCE_EXHAUSTED,
            Code.ABORTED,
        }
        for item in recovery_chain:
            if isinstance(item, ConnectError):
                if item.code not in retryable_codes:
                    return None
                rpc_reason = f"rpc_{item.code.value}"
    if rpc_reason is not None:
        return rpc_reason
    for status in statuses:
        if status is not None:
            return f"http_{status}"
    if any(isinstance(item, sdk.RateLimitException) for item in recovery_chain):
        return "rate_limit"
    if stage != "create" and any(
        isinstance(item, sdk.SandboxNotFoundException) for item in recovery_chain
    ):
        return "sandbox_missing"
    transport_types = (
        ConnectionError,
        TimeoutError,
        sdk.TimeoutException,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    )
    if any(isinstance(item, transport_types + pyqwest_types) for item in recovery_chain):
        return "transport"
    return "command_exit" if isinstance(error, CommandExitException) else None


def _is_upload_transport_error(error: Exception) -> bool:
    return _e2b_retry_reason(error, "upload") is not None


def _is_warmup_transport_error(error: Exception) -> bool:
    return _e2b_retry_reason(error, "warmup") is not None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _vepfs_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _VEPFS_DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _vepfs_path(value: Any, directory: str, digest: str, suffix: str) -> str:
    expected = f"{_VEPFS_ROOT}/{directory}/{digest}{suffix}"
    if value != expected:
        raise ValueError(
            f"shared {directory} path must be its content-addressed path under {_VEPFS_ROOT}"
        )
    return value


@dataclass(frozen=True)
class _VepfsRequest:
    """Optional vePFS request used by the existing code_e2b reward."""

    volume_name: str
    reader_path: str
    reader_sha256: str
    payload: bytes

    def command(self, request_path: str) -> tuple[str, bool]:
        """Return a verified bootstrap and whether its request needs uploading."""
        # Never interpolate candidate source as shell/Python syntax. The reader
        # remains shared; only its locator/hash and the small descriptor travel.
        setup = (
            "import hashlib,io,os,pathlib,stat,sys\n"
            "try:\n"
            f" p=pathlib.Path({self.reader_path!r})\n"
            " assert str(p.resolve(strict=True))==str(p), 'reader symlink'\n"
            " fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)\n"
            " with os.fdopen(fd,'rb') as f:\n"
            "  assert stat.S_ISREG(os.fstat(f.fileno()).st_mode), 'reader not regular'\n"
            "  source=f.read(262145)\n"
            " assert len(source)<=262144, 'reader too large'\n"
            f" assert hashlib.sha256(source).hexdigest()=={self.reader_sha256!r}, 'reader hash mismatch'\n"
            "except (AssertionError,FileNotFoundError,PermissionError) as e:\n"
            " print('reader validation failed: '+type(e).__name__+': '+str(e),file=sys.stderr)\n"
            f" raise SystemExit({_VEPFS_BOOTSTRAP_CONFIG_EXIT})\n"
        )
        finish = (
            "sys.argv=[str(p),'--stdio'];"
            "exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})"
        )
        inline = (
            setup
            + "import base64;sys.stdin=io.TextIOWrapper(io.BytesIO(base64.b64decode("
            + repr(base64.b64encode(self.payload).decode("ascii"))
            + ")),encoding='utf-8');"
            + finish
        )
        command = "/root/miniconda3/bin/python -c " + shlex.quote(inline)
        if len(command.encode("utf-8")) <= _VEPFS_MAX_INLINE_COMMAND_BYTES:
            return command, False
        source = setup + f"sys.stdin=open({request_path!r},encoding='utf-8');" + finish
        return "/root/miniconda3/bin/python -c " + shlex.quote(source), True


def _prepare_vepfs_request(
    task: Mapping[str, Any], request: Mapping[str, Any]
) -> _VepfsRequest | None:
    """A present but invalid reference fails; it never silently uploads cases."""
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping) or "cases_ref" not in metadata:
        return None
    reference = metadata["cases_ref"]
    if not isinstance(reference, Mapping):
        raise ValueError("metadata.cases_ref must be a mapping")
    if set(reference) != {"protocol_version", "manifest_path", "manifest_sha256", "case_count"}:
        raise ValueError("metadata.cases_ref has missing or unsupported fields")
    version = reference["protocol_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("unsupported cases_ref protocol_version")
    count = reference["case_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count != len(request["cases"])
    ):
        raise ValueError("cases_ref case_count differs from the complete original label")
    digest = _vepfs_digest(reference["manifest_sha256"], "manifest_sha256")
    _vepfs_path(reference["manifest_path"], "manifests", digest, ".json")
    volume = os.environ.get("E2B_VEPFS_VOLUME", "")
    if not volume or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", volume):
        raise ValueError("E2B_VEPFS_VOLUME must name the published shared volume")
    reader_digest = _vepfs_digest(
        os.environ.get("E2B_VEPFS_READER_SHA256"), "E2B_VEPFS_READER_SHA256"
    )
    reader_path = _vepfs_path(
        os.environ.get("E2B_VEPFS_READER_PATH"), "readers", reader_digest, ".py"
    )
    header = {key: value for key, value in request.items() if key not in {"cases", "code"}}
    descriptor = {
        "protocol_version": 1,
        "code": request["code"],
        "reference": dict(reference),
        "request_header_sha256": hashlib.sha256(_canonical_json_bytes(header)).hexdigest(),
    }
    payload = _canonical_json_bytes(descriptor) + b"\n"
    if len(payload) > _VEPFS_MAX_DESCRIPTOR_BYTES:
        raise ValueError("shared evaluation descriptor exceeds the 16 MiB protocol limit")
    return _VepfsRequest(volume, reader_path, reader_digest, payload)


def _python_prelude(task: Mapping[str, Any], request: Mapping[str, Any]) -> str | None:
    """Select original-source imports using prepared metadata, never prompt text."""
    mode = request.get("input_mode")
    if request.get("language") != "python" or mode not in ("stdio", "function_call"):
        return None
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("data_source") in ("livecodebench_v5", "livecodebench_v6"):
        return _LCB_PYTHON_PRELUDE
    provenance = metadata.get("provenance")
    if metadata.get("source") != "deepcoder" or not isinstance(provenance, Mapping):
        return None
    dataset = provenance.get("dataset")
    if dataset == "lcbv5":
        return _LCB_PYTHON_PRELUDE
    if dataset == "taco":
        return _TACO_STDIO_PYTHON_PRELUDE if mode == "stdio" else _TACO_PYTHON_PRELUDE
    if dataset == "primeintellect" and mode == "function_call":
        return _PRIME_PYTHON_PRELUDE
    return None


def _wrap_python_prelude(code: str, prelude: str, request: Mapping[str, Any]) -> str:
    """Prepare source for execution in E2B; nothing is executed in the driver."""
    # Keep helper imports out of the outer SandboxFusion dispatcher. Otherwise an
    # imported callable such as itertools.count can shadow Solution.count. The
    # candidate's functions still see all prelude bindings through their globals.
    # A real module also supports dataclasses, type resolution and self-imports.
    return f"""def _code_e2b_prepare():
    import sys as _sys
    import types as _types
    _outer = globals()
    _outer.pop('_code_e2b_prepare', None)
    _name = _outer.get('__name__', '__main__')
    _module = _types.ModuleType(_name)
    _scope = _module.__dict__
    for _key in ('__builtins__', '__file__', '__package__', '__loader__', '__spec__'):
        if _key in _outer:
            _scope[_key] = _outer[_key]
    _sys.modules[_name] = _module
    exec(compile({prelude!r}, '<code-e2b-prelude>', 'exec', dont_inherit=True), _scope)
    _imported = dict(_scope)
    exec(compile({code!r}, '<candidate>', 'exec', dont_inherit=True), _scope)
    if {request.get('input_mode') == 'function_call'!r}:
        if 'Solution' in _scope:
            _outer['Solution'] = _scope['Solution']
        _entry = {request.get('fn_name')!r}
        if _entry in _scope and ('Solution' not in _scope or _entry not in _imported
                                or _scope[_entry] is not _imported[_entry]):
            _outer[_entry] = _scope[_entry]
_code_e2b_prepare()
"""


def _build_request(
    task: dict[str, Any], response: str, *, finish_reason: str | None = None
) -> dict[str, Any]:
    """Insert the answer into the prepared label without mutating the sample."""
    label = task.get("label")
    if not isinstance(label, Mapping):
        raise TypeError("Code E2B task.label must be a mapping of prepared judge fields")
    request = dict(label)
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("judge request must contain at least one case")
    template = request.pop("code_template", None)
    if template is not None and (not isinstance(template, str) or "__MODEL_CODE__" not in template):
        raise ValueError("Code E2B code_template must contain __MODEL_CODE__")
    try:
        code = _extract_code(response)
    except (TypeError, ValueError) as error:
        empty_source = isinstance(error, _EmptyCodeError)
        logger.log(
            logging.WARNING if empty_source else logging.ERROR,
            "Code E2B code extraction failed: %s; task_id=%r; group_id=%r; "
            "sample_id=%r; finish_reason=%r; reward=%r; reason=%r; response=%r",
            error,
            task.get("task_id"),
            task.get("group_id"),
            task.get("sample_id"),
            finish_reason,
            0.0 if empty_source else None,
            "no_code" if empty_source else None,
            response,
        )
        raise
    if template is not None:
        code = template.replace("__MODEL_CODE__", repr(code), 1)
    elif (prelude := _python_prelude(task, request)) is not None:
        code = _wrap_python_prelude(code, prelude, request)
    request["code"] = code
    return request


def _extract_code(response: str) -> str:
    """Accept plain source or the final non-empty fenced block of a response."""
    if not isinstance(response, str):
        raise TypeError("model_code must be a string")
    response = response.rsplit("</think>", 1)[-1]
    blocks = _FENCED_CODE.findall(response)
    code = (
        next((block.strip() for block in reversed(blocks) if block.strip()), "")
        if blocks
        else response.strip()
    )
    if not code:
        raise _EmptyCodeError("model_code must contain non-empty source")
    return code


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _timeouts(
    request: dict[str, Any],
    command_timeout: float | None = None,
    request_timeout: float | None = None,
    *,
    upload_retry_count: int = 0,
    whole_task: bool = False,
) -> tuple[float, float, float, int, int]:
    rpc = _positive_number(
        (
            request_timeout
            if request_timeout is not None
            else os.environ.get("E2B_REQUEST_TIMEOUT", 60)
        ),
        "request_timeout",
    )
    base_upload = _positive_number(
        os.environ.get("E2B_UPLOAD_REQUEST_TIMEOUT", max(_UPLOAD_REQUEST_TIMEOUT_S, rpc)),
        "upload_request_timeout",
    )
    # Creation can consume read + write budgets in the pyqwest adapter. Upload
    # and warmup have explicit local deadlines; leave slack for other setup.
    reserve = 2 * rpc + base_upload + min(rpc, _COMMAND_WARMUP_TIMEOUT_S) + 30
    case_count = len(request.get("cases", []))
    per_case = (
        _positive_number(request.get("compile_timeout", 10), "compile_timeout")
        + _positive_number(request.get("run_timeout", 2), "run_timeout")
        + 2
    )
    if request.get("generated_checker"):
        per_case += _positive_number(request.get("checker_timeout", 5), "checker_timeout") + 2
    estimated = max(60, math.ceil(case_count * per_case + 30))
    # Legacy batches keep their one-hour default ceiling. Shared whole tasks
    # retain their full estimate; longer upload retries preserve judge time.
    default_budget = estimated if whole_task else min(estimated, max(1, 3600 - math.ceil(reserve)))
    configured = (
        command_timeout
        if command_timeout is not None
        else os.environ.get("E2B_COMMAND_TIMEOUT", default_budget)
    )
    command = _positive_number(configured, "command_timeout")
    upload = base_upload * min(
        1 + _UPLOAD_RETRY_TIMEOUT_STEP * upload_retry_count,
        _UPLOAD_RETRY_TIMEOUT_MAX_MULTIPLIER,
    )
    # Increase the attempt budget with the upload deadline, after computing the
    # judge budget so it stays the same across retries of this batch.
    reserve += upload - base_upload
    # Allow twenty minutes for setup stalls even on small batches, without
    # shortening longer judge budgets. Each retry gets its own deadline.
    lifetime = max(_MIN_ATTEMPT_TIMEOUT_S, math.ceil(command + reserve))
    if lifetime > 86400:
        raise ValueError(
            "command_timeout plus upload/cleanup reserve exceeds E2B's 24-hour maximum"
        )
    return command, rpc, upload, lifetime, estimated


def _redact_error_text(text: str) -> str:
    # SDK exceptions may echo headers or URLs. Do not emit configured credentials.
    secrets: set[str] = set()
    for key, value in os.environ.items():
        if (
            isinstance(value, str)
            and value
            and any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        ):
            # Output may contain JSON-escaped credentials as well as plain text.
            secrets.update(
                (value, json.dumps(value)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1])
            )
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"\be2b_[A-Za-z0-9_-]+", "[REDACTED]", text)

    def redact_url(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group())
            host = parsed.hostname
            if host is None:
                return "[REDACTED_URL]"
            host = f"[{host}]" if ":" in host else host
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{parsed.scheme}://{host}{port}"
        except ValueError:
            return "[REDACTED_URL]"

    text = re.sub(r"https?://[^\s\"'<>]+", redact_url, text)
    return text


def _error_message(exc: Exception | str) -> str:
    if isinstance(exc, Exception):
        try:
            from e2b import CommandExitException  # pylint: disable=import-outside-toplevel
        except ImportError:
            pass
        else:
            if isinstance(exc, CommandExitException):
                # Detailed output belongs to one failure event, not every
                # re-raised traceback and retry log.
                return f"Command exited with code {exc.exit_code} (see command failure details)"
    return _redact_error_text(str(exc))[:2000]


def _command_failure_details(error: Exception) -> dict[str, Any]:
    """Summarize command output without interpreting the checker's protocol."""
    try:
        from e2b import CommandExitException  # pylint: disable=import-outside-toplevel
    except ImportError:
        return {}

    if not isinstance(error, CommandExitException):
        return {}

    def excerpt(value: str, limit: int) -> str:
        redacted = _redact_error_text(value)
        return (
            redacted[-limit:]
            .encode("utf-8", errors="replace")[-limit:]
            .decode("utf-8", errors="ignore")
        )

    stdout = error.stdout if isinstance(error.stdout, str) else ""
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    return {
        "command_exit_code": error.exit_code,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "stdout_tail": excerpt(stdout, 512),
        "stderr_tail": excerpt(stderr, 1024),
    }


def _parse_result(  # pylint: disable=too-many-branches
    stdout: str, expected_cases: int
) -> dict[str, Any]:
    """Validate the judge protocol instead of guessing success from stdout text."""
    result_bytes = len(stdout.encode("utf-8"))
    if result_bytes > _MAX_RESULT_BYTES:
        raise ValueError(
            f"judge result has {result_bytes} bytes, "
            f"exceeding _MAX_RESULT_BYTES={_MAX_RESULT_BYTES}"
        )
    result = json.loads(stdout)
    if not isinstance(result, dict):
        raise ValueError("judge result must be a JSON object")
    if result.get("error"):
        raise ValueError(f"judge returned an error: {result['error']}")
    accepted, score, tests = result.get("accepted"), result.get("score"), result.get("tests")
    if not isinstance(accepted, bool):
        raise ValueError("judge result has no boolean accepted field")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("judge score must be a finite number between zero and one")
    if not isinstance(tests, list) or not tests or len(tests) != expected_cases:
        raise ValueError(f"judge returned an incomplete test list (expected {expected_cases})")
    test_scores = []
    for case in tests:
        if not isinstance(case, dict) or not isinstance(case.get("passed"), bool):
            raise ValueError("each judge test must have a boolean passed field")
        partial = case.get("score", 1.0 if case["passed"] else 0.0)
        if (
            isinstance(partial, bool)
            or not isinstance(partial, (int, float))
            or not math.isfinite(partial)
            or not 0 <= partial <= 1
        ):
            raise ValueError("test score must be a finite number between zero and one")
        test_scores.append(partial)
        if accepted and (not case["passed"] or partial != 1):
            raise ValueError("judge accepted a test without full credit")
    if accepted != all(case["passed"] for case in tests):
        raise ValueError("judge accepted field disagrees with its test results")
    if not math.isclose(score, sum(test_scores) / len(test_scores), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("judge score disagrees with its per-test scores")
    if accepted and score != 1:
        raise ValueError("judge accepted a request without full credit")
    return {
        "status": "ok",
        "reward": float(accepted),
        "correct": accepted,
        "reason": "accepted" if accepted else "failed_tests",
        "sandbox_result": result,
    }


def _iter_request_batches(
    request: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], str, int]]:
    """Yield complete requests while encoding each case once, never a whole task.

    Byte/count targets are soft for one indivisible case. The hard request limit
    includes source/checker overhead and rejects that case explicitly; no tests
    are dropped. Only the current batch's serialized strings stay in memory.
    """
    cases = request.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("judge request must contain at least one case")
    base = {key: value for key, value in request.items() if key != "cases"}
    base_json = json.dumps(base, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    prefix = base_json[:-1] + ("," if base else "") + '"cases":['
    suffix = "]}\n"
    overhead = len(prefix.encode("utf-8")) + len(suffix.encode("utf-8"))
    if overhead >= _MAX_REQUEST_BYTES:
        raise ValueError(
            f"judge source/checker has {overhead} bytes, "
            f"exceeding _MAX_REQUEST_BYTES={_MAX_REQUEST_BYTES}"
        )
    current_cases: list[Any] = []
    encoded_cases: list[str] = []
    current_bytes = overhead
    for index, case in enumerate(cases):
        encoded = json.dumps(case, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        case_bytes = len(encoded.encode("utf-8"))
        if overhead + case_bytes > _MAX_REQUEST_BYTES:
            raise ValueError(
                f"case {index + 1} needs {overhead + case_bytes} request bytes, "
                f"exceeding _MAX_REQUEST_BYTES={_MAX_REQUEST_BYTES}; no case was truncated"
            )
        separator_bytes = 1 if current_cases else 0
        if current_cases and (
            len(current_cases) >= _BATCH_CASES
            or current_bytes + separator_bytes + case_bytes > _BATCH_BYTES
        ):
            yield {**base, "cases": current_cases}, prefix + ",".join(
                encoded_cases
            ) + suffix, current_bytes
            current_cases, encoded_cases = [], []
            current_bytes = overhead
        current_bytes += case_bytes + (1 if current_cases else 0)
        current_cases.append(case)
        encoded_cases.append(encoded)
    if current_cases:
        yield {**base, "cases": current_cases}, prefix + ",".join(
            encoded_cases
        ) + suffix, current_bytes


async def _iter_request_batches_async(
    request: dict[str, Any],
) -> AsyncIterator[tuple[dict[str, Any], bytes, int]]:
    """Pack one batch at a time outside the shared rollout event loop."""
    batches = _iter_request_batches(request)

    def prepare_next_batch() -> tuple[dict[str, Any], bytes, int] | None:
        # A default keeps StopIteration from escaping through the executor Future.
        prepared = next(batches, None)
        if prepared is None:
            return None
        batch, payload, payload_bytes = prepared
        # Otherwise the HTTP client's multipart encoder converts the entire
        # payload to UTF-8 again on the event-loop thread during upload.
        return batch, payload.encode("utf-8"), payload_bytes

    while True:
        # Do not prefetch: the consumer finishes judging before requesting more.
        prepared = await asyncio.to_thread(prepare_next_batch)
        if prepared is None:
            return
        yield prepared


def _safe_url_endpoint(value: Any) -> tuple[str | None, int | None]:
    """Keep only hostname and port, excluding URL credentials and query parameters."""
    if not isinstance(value, str):
        return None, None
    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        port = parsed.port
        if port is None:
            port = {"http": 80, "https": 443}.get(parsed.scheme)
        return parsed.hostname, port
    except ValueError:
        return None, None


def _safe_sandbox_host(sandbox: Any) -> str | None:
    """Read the public sandbox port hostname without importing SDK internals."""
    try:
        get_host = getattr(sandbox, "get_host", None)
        if not callable(get_host):
            return None
        host = get_host(_SANDBOX_ENVD_PORT)
        return _safe_url_endpoint(host)[0]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


class _BatchDiagnostics:
    """Keep timings and bounded metadata local to one sandbox request."""

    def __init__(
        self, batch: dict[str, Any], payload: str | bytes, context: Mapping[str, Any] | None
    ):
        context = context or {}
        self.fields = {
            key: context.get(key)
            for key in ("batch_index", "task_id", "group_id", "sample_id", "finish_reason")
        }
        self.fields["attempt"] = context.get("attempt", 1)
        payload_bytes = context.get("payload_bytes")
        if payload_bytes is None:
            payload_bytes = (
                len(payload) if isinstance(payload, bytes) else len(payload.encode("utf-8"))
            )
        cases = batch.get("cases")
        self.fields.update(
            case_count=len(cases) if isinstance(cases, list) else None,
            payload_bytes=payload_bytes,
            command_timeout_s=None,
            rpc_timeout_s=None,
            upload_timeout_s=None,
            command_request_timeout_s=None,
            warmup_timeout_s=None,
            warmup_phase=None,
            warmup_pid=None,
            lifetime_timeout_s=None,
            estimated_timeout_s=None,
            sandbox_id=None,
            sandbox_host=None,
            sandbox_host_source="get_host",
            sandbox_port=_SANDBOX_ENVD_PORT,
            envd_host=None,
            envd_port=None,
        )
        self.started = time.monotonic()
        self.stage_started = self.started
        self.run_started: float | None = None
        self.run_finished: float | None = None
        self.stage = "config"

    def start_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_started = time.monotonic()
        if stage == "run":
            self.run_started = self.stage_started
        self.log("stage_start")

    def log(self, event: str, level: int = logging.INFO, **details: Any) -> None:
        if (
            level == logging.INFO
            and event in ("stage_start", "stage_complete")
            and self.stage in ("sdk_import", "cleanup")
        ):
            level = logging.DEBUG
        now = time.monotonic()
        if (
            self.stage == "run"
            and event in ("stage_complete", "failure", "cancelled")
            and self.run_finished is None
        ):
            self.run_finished = now
        run_elapsed = None
        if self.run_started is not None:
            run_end = self.run_finished if self.run_finished is not None else now
            run_elapsed = max(0.0, run_end - self.run_started)
        command = self.fields["command_timeout_s"]
        remaining = None
        if run_elapsed is not None and command is not None:
            remaining = max(0.0, command - run_elapsed)
        fields = {
            **self.fields,
            "event": event,
            "stage": self.stage,
            "total_elapsed_s": max(0.0, now - self.started),
            "stage_elapsed_s": max(0.0, now - self.stage_started),
            "run_elapsed_s": run_elapsed,
            "command_time_remaining_s": remaining,
            **details,
        }
        logger.log(
            level,
            "Code E2B batch event=%s; %s",
            event,
            "; ".join(f"{key}={value!r}" for key, value in fields.items()),
            extra={"code_e2b_batch": fields},
        )


async def _probe_sandbox(sandbox: Any, diagnostics: _BatchDiagnostics, rpc: float) -> None:
    """Try a bounded health check without replacing the original judge failure."""
    started = time.monotonic()
    connectivity = "unknown"
    probe_error = None
    try:
        is_running = getattr(sandbox, "is_running", None)
        if not callable(is_running):
            probe_error = "is_running is unavailable"
        else:
            async with asyncio.timeout(_SANDBOX_PROBE_TIMEOUT_S):
                running = await is_running(request_timeout=min(rpc, _SANDBOX_PROBE_TIMEOUT_S))
            if not isinstance(running, bool):
                raise TypeError("is_running returned a non-boolean result")
            connectivity = "running" if running else "not_running"
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise
        probe_error = "CancelledError"
    except Exception as error:  # pylint: disable=broad-exception-caught
        probe_error = f"{type(error).__name__}: {_error_message(error)}"
    diagnostics.log(
        "connectivity_probe",
        logging.WARNING,
        connectivity=connectivity,
        probe_error=probe_error,
        probe_elapsed_s=max(0.0, time.monotonic() - started),
    )


@dataclass
class _WarmupState:
    """Retain ownership of a shielded task and any late command handle."""

    task: asyncio.Task | None = None
    handle: Any = None
    abort_requested: bool = False
    disconnect_started: bool = False


async def _disconnect_warmup(state: _WarmupState, diagnostics: _BatchDiagnostics) -> None:
    if state.handle is None or state.disconnect_started:
        return
    state.disconnect_started = True
    try:
        async with asyncio.timeout(_WARMUP_CLEANUP_TIMEOUT_S):
            await state.handle.disconnect()
    except Exception as error:  # pylint: disable=broad-exception-caught
        diagnostics.log(
            "cleanup_failure",
            logging.WARNING,
            operation="warmup_disconnect",
            error_type=type(error).__name__,
            error=_error_message(error),
        )


def _observe_warmup_completion(task: asyncio.Task) -> None:
    # Also retrieve late exceptions if bounded cleanup had to abandon waiting.
    if not task.cancelled():
        task.exception()


async def _await_warmup(
    sandbox: Any,
    state: _WarmupState,
    diagnostics: _BatchDiagnostics,
    warmup_timeout: float,
    batch_deadline: asyncio.Timeout,
) -> Any:
    """Separate a cancelled SDK operation from cancellation of its caller."""

    async def start_and_wait():
        try:
            state.handle = await sandbox.commands.run(
                "true",
                background=True,
                user="root",
                timeout=warmup_timeout,
                request_timeout=warmup_timeout,
            )
            # A transport can deliver the start confirmation after cancellation.
            # Never continue waiting/judging after cleanup has requested abort.
            if state.abort_requested:
                raise asyncio.CancelledError()
            diagnostics.fields.update(warmup_phase="awaiting_end", warmup_pid=state.handle.pid)
            diagnostics.log("warmup_started")
            return await state.handle.wait()
        finally:
            if state.abort_requested:
                await _disconnect_warmup(state, diagnostics)

    state.task = asyncio.create_task(start_and_wait(), name="code-e2b-warmup")
    state.task.add_done_callback(_observe_warmup_completion)
    async with asyncio.timeout(warmup_timeout) as warmup_deadline:
        try:
            # Shield does not suppress caller cancellation. It only prevents
            # that cancellation from silently becoming the child's cancellation.
            return await asyncio.shield(state.task)
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            caller_cancelling = caller.cancelling() if caller is not None else None
            stopping = reward_stop_requested()
            warmup_expired = warmup_deadline.expired()
            attempt_expired = batch_deadline.expired()
            internal = (
                caller_cancelling == 0
                and not stopping
                and not warmup_expired
                and not attempt_expired
                and state.task.done()
                and state.task.cancelled()
                and not state.abort_requested
            )
            diagnostics.log(
                "warmup_cancelled",
                logging.WARNING if internal else logging.INFO,
                cancellation_kind="internal_warmup" if internal else "propagated",
                caller_cancelling=caller_cancelling,
                worker_stop_requested=stopping,
                warmup_deadline_expired=warmup_expired,
                attempt_deadline_expired=attempt_expired,
                owned_warmup_done=state.task.done(),
                owned_warmup_cancelled=state.task.cancelled(),
                warmup_abort_requested=state.abort_requested,
            )
            if internal:
                raise _UnexpectedWarmupCancellation(
                    "owned warmup task cancelled while caller remained active"
                ) from None
            # Leave external cancellation and timer conversion to asyncio.
            # In particular, never uncancel the caller to enable a retry.
            raise


def _raise_if_reward_stopping() -> None:
    caller = asyncio.current_task()
    if reward_stop_requested() or (caller is not None and caller.cancelling()):
        raise asyncio.CancelledError("Code E2B caller is stopping")


async def _run_batch(
    batch: dict[str, Any],
    payload: str | bytes,
    *,
    context: Mapping[str, Any] | None = None,
    upload_retry_count: int = 0,
    shared: _VepfsRequest | None = None,
) -> dict[str, Any]:
    """Upload one complete request, validate its verdict, and release its sandbox."""
    sandbox = None
    warmup_state = _WarmupState()
    batch_deadline = None
    batch_error = None
    evaluation_complete = False
    diagnostics = _BatchDiagnostics(batch, payload, context)
    rpc = 60.0
    output_bytes = 0
    timeout_error_types: tuple[type[Exception], ...] = (TimeoutError,)

    def check_output(chunk: str) -> None:
        nonlocal output_bytes
        output_bytes += len(chunk.encode("utf-8"))
        if output_bytes > _MAX_RESULT_BYTES:
            raise ValueError(f"judge output exceeds _MAX_RESULT_BYTES={_MAX_RESULT_BYTES}")

    try:
        command, rpc, upload, lifetime, estimated = _timeouts(
            batch, upload_retry_count=upload_retry_count, whole_task=shared is not None
        )
        warmup_timeout = min(rpc, _COMMAND_WARMUP_TIMEOUT_S)
        diagnostics.fields.update(
            command_timeout_s=command,
            rpc_timeout_s=rpc,
            upload_timeout_s=upload,
            command_request_timeout_s=0,
            warmup_timeout_s=warmup_timeout,
            lifetime_timeout_s=lifetime,
            estimated_timeout_s=estimated,
        )
        if not os.environ.get("E2B_API_KEY"):
            raise ValueError("E2B_API_KEY is required in the training process environment")
        template = os.environ.get("E2B_TEMPLATE", "").strip()
        if not template:
            raise ValueError(
                "E2B_TEMPLATE must name a prepared code-judge template accessible to your "
                "E2B account; set it in the training process environment"
            )
        diagnostics.start_stage("sdk_import")
        from e2b import (  # pylint: disable=import-outside-toplevel
            AsyncSandbox,
            CommandExitException,
            TimeoutException,
        )

        timeout_error_types = (TimeoutError, TimeoutException)

        diagnostics.log("stage_complete")

        opts: dict[str, Any] = {
            "template": template,
            "api_key": os.environ["E2B_API_KEY"],
            "timeout": lifetime,
            "request_timeout": rpc,
            "allow_internet_access": False,
        }
        if shared is not None:
            opts["volume_mounts"] = {_VEPFS_MOUNT_PATH: shared.volume_name}
        if os.environ.get("E2B_DOMAIN"):
            opts["domain"] = os.environ["E2B_DOMAIN"]
        # Bound the complete batch, including creation and upload. Cancellation
        # still reaches the cleanup below before propagating to the worker.
        async with asyncio.timeout(lifetime) as batch_deadline:
            diagnostics.start_stage("create")
            sandbox = await AsyncSandbox.create(**opts)
            diagnostics.fields.update(
                sandbox_id=getattr(sandbox, "sandbox_id", None),
                sandbox_host=_safe_sandbox_host(sandbox),
            )
            try:
                envd_host, envd_port = _safe_url_endpoint(getattr(sandbox, "envd_api_url", None))
                diagnostics.fields.update(envd_host=envd_host, envd_port=envd_port)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            diagnostics.log("stage_complete")
            judge_command, needs_upload = (
                shared.command(_REQUEST_PATH) if shared is not None else (_JUDGE_COMMAND, True)
            )
            diagnostics.start_stage("upload")
            diagnostics.fields["shared_cases"] = shared is not None
            diagnostics.fields["request_upload_bytes"] = len(payload) if needs_upload else 0
            # Shared requests contain prepared code/references only. Small requests
            # travel inside the bootstrap; the reader and label.cases stay shared.
            if needs_upload:
                async with asyncio.timeout(upload):
                    await sandbox.files.write(
                        _REQUEST_PATH, payload, user="root", request_timeout=upload
                    )
            diagnostics.log("stage_complete")
            # File uploads use a different HTTP pool. Establish the command RPC
            # connection before tuning its existing socket, as in the executor.
            diagnostics.start_stage("warmup")
            diagnostics.fields["warmup_phase"] = "awaiting_start"
            warmup = await _await_warmup(
                sandbox, warmup_state, diagnostics, warmup_timeout, batch_deadline
            )
            if warmup.exit_code != 0:
                raise RuntimeError(f"command warmup exited with code {warmup.exit_code}")
            diagnostics.fields["warmup_phase"] = "complete"
            diagnostics.log("warmup_finished", exit_code=warmup.exit_code)
            diagnostics.log("stage_complete")
            diagnostics.start_stage("transport")
            # DNS resolution and scanning process sockets must not stall other
            # sandboxes sharing this rollout event loop.
            transport = await asyncio.to_thread(set_e2b_tcp_user_timeout, sandbox)
            diagnostics.fields.update(
                tcp_user_timeout_ms=transport.timeout_ms,
                tcp_matched_sockets=transport.matched_sockets,
                tcp_configured_sockets=transport.configured_sockets,
                tcp_failed_sockets=transport.failed_sockets,
                tcp_timeout_status=transport.status,
                tcp_previous_timeouts_ms=transport.previous_timeouts_ms,
                tcp_current_timeouts_ms=transport.current_timeouts_ms,
            )
            diagnostics.log("stage_complete")
            diagnostics.start_stage("run")
            # Disable the HTTP request deadline for the long command stream.
            # Both the server command deadline and this local timer retain the
            # judging budget, independently of the whole-batch lifetime reserve.
            async with asyncio.timeout(command):
                execution = await sandbox.commands.run(
                    judge_command,
                    user="root",
                    timeout=command,
                    request_timeout=0,
                    on_stdout=check_output,
                    on_stderr=check_output,
                )
            diagnostics.log(
                "stage_complete", exit_code=execution.exit_code, output_bytes=output_bytes
            )
            if execution.exit_code != 0:
                raise CommandExitException(
                    exit_code=execution.exit_code,
                    stdout=execution.stdout,
                    stderr=execution.stderr,
                    error=execution.error,
                )
            diagnostics.start_stage("parse")
            verdict = await asyncio.to_thread(_parse_result, execution.stdout, len(batch["cases"]))
            diagnostics.log("stage_complete")
            evaluation_complete = True
    except Exception as error:
        # Snapshot the outer timer before probes/cleanup. Its expiry is an
        # infrastructure failure, not a stage-local timeout or a model verdict.
        attempt_deadline_expired = batch_deadline is not None and batch_deadline.expired()
        retry_reason = (
            None if attempt_deadline_expired else _e2b_retry_reason(error, diagnostics.stage)
        )
        is_timeout = (
            not attempt_deadline_expired
            and diagnostics.stage == "run"
            and retry_reason is not None
            and isinstance(error, timeout_error_types + (httpx.TimeoutException,))
        )
        retryable_create = (
            not attempt_deadline_expired
            and diagnostics.stage == "create"
            and retry_reason is not None
        )
        retryable_upload = (
            diagnostics.stage == "upload"
            and batch_deadline is not None
            and not attempt_deadline_expired
            and retry_reason is not None
        )
        retryable_warmup = (
            not attempt_deadline_expired
            and diagnostics.stage == "warmup"
            and retry_reason is not None
        )
        retryable_run = (
            shared is not None
            and not attempt_deadline_expired
            and diagnostics.stage == "run"
            and retry_reason is not None
        )
        # Snapshot deadlines above before any diagnostic work. SDK command
        # output may be large; keep redaction off the rollout event loop.
        command_details = await asyncio.to_thread(_command_failure_details, error)
        diagnostics.log(
            "failure",
            (
                logging.WARNING
                if is_timeout
                or retryable_create
                or retryable_upload
                or retryable_warmup
                or retryable_run
                else logging.ERROR
            ),
            error_type=type(error).__name__,
            error=_error_message(error),
            attempt_deadline_expired=attempt_deadline_expired,
            retryable_create=retryable_create,
            retryable_timeout=is_timeout,
            retryable_upload=retryable_upload,
            retryable_warmup=retryable_warmup,
            retryable_run=retryable_run,
            retry_reason=retry_reason,
            **command_details,
        )
        if diagnostics.stage in ("warmup", "run") and sandbox is not None:
            await _probe_sandbox(sandbox, diagnostics, rpc)
        batch_error = CodeE2BError(
            diagnostics.stage,
            error,
            is_timeout=is_timeout,
            attempt_deadline_expired=attempt_deadline_expired,
            retryable_create=retryable_create,
            retryable_upload=retryable_upload,
            retryable_warmup=retryable_warmup,
            retryable_run=retryable_run,
            diagnostic_context={key: diagnostics.fields.get(key) for key in _ERROR_CONTEXT_FIELDS},
        )
        raise batch_error from None
    except asyncio.CancelledError:
        diagnostics.log("cancelled")
        raise
    finally:
        if sandbox is not None:
            diagnostics.start_stage("cleanup")
            warmup_state.abort_requested = True
            cleanup_cancelled = False
            try:
                try:
                    # Task termination and handle release share the existing
                    # three-second cleanup allowance; they do not extend judging.
                    async with asyncio.timeout(_WARMUP_CLEANUP_TIMEOUT_S):
                        if warmup_state.task is not None and not warmup_state.task.done():
                            warmup_state.task.cancel()
                            # wait() observes completion without re-raising a
                            # child's CancelledError as caller cancellation.
                            await asyncio.wait({warmup_state.task})
                        await _disconnect_warmup(warmup_state, diagnostics)
                except Exception as error:  # pylint: disable=broad-exception-caught
                    diagnostics.log(
                        "cleanup_failure",
                        logging.WARNING,
                        operation="warmup_task_cleanup",
                        error_type=type(error).__name__,
                        error=_error_message(error),
                    )
                except asyncio.CancelledError:
                    cleanup_cancelled = True
                    raise
            finally:
                if warmup_state.task is not None and not warmup_state.task.done():
                    # Do not stack another sandbox on top of an unresolved
                    # owned task. Still preserve external cancellation below.
                    if batch_error is not None:
                        batch_error.warmup_cleanup_incomplete = True
                    diagnostics.log("warmup_cleanup_incomplete", logging.WARNING)
                # Cleanup cannot invalidate a complete verdict or replace the
                # original failure. Retry termination once only after scoring.
                from e2b.exceptions import SandboxNotFoundException

                cleanup_attempts = 2 if evaluation_complete and not cleanup_cancelled else 1
                cleanup_confirmed = False
                for cleanup_attempt in range(1, cleanup_attempts + 1):
                    if cleanup_attempt > 1:
                        _raise_if_reward_stopping()
                    try:
                        async with asyncio.timeout(_SANDBOX_CLEANUP_TIMEOUT_S):
                            await sandbox.kill(request_timeout=min(rpc, _SANDBOX_CLEANUP_TIMEOUT_S))
                        cleanup_confirmed = True
                    except SandboxNotFoundException:
                        cleanup_confirmed = True
                    except Exception as error:  # pylint: disable=broad-exception-caught
                        diagnostics.log(
                            "cleanup_failure",
                            logging.WARNING,
                            operation="kill",
                            cleanup_attempt=cleanup_attempt,
                            cleanup_confirmed=False,
                            evaluation_complete=evaluation_complete,
                            error_type=type(error).__name__,
                            error=_error_message(error),
                        )
                        if shared is not None and batch_error is not None:
                            batch_error.sandbox_cleanup_incomplete = True
                    if cleanup_confirmed:
                        diagnostics.log(
                            "stage_complete",
                            operation="kill",
                            cleanup_attempt=cleanup_attempt,
                            cleanup_confirmed=True,
                        )
                        break
                if not cleanup_confirmed:
                    diagnostics.log(
                        "cleanup_unconfirmed",
                        logging.WARNING,
                        operation="kill",
                        cleanup_attempts=cleanup_attempts,
                        cleanup_confirmed=False,
                        evaluation_complete=evaluation_complete,
                        original_stage=batch_error.stage if batch_error is not None else None,
                        action=(
                            "return_verdict"
                            if evaluation_complete and not cleanup_cancelled
                            else "preserve_original_error"
                        ),
                    )
    diagnostics.log("batch_complete", correct=verdict["correct"], reward=verdict["reward"])
    return verdict


async def _run_batch_with_timeout_retry(
    batch: dict[str, Any],
    payload: str | bytes,
    *,
    context: Mapping[str, Any],
    shared: _VepfsRequest | None = None,
) -> tuple[dict[str, Any] | None, _BatchRetryCounts]:
    """Share independent stage budgets; each sandbox has a fresh attempt deadline."""
    counts = _BatchRetryCounts()
    # Setup stages and judging have independent recovery budgets. Never reset
    # counts when another stage requires a fresh sandbox.
    max_attempts = (
        1
        + (_MAX_CREATE_FAILURES - 1)
        + (_MAX_UPLOAD_FAILURES - 1)
        + (_MAX_WARMUP_FAILURES - 1)
        + (_MAX_RUN_TIMEOUTS - 1)
    )
    for attempt in range(1, max_attempts + 1):
        _raise_if_reward_stopping()
        try:
            verdict = await _run_batch(
                batch,
                payload,
                context={**context, "attempt": attempt},
                upload_retry_count=counts.upload_retries,
                **({"shared": shared} if shared is not None else {}),
            )
            if counts.create_failures:
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "create_recovered",
                    "create_failure_count": counts.create_failures,
                    "create_retry_count": counts.create_retries,
                }
                logger.info(
                    "Code E2B creation recovered: %s", fields, extra={"code_e2b_retry": fields}
                )
            if counts.upload_failures:
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "upload_recovered",
                    "upload_failure_count": counts.upload_failures,
                    "upload_retry_count": counts.upload_retries,
                }
                logger.info(
                    "Code E2B upload recovered: %s", fields, extra={"code_e2b_retry": fields}
                )
            if counts.warmup_failures:
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "warmup_recovered",
                    "warmup_failure_count": counts.warmup_failures,
                    "warmup_retry_count": counts.warmup_retries,
                }
                logger.info(
                    "Code E2B warmup recovered: %s", fields, extra={"code_e2b_retry": fields}
                )
            return verdict, counts
        except CodeE2BError as error:
            # Defend the policy here too: no stage retry or zero scoring may
            # consume an expired whole-attempt deadline, even with stale flags.
            if error.attempt_deadline_expired or error.warmup_cleanup_incomplete:
                raise
            _raise_if_reward_stopping()
            if shared is not None and (error.is_timeout or error.retryable_run):
                counts.run_failures += 1
                counts.run_timeouts += int(error.is_timeout)
                if counts.run_failures >= _MAX_RUN_TIMEOUTS:
                    # Match legacy timeout scoring without inventing case verdicts.
                    # Non-timeout failures still propagate after the shared budget.
                    if error.is_timeout:
                        return None, counts
                    raise
                counts.run_retries += 1
                delay = random.uniform(1.0, 3.0)
                logger.warning(
                    "Retrying whole-task evaluator after an incomplete command: %s; "
                    "run_failure_count=%s; run_retry_count=%s; retries_remaining=%s; "
                    "sandbox_cleanup_incomplete=%s; retry_delay_s=%s",
                    error,
                    counts.run_failures,
                    counts.run_retries,
                    _MAX_RUN_TIMEOUTS - counts.run_failures - 1,
                    error.sandbox_cleanup_incomplete,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if error.stage == "create" and error.retryable_create:
                counts.create_failures += 1
                exhausted = counts.create_failures >= _MAX_CREATE_FAILURES
                delay = 0.0 if exhausted else random.uniform(1.0, 3.0)
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "create_exhausted" if exhausted else "create_retry_start",
                    "error_type": error.error_type,
                    "error": str(error),
                    "create_failure_count": counts.create_failures,
                    "create_retry_count": counts.create_retries + int(not exhausted),
                    "retries_remaining": max(_MAX_CREATE_FAILURES - counts.create_failures - 1, 0),
                    "sandbox_cleanup_incomplete": error.sandbox_cleanup_incomplete,
                    "retry_delay_s": delay,
                }
                logger.log(
                    logging.ERROR if exhausted else logging.WARNING,
                    "Code E2B creation retry: %s",
                    fields,
                    extra={"code_e2b_retry": fields},
                )
                if exhausted:
                    raise
                counts.create_retries += 1
                # A failed create may not return an ID; the server-side
                # lifetime remains the fallback for any unknown instance.
                await asyncio.sleep(delay)
                continue
            if error.stage == "upload" and error.retryable_upload:
                counts.upload_failures += 1
                exhausted = counts.upload_failures >= _MAX_UPLOAD_FAILURES
                backoff = min(
                    _UPLOAD_RETRY_INITIAL_DELAY_S * 2 ** (counts.upload_failures - 1),
                    _UPLOAD_RETRY_MAX_DELAY_S,
                )
                delay = 0.0 if exhausted else random.uniform(backoff / 2, backoff)
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "upload_exhausted" if exhausted else "upload_retry_start",
                    "error_type": error.error_type,
                    "error": str(error),
                    "upload_failure_count": counts.upload_failures,
                    "upload_retry_count": counts.upload_retries + int(not exhausted),
                    "retries_remaining": max(_MAX_UPLOAD_FAILURES - counts.upload_failures - 1, 0),
                    "sandbox_cleanup_incomplete": error.sandbox_cleanup_incomplete,
                    "retry_delay_s": delay,
                }
                logger.log(
                    logging.ERROR if exhausted else logging.WARNING,
                    "Code E2B upload retry: %s",
                    fields,
                    extra={"code_e2b_retry": fields},
                )
                if exhausted:
                    raise
                counts.upload_retries += 1
                # Cleanup finishes before backing off; reuse the exact payload,
                # and let cancellation propagate without creating another sandbox.
                await asyncio.sleep(delay)
                continue
            if error.stage == "warmup" and error.retryable_warmup:
                counts.warmup_failures += 1
                exhausted = counts.warmup_failures >= _MAX_WARMUP_FAILURES
                delay = 0.0 if exhausted else random.uniform(1.0, 3.0)
                fields = {
                    **context,
                    "attempt": attempt,
                    "event": "warmup_exhausted" if exhausted else "warmup_retry_start",
                    "error_type": error.error_type,
                    "error": str(error),
                    "warmup_failure_count": counts.warmup_failures,
                    "warmup_retry_count": counts.warmup_retries + int(not exhausted),
                    "retries_remaining": max(_MAX_WARMUP_FAILURES - counts.warmup_failures - 1, 0),
                    "sandbox_cleanup_incomplete": error.sandbox_cleanup_incomplete,
                    "retry_delay_s": delay,
                }
                logger.log(
                    logging.ERROR if exhausted else logging.WARNING,
                    "Code E2B warmup retry: %s",
                    fields,
                    extra={"code_e2b_retry": fields},
                )
                if exhausted:
                    raise
                counts.warmup_retries += 1
                # _run_batch has already performed bounded cleanup. Cancellation
                # during this asynchronous wait propagates without another create.
                await asyncio.sleep(delay)
                continue
            if error.stage != "run" or not error.is_timeout:
                raise
            counts.run_timeouts += 1
            exhausted = counts.run_timeouts >= _MAX_RUN_TIMEOUTS
            if not exhausted:
                counts.run_retries += 1
            fields = {
                **context,
                "attempt": attempt,
                "event": (
                    "timeout_exhausted"
                    if counts.run_timeouts >= _MAX_RUN_TIMEOUTS
                    else "retry_start"
                ),
                "error_type": error.error_type,
                "error": str(error),
                "timeout_count": counts.run_timeouts,
                "retry_count": counts.run_retries,
            }
            logger.warning(
                "Code E2B timeout: %s",
                "; ".join(f"{key}={value!r}" for key, value in fields.items()),
                extra={"code_e2b_retry": fields},
            )
            if counts.run_timeouts >= _MAX_RUN_TIMEOUTS:
                return None, counts
    raise AssertionError("unreachable retry budget")


async def code_e2b_reward(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return 1 only if every prepared case passes, otherwise return 0."""
    try:
        request = _build_request(
            task, result["response"], finish_reason=result.get("finish_reason")
        )
    except _EmptyCodeError:
        return {"status": "ok", "reward": 0.0, "correct": False, "reason": "no_code"}
    shared = _prepare_vepfs_request(task, request)

    async def evaluation_requests():
        if shared is not None:
            yield request, shared.payload, len(shared.payload)
        else:
            async for prepared in _iter_request_batches_async(request):
                yield prepared

    scores = []
    tests = []
    accepted = True
    completed_batches = 0
    completed_cases = 0
    timeout_count = retry_count = timeout_batches = unevaluated_cases = 0
    create_failure_count = create_retry_count = 0
    upload_failure_count = upload_retry_count = 0
    warmup_failure_count = warmup_retry_count = 0
    run_failure_count = run_retry_count = 0
    shared_profile = None
    batch_index = 0
    async for batch, payload, payload_bytes in evaluation_requests():
        batch_index += 1
        verdict, batch_retries = await _run_batch_with_timeout_retry(
            batch,
            payload,
            context={
                "batch_index": batch_index,
                "task_id": task.get("task_id"),
                "group_id": task.get("group_id"),
                "sample_id": task.get("sample_id"),
                "finish_reason": result.get("finish_reason"),
                "payload_bytes": payload_bytes,
            },
            **({"shared": shared} if shared is not None else {}),
        )
        count = len(batch["cases"])
        timeout_count += batch_retries.run_timeouts
        retry_count += batch_retries.run_retries
        create_failure_count += batch_retries.create_failures
        create_retry_count += batch_retries.create_retries
        upload_failure_count += batch_retries.upload_failures
        upload_retry_count += batch_retries.upload_retries
        warmup_failure_count += batch_retries.warmup_failures
        warmup_retry_count += batch_retries.warmup_retries
        run_failure_count += batch_retries.run_failures
        run_retry_count += batch_retries.run_retries
        if verdict is None:
            accepted = False
            timeout_batches += 1
            unevaluated_cases += count
            continue
        if shared is not None:
            shared_profile = verdict["sandbox_result"].get("_profile")
        scores.append(verdict["sandbox_result"]["score"] * count)
        tests.extend(verdict["sandbox_result"]["tests"])
        accepted = accepted and verdict["correct"]
        completed_batches += 1
        completed_cases += count
    total_cases = len(request["cases"])
    if completed_cases + unevaluated_cases != total_cases:
        raise ValueError("not every requested case was evaluated or marked as timed out")
    # A batch timeout supplies no per-case verdict. Keep real test results and
    # distinguish missing results from known failures instead of inventing scores.
    score = None if timeout_batches else math.fsum(scores) / completed_cases
    metrics = {
        "passed_cases": float(sum(case["passed"] for case in tests)),
        "total_cases": float(total_cases),
        "completed_batches": float(completed_batches),
    }
    if timeout_count:
        metrics.update(
            timeout_count=float(timeout_count),
            retry_count=float(retry_count),
            timeout_batches=float(timeout_batches),
            evaluated_cases=float(completed_cases),
            unevaluated_cases=float(unevaluated_cases),
        )
    if create_failure_count:
        metrics.update(
            create_failure_count=float(create_failure_count),
            create_retry_count=float(create_retry_count),
        )
    if upload_failure_count:
        metrics.update(
            upload_failure_count=float(upload_failure_count),
            upload_retry_count=float(upload_retry_count),
        )
    if warmup_failure_count:
        metrics.update(
            warmup_failure_count=float(warmup_failure_count),
            warmup_retry_count=float(warmup_retry_count),
        )
    if run_failure_count:
        metrics.update(
            run_failure_count=float(run_failure_count),
            run_retry_count=float(run_retry_count),
            retry_count=float(retry_count),
        )
    reward_result: dict[str, Any] = {
        "status": "ok",
        "reward": float(accepted),
        "correct": accepted,
        "reason": (
            "timeout_after_retry" if timeout_batches else "accepted" if accepted else "failed_tests"
        ),
        "sandbox_result": {"accepted": accepted, "score": score, "tests": tests},
        "metrics": metrics,
    }
    if shared_profile is not None:
        reward_result["sandbox_result"]["_profile"] = shared_profile
    if timeout_count:
        reward_result["timed_out"] = bool(timeout_batches)
        fields = {
            "event": "evaluation_complete",
            "task_id": task.get("task_id"),
            "group_id": task.get("group_id"),
            "sample_id": task.get("sample_id"),
            "finish_reason": result.get("finish_reason"),
            "reward": reward_result["reward"],
            "reason": reward_result["reason"],
            "timed_out": bool(timeout_batches),
            **metrics,
        }
        logger.info(
            "Code E2B timeout summary: %s",
            "; ".join(f"{key}={value!r}" for key, value in fields.items()),
            extra={"code_e2b_retry": fields},
        )
    return reward_result
