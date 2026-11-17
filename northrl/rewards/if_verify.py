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

"""Run a sample-provided ``evaluate(response)`` checker in an E2B sandbox."""

import json
from collections.abc import Mapping
from typing import Any

from .e2b_checker import run_e2b_checker

_EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")
_VERDICT_PREFIX = "__NORTHRL_IF_VERIFY_VERDICT__"
_ERROR_PREFIX = "__NORTHRL_IF_VERIFY_ERROR__"


def _answer_text(response: str) -> str:
    """Match bp's answer-only extraction without changing tokens in the body."""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]
    elif "###Response" in response:
        response = response.rsplit("###Response", 1)[-1]

    end = len(response)
    while end:
        if response[end - 1].isspace():
            end -= 1
            continue
        for token in _EOS_TOKENS:
            start = end - len(token)
            if start >= 0 and response.startswith(token, start, end):
                end = start
                break
        else:
            break
    return response[:end].lstrip()


def _checker_harness(checker: str, response: str) -> str:
    """Build sandbox source; neither checker nor response executes in NorthRL."""
    guard = """\
try:
    import resource as _rl
    for _name, _value in (("RLIMIT_AS", 1073741824), ("RLIMIT_CPU", 10),
                          ("RLIMIT_FSIZE", 67108864), ("RLIMIT_NPROC", 64)):
        try:
            _limit = getattr(_rl, _name)
            _rl.setrlimit(_limit, (_value, _value))
        except Exception:
            pass
except Exception:
    pass
try:
    import socket as _sock
    def _no_net(*args, **kwargs):
        raise OSError("network disabled in judge sandbox")
    _sock.socket = _no_net
    _sock.create_connection = _no_net
except Exception:
    pass
"""
    serialized_response = json.dumps(response, ensure_ascii=False)
    return (
        guard
        + "import json\n\n"
        + checker.rstrip()
        + "\n\n"
        + f"_ANSWER = json.loads({serialized_response!r})\n"
        + f"_VERDICT_PREFIX = {_VERDICT_PREFIX!r}\n"
        + f"_ERROR_PREFIX = {_ERROR_PREFIX!r}\n"
        + """\
try:
    _value = evaluate(_ANSWER)
except Exception as _error:
    print(_ERROR_PREFIX, repr(_error))
else:
    print(_VERDICT_PREFIX, "1" if _value else "0")
"""
    )


def _parse_verdict(stdout: str) -> str:
    """Return pass/fail/error from the harness's final non-empty output line."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return "checker_error"
    final_line = lines[-1]
    if final_line == f"{_VERDICT_PREFIX} 1":
        return "pass"
    if final_line == f"{_VERDICT_PREFIX} 0":
        return "fail_checker"
    return "checker_error"


async def _run_checker(
    checker: str, response: str, *, metrics: dict[str, float] | None = None
) -> str:
    """Run the checker, distinguishing three run-only timeouts from infrastructure errors."""
    result = await run_e2b_checker(
        _checker_harness(checker, response),
        "/tmp/if_verify_check.py",
        "IfVerify",
        metrics=metrics,
    )
    if result is None:
        return "timeout_after_retry"
    if result.exit_code != 0:
        return "checker_error"
    return _parse_verdict(result.stdout)


def _reward_result(reason: str, metrics: dict[str, float] | None = None) -> dict[str, Any]:
    correct = reason == "pass"
    score: dict[str, Any] = {
        "reward": float(correct),
        "correct": correct,
        "reason": reason,
    }
    if metrics:
        score["metrics"] = metrics
    return score


async def if_verify_reward(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return one iff the row's sandboxed ``evaluate(response)`` is truthy."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("IfVerify response must be a string")

    metadata = task["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("IfVerify metadata must be a mapping")

    checker = metadata.get("checker")
    if checker is None or checker == "":
        return _reward_result("no_checker")
    if not isinstance(checker, str):
        raise TypeError("IfVerify metadata.checker must be a string")
    if not checker.strip():
        return _reward_result("no_checker")

    metrics: dict[str, float] = {}
    reason = await _run_checker(checker, _answer_text(response), metrics=metrics)
    return _reward_result(reason, metrics)
