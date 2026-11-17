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

"""Score VerInstruct with sandbox hard checks followed by an LLM soft check."""

from collections.abc import Mapping
from typing import Any

from .e2b_checker import run_e2b_checker
from .recast import recast_reward

_EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")


def _answer_text(response: str) -> str:
    """Match bp's answer extraction, scanning trailing tokens in linear time."""
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


def _hard_harness(instruction: str, response: str, funcs: list[str]) -> str:
    """Build source for the sandbox; dataset code never executes in NorthRL."""
    # Match bp's resource guards. Isolation must be provided by the sandbox service.
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
    return (
        guard
        + f"_INSTR = {instruction!r}\n"
        + f"_RESP = {response!r}\n"
        + f"_FUNCS = {funcs!r}\n"
        + """\
_ok = True
for _code in _FUNCS:
    try:
        _ns = {}
        exec(_code, _ns)
        if not bool(_ns['check_following'](_INSTR, _RESP)):
            _ok = False
            break
    except Exception:
        _ok = False
        break
print(_ok)
"""
    )


async def _hard_ok(
    instruction: str,
    response: str,
    funcs: list[str],
    *,
    metrics: dict[str, float] | None = None,
) -> bool | None:
    """Return hard-check correctness, or None after three run-only timeouts."""
    result = await run_e2b_checker(
        _hard_harness(instruction, response, funcs),
        "/tmp/verinstruct_check.py",
        "VerInstruct",
        metrics=metrics,
    )
    if result is None:
        return None
    return result.exit_code == 0 and result.stdout.strip() == "True"


async def verinstruct_reward(task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return hard-then-soft judgment; give zero after repeated sandbox timeouts."""
    response = result["response"]
    if not isinstance(response, str):
        raise TypeError("VerInstruct response must be a string")
    response = _answer_text(response)
    metadata = task.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("VerInstruct metadata must be a dictionary")
    funcs = metadata.get("hard_functions")
    if funcs is not None and hasattr(funcs, "tolist"):
        funcs = funcs.tolist()
    if funcs is None:
        funcs = []
    if not isinstance(funcs, list) or any(not isinstance(code, str) for code in funcs):
        raise TypeError("VerInstruct hard_functions must be a list of Python source strings")
    metrics: dict[str, float] = {}
    if funcs:
        prompt = task["prompt"]
        instruction = str(
            metadata.get("instruction")
            or metadata.get("prompt_text")
            or (prompt if isinstance(prompt, str) else "")
        )
        hard_ok = await _hard_ok(instruction, response, funcs, metrics=metrics)
        if hard_ok is None:
            return {
                "reward": 0.0,
                "correct": False,
                "reason": "timeout_after_retry",
                "metrics": metrics,
            }
        if not hard_ok:
            score: dict[str, Any] = {"reward": 0.0, "correct": False, "reason": "fail_hard"}
            if metrics:
                score["metrics"] = metrics
            return score
    score = await recast_reward(task, {**result, "response": response})
    if score["reason"] == "fail_llm":
        score = {**score, "reason": "fail_soft"}
    if metrics:
        score = {**score, "metrics": {**score.get("metrics", {}), **metrics}}
    return score
