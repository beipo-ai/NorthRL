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

"""JobBench adapter for a preinstalled OpenCode V2 2.2.0 binary.

The caller owns binary validation, the root-managed configuration file, process
isolation, and relay credentials. This module never installs or starts software.
"""

import json
from collections.abc import Iterator
from pathlib import Path


def configuration(max_turns: int) -> dict:
    """Build V2 configuration for /etc/opencode/opencode.json.

    V2 calls the model-step limit ``steps`` and reserves the final step for a
    summary without tools. It is not identical to Claude Code's turn limit.
    """
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("JobBench max_turns must be a positive integer")
    return {
        "$schema": "https://opencode.ai/config.json",
        "update": "disable",
        "providers": {
            "localrelay": {
                "name": "NorthRL relay",
                "env": ["OPENAI_API_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": "{env:OPENAI_BASE_URL}"},
                "models": {"northrl": {"name": "NorthRL"}},
            }
        },
        "agents": {
            "build": {
                "steps": max_turns,
                "permissions": [
                    {"action": "*", "resource": "*", "effect": "deny"},
                    *(
                        {"action": action, "resource": "*", "effect": "allow"}
                        for action in (
                            "shell",
                            "read",
                            "edit",
                            "glob",
                            "grep",
                            "external_directory",
                        )
                    ),
                ],
            }
        },
        # Serper is provided separately through the JobBench shell helper.
        # Disabling native search also avoids an interactive provider selection.
        "websearch": False,
    }


def command(binary: Path, user: str) -> list[str]:
    """Return isolated CLI arguments; the caller appends the prompt afterward."""
    return [
        "runuser",
        "--user",
        user,
        "--preserve-environment",
        "--",
        str(binary),
        "run",
        "--standalone",
        "--auto",
        "--format",
        "json",
        "--agent",
        "build",
        "--model",
        "localrelay/northrl",
        "--",
    ]


def _events(stdout: str) -> Iterator[dict]:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def final_answer(stdout: str) -> str | None:
    """Collect text from the last assistant message, including reconciled chunks.

    OpenCode emits ``text`` events with ``part.messageID`` and ``part.text``;
    it has no Claude-style ``result`` event. Step events establish message order
    before delayed text from an earlier message is reconciled into the stream.
    """
    messages: dict[str, list[str]] = {}
    for event in _events(stdout):
        if event.get("type") not in ("step_start", "step_finish", "text"):
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        message_id = part.get("messageID")
        if not isinstance(message_id, str) or not message_id:
            continue
        chunks = messages.setdefault(message_id, [])
        if event["type"] == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    if not messages:
        return None
    answer = "".join(messages[next(reversed(messages))]).strip()
    return answer or None


def has_error(stdout: str) -> bool:
    """Identify an explicit CLI failure without treating tool failures as fatal."""
    return any(event.get("type") == "error" for event in _events(stdout))
