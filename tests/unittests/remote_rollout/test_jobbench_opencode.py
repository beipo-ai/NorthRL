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

"""Check the V2 configuration and CLI protocol used by the policy agent."""

import json
from pathlib import Path

import pytest

from northrl.rollout_entrypoints.jobbench import opencode


def _stream(*events):
    return "\n".join(json.dumps(event) for event in events)


def test_configuration_uses_relay_credentials_and_disables_unattended_interactions():
    config = opencode.configuration(17)
    provider = config["providers"]["localrelay"]
    assert provider["env"] == ["OPENAI_API_KEY"]
    assert provider["settings"]["baseURL"] == "{env:OPENAI_BASE_URL}"
    assert provider["package"] == "@opencode/ai/providers/openai-compatible"
    assert "northrl" in provider["models"]
    assert config["agents"]["build"]["steps"] == 17
    assert config["websearch"] is False
    assert config["update"] == "disable"
    rules = config["agents"]["build"]["permissions"]
    for action in ("shell", "read", "edit", "glob", "grep", "external_directory"):
        matching = [rule for rule in rules if rule["action"] in ("*", action)]
        assert matching[-1]["effect"] == "allow"
    for action in ("subagent", "question", "skill", "websearch", "webfetch"):
        matching = [rule for rule in rules if rule["action"] in ("*", action)]
        assert matching[-1]["effect"] == "deny"
    assert "ANTHROPIC" not in json.dumps(config)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "10", None])
def test_configuration_rejects_invalid_step_limits(limit):
    with pytest.raises(ValueError, match="positive integer"):
        opencode.configuration(limit)


def test_command_preserves_argv_boundaries_for_binary_and_prompt():
    binary = Path("/opt/jobbench runtime/opencode")
    argv = opencode.command(binary, "jobbench-agent")
    assert argv[:6] == [
        "runuser",
        "--user",
        "jobbench-agent",
        "--preserve-environment",
        "--",
        str(binary),
    ]
    assert argv[6:] == [
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
    prompt = '--model other/provider; echo "do not execute as shell"'
    assert [*argv, prompt][-2:] == ["--", prompt]


def test_final_answer_groups_chunks_and_ignores_delayed_previous_message_text():
    stdout = _stream(
        {"type": "step_start", "part": {"messageID": "first"}},
        {"type": "text", "part": {"messageID": "first", "text": "Working. "}},
        {"type": "step_finish", "part": {"messageID": "first"}},
        {"type": "step_start", "part": {"messageID": "last"}},
        {"type": "text", "part": {"messageID": "last", "text": "  Saved "}},
        {"type": "text", "part": {"messageID": "last", "text": "the report.\n"}},
        {"type": "step_finish", "part": {"messageID": "last"}},
        {"type": "text", "part": {"messageID": "first", "text": "Old reconciled text."}},
    )
    assert opencode.final_answer(stdout) == "Saved the report."
    assert not opencode.has_error(stdout)


def test_final_answer_tolerates_malformed_output_and_non_text_events():
    stdout = "not JSON\n{incomplete\n" + _stream(
        [],
        None,
        42,
        {"type": "text", "part": None},
        {"type": "text", "part": {"messageID": [], "text": "invalid"}},
        {"type": "text", "part": {"text": "no message ID"}},
        {"type": "text", "part": {"messageID": "last", "text": None}},
        {"type": "reasoning", "part": {"messageID": "last", "text": "private reasoning"}},
        {"type": "tool_use", "part": {"state": {"status": "error"}}},
        {"type": "result", "result": "Claude result is a different protocol"},
    )
    assert opencode.final_answer(stdout) is None
    assert not opencode.has_error(stdout)


def test_fatal_error_is_separate_from_partial_text():
    stdout = _stream(
        {"type": "text", "part": {"messageID": "last", "text": "Partial answer"}},
        {"type": "error", "error": {"message": "The provider disconnected"}},
    )
    assert opencode.final_answer(stdout) == "Partial answer"
    assert opencode.has_error(stdout)


def test_final_tool_only_step_does_not_report_earlier_commentary_as_final_answer():
    stdout = _stream(
        {"type": "text", "part": {"messageID": "first", "text": "I will inspect it."}},
        {"type": "step_start", "part": {"messageID": "last"}},
        {"type": "step_finish", "part": {"messageID": "last"}},
    )
    assert opencode.final_answer(stdout) is None
