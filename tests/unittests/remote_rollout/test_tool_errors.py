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

"""Tool failures are classified conservatively and counted once per receipt."""

import json

import pytest

from northrl.remote_rollout.metrics import RolloutMetrics
from northrl.remote_rollout.tool_errors import ENV_TOOL_ERROR_CATEGORIES, TOOL_ERROR_CATEGORIES


def _history(message_format, tool_name, command, result, tool_id="call1"):
    if message_format == "openai":
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"command": command}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_id, **result},
        ]
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {"command": command},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, **result}],
        },
    ]


def _assert_categories(values, expected):
    counts = {category: values[f"tool_error_{category}"] for category in TOOL_ERROR_CATEGORIES}
    assert {category: count for category, count in counts.items() if count} == expected
    assert sum(counts.values()) == values["tool_result_errors"]


@pytest.mark.parametrize("message_format", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "tool_name,content,category",
    [
        ("Bash", "Exit code 127\n/bin/bash: line 1: cargo: command not found", "command_not_found"),
        ("Bash", "ModuleNotFoundError: No module named 'pytest'", "dependency_missing"),
        ("Bash", "curl: (6) Could not resolve host: example.invalid", "network"),
        ("Bash", "PermissionError: [Errno 13] Permission denied", "permission"),
        ("Bash", "OSError: [Errno 28] No space left on device", "resource"),
        ("Read", "Error: File does not exist.", "file_not_found"),
        ("Edit", "No changes to make: old_string and new_string are the same.", "edit"),
        ("Bash", "error[E0308]: mismatched types", "test_or_compile"),
        ("CustomTool", "Something unusual happened", "other"),
    ],
)
def test_error_categories_cover_both_provider_formats(message_format, tool_name, content, category):
    metrics = RolloutMetrics()
    history = _history(
        message_format, tool_name, "cargo test", {"is_error": True, "content": content}
    )
    metrics.observe_messages(message_format, history)
    metrics.observe_messages(message_format, history)
    metrics.observe_messages(message_format, history[-1:])
    values = metrics.snapshot()
    assert values["tool_results_observed"] == 1
    _assert_categories(values, {category: 1})
    assert (category in ENV_TOOL_ERROR_CATEGORIES) == (category in TOOL_ERROR_CATEGORIES[:5])


@pytest.mark.parametrize("message_format", ["openai", "anthropic"])
def test_cargo_nonzero_exit_header_is_detected_after_history_compaction(message_format):
    metrics = RolloutMetrics()
    history = _history(
        message_format,
        "Bash",
        "cargo test -p cxx --test test test_c_return -- --nocapture",
        {"content": [{"type": "text", "text": "Exit code 127\ncargo: command not found"}]},
    )
    metrics.observe_messages(message_format, history[:1])
    metrics.observe_messages(message_format, [])
    metrics.observe_messages(message_format, history[1:])
    values = metrics.snapshot()
    assert values["tool_results_observed"] == 1
    _assert_categories(values, {"command_not_found": 1})


@pytest.mark.parametrize(
    "content",
    [
        "Example: ModuleNotFoundError: No module named 'pytest'",
        "Exit code 127\ncargo: command not found",
        '{"exit_code": 127, "stderr": "cargo: command not found"}',
        [{"type": "text", "text": "Exit code 127\ncargo: command not found"}],
    ],
)
def test_successful_read_contents_are_not_treated_as_failures(content):
    metrics = RolloutMetrics()
    metrics.observe_messages("anthropic", _history("anthropic", "Read", "", {"content": content}))
    assert metrics.snapshot()["tool_results_observed"] == 1
    _assert_categories(metrics.snapshot(), {})


@pytest.mark.parametrize(
    "result",
    [
        {"exit_code": 127, "content": "cargo: command not found"},
        {"content": {"exit_code": 127, "stderr": "cargo: command not found"}},
        {"content": '{"exit_code": 127, "stderr": "cargo: command not found"}'},
    ],
)
def test_structured_nonzero_status_is_detected_without_error_flag(result):
    metrics = RolloutMetrics()
    metrics.observe_messages("openai", _history("openai", "Bash", "cargo test", result))
    _assert_categories(metrics.snapshot(), {"command_not_found": 1})


@pytest.mark.parametrize("exit_code", [None, True, False, "1", 1.5, float("nan"), {}, []])
def test_malformed_exit_status_is_not_an_error_signal(exit_code):
    metrics = RolloutMetrics()
    metrics.observe_messages(
        "openai",
        _history(
            "openai",
            "Bash",
            "echo example",
            {"exit_code": exit_code, "content": "Permission denied"},
        ),
    )
    _assert_categories(metrics.snapshot(), {})


def test_zero_status_and_embedded_error_examples_do_not_inflate_counts():
    metrics = RolloutMetrics()
    metrics.observe_messages(
        "openai",
        _history(
            "openai",
            "Bash",
            "cat README.md",
            {
                "exit_code": 0,
                "content": "Exit code 0\nExample:\nExit code 127\ncargo: command not found",
            },
        ),
    )
    _assert_categories(metrics.snapshot(), {})


@pytest.mark.parametrize(
    "command,content,category",
    [
        ("cd /repo && cargo test", "Exit code 1", "test_or_compile"),
        (
            "cargo test",
            "Exit code 1\nModuleNotFoundError: No module named 'pytest'\nFAILED test_a",
            "dependency_missing",
        ),
        ("cargo test", "Exit code 127\ncargo: command not found", "command_not_found"),
        ("grep missing README.md", "Exit code 1", "other"),
        ("echo pytest", "Exit code 1", "other"),
    ],
)
def test_specific_environment_evidence_takes_precedence_over_test_command(
    command, content, category
):
    metrics = RolloutMetrics()
    metrics.observe_messages("openai", _history("openai", "Bash", command, {"content": content}))
    _assert_categories(metrics.snapshot(), {category: 1})


def test_later_error_flag_upgrades_once_without_migrating_existing_category():
    metrics = RolloutMetrics()
    history = [{"role": "tool", "tool_call_id": "call1", "content": "unknown output"}]
    metrics.observe_messages("openai", history)
    assert metrics.snapshot()["tool_result_errors"] == 0
    history[0]["is_error"] = True
    metrics.observe_messages("openai", history)
    _assert_categories(metrics.snapshot(), {"other": 1})
    history[0]["content"] = "cargo: command not found"
    metrics.observe_messages("openai", history)
    values = metrics.snapshot()
    assert values["tool_results_observed"] == 1
    _assert_categories(values, {"other": 1})


@pytest.mark.parametrize("message_format", ["openai", "anthropic"])
@pytest.mark.parametrize("tool_id", [None, "", " ", 123, {}])
def test_missing_or_invalid_ids_cannot_be_counted_reliably(message_format, tool_id):
    metrics = RolloutMetrics()
    metrics.observe_messages(
        message_format,
        _history(message_format, "Bash", "cargo test", {"is_error": True}, tool_id=tool_id),
    )
    assert metrics.snapshot()["tool_results_observed"] == 0
    _assert_categories(metrics.snapshot(), {})


def test_malformed_messages_and_arguments_do_not_break_observation():
    metrics = RolloutMetrics()
    metrics.observe_messages(
        "openai",
        [
            None,
            1,
            "text",
            {"role": "assistant", "tool_calls": [None, {}, {"id": "bad", "function": "broken"}]},
        ],
    )
    history = _history(
        "openai", "Bash", "cargo test", {"content": "Exit code 127\ncargo: command not found"}
    )
    history[0]["tool_calls"][0]["function"]["arguments"] = "{broken"
    metrics.observe_messages("openai", history)
    metrics.observe_messages(
        "anthropic",
        [
            {
                "role": "user",
                "content": [None, 1, {"type": "text", "text": "cargo: command not found"}],
            }
        ],
    )
    assert metrics.snapshot()["tool_results_observed"] == 1
    _assert_categories(metrics.snapshot(), {"command_not_found": 1})
