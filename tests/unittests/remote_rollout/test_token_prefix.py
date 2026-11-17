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

"""Exact native message equality and unambiguous original-token selection."""

import json
from copy import deepcopy

import pytest

from northrl.remote_rollout.token_prefix import TokenPrefixCache, request_context


def _history():
    return [
        {"role": "user", "content": "List files."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Inspect the directory.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"ls","timeout":30}'},
                }
            ],
            "vendor": {"enabled": True, "limit": 1, "parts": [1, 2]},
        },
    ]


def _next(history):
    return [*deepcopy(history), {"role": "tool", "tool_call_id": "call-1", "content": "main.py"}]


def _payload():
    return {
        "model": "fixed-sampler",
        "messages": _history(),
        "tools": [
            {"type": "function", "function": {"name": "shell", "description": "Run commands"}}
        ],
        "sampling_params": {"max_tokens": 128, "temperature": 0.5},
    }


def test_equal_objects_match_without_changing_messages_or_counting_new_content():
    cache = TokenPrefixCache()
    history = _history()
    context = request_context(_payload(), api="openai")
    cache.remember(history, [10, 20], context)
    reordered = json.loads(json.dumps(history, sort_keys=True))
    cache.remember(reordered, [10, 20], context)
    current = _next(reordered)
    current[-1]["content"] = "Any new content."
    assert cache.match(current, context) == [10, 20]
    assert history == _history()


@pytest.mark.parametrize(
    "path,value",
    [
        ((0, "content"), "List files. "),
        ((0, "content"), [{"type": "text", "text": "List files."}]),
        ((1, "role"), "user"),
        ((1, "content"), ""),
        ((1, "reasoning_content"), "Changed reasoning."),
        ((1, "tool_calls", 0, "id"), "call-2"),
        ((1, "tool_calls", 0, "function", "name"), "read"),
        ((1, "tool_calls", 0, "function", "arguments"), '{"timeout":30,"command":"ls"}'),
        ((1, "tool_calls", 0, "function", "arguments"), '{"command": "ls", "timeout": 30}'),
        ((1, "vendor", "enabled"), 1),
        ((1, "vendor", "limit"), 1.0),
        ((1, "vendor", "parts"), [2, 1]),
        ((1, "new_field"), None),
    ],
)
def test_changed_historical_fields_prevent_reuse(path, value):
    cache = TokenPrefixCache()
    context = request_context(_payload(), api="openai")
    cache.remember(_history(), [10, 20], context)
    current = _next(_history())
    target = current
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert cache.match(current, context) is None


def test_history_must_be_a_complete_contiguous_prefix_with_new_messages():
    history = _history()
    context = request_context(_payload(), api="openai")
    cache = TokenPrefixCache()
    cache.remember(history, [10, 20], context)
    assert cache.match(history, context) is None
    assert cache.match(history[:1], context) is None
    assert cache.match(_next(history)[1:], context) is None
    assert cache.match([{"role": "system", "content": "new"}, *_next(history)], context) is None
    current = _next(history)
    current[1].pop("content")
    assert cache.match(current, context) is None  # Missing differs from explicit null.


def test_longest_branch_is_selected_but_ambiguous_history_never_falls_back():
    history = _history()
    longer = [*_next(history), {"role": "assistant", "content": "Done."}]
    sibling = [*_next(history), {"role": "assistant", "content": "Another branch."}]
    context = request_context(_payload(), api="openai")
    cache = TokenPrefixCache()
    cache.remember(history, [10, 20], context)
    cache.remember(longer, [10, 20, 30, 40], context)
    cache.remember(sibling, [10, 20, 30, 50], context)
    assert cache.match(_next(longer), context) == [10, 20, 30, 40]
    assert cache.match(_next(sibling), context) == [10, 20, 30, 50]
    cache.remember(longer, [10, 20, 30, 41], context)
    cache.remember(longer, [10, 20, 30, 40], context)
    assert cache.match(_next(longer), context) is None
    assert cache.match(_next(history), context) == [10, 20]


def test_cache_owns_snapshots_and_clear_discards_them():
    cache = TokenPrefixCache()
    context = request_context(_payload(), api="openai")
    history, tokens = _history(), [10, 20]
    cache.remember(history, tokens, context)
    history[0]["content"] = "changed"
    tokens.append(30)
    matched = cache.match(_next(_history()), context)
    assert matched == [10, 20]
    matched.append(40)
    assert cache.match(_next(_history()), context) == [10, 20]
    cache.clear()
    assert cache.match(_next(_history()), context) is None


@pytest.mark.parametrize(
    "update",
    [
        {"tools": []},
        {"tool_choice": "auto"},
        {"response_format": {"type": "json_object"}},
        {"reasoning_effort": "high"},
        {"sampling_params": {"max_tokens": 128, "vendor": True}},
        {"weaver_options": {"vendor": True}},
    ],
)
def test_request_settings_keep_histories_separate(update):
    context = request_context(_payload(), api="openai")
    changed = request_context({**_payload(), **update}, api="openai")
    cache = TokenPrefixCache()
    cache.remember(_history(), [10, 20], context)
    assert cache.match(_next(_history()), changed) is None
    cache.remember(_history(), [30, 40], changed)
    assert cache.match(_next(_history()), context) == [10, 20]
    assert cache.match(_next(_history()), changed) == [30, 40]


def test_only_transport_messages_and_named_sampling_fields_are_excluded_from_context():
    payload = _payload()
    original = deepcopy(payload)
    changed = deepcopy(payload)
    changed.update(messages=_next(_history()), stream=True, stream_options={"include_usage": True})
    changed["history_tokens"] = [10, 20]
    for name in ("temperature", "top_p", "top_k", "seed"):
        changed[name] = 7
        changed["sampling_params"][name] = 7
    assert request_context(changed, api="openai") == request_context(payload, api="openai")
    assert payload == original


def _anthropic_payload():
    return {
        "model": "fixed-sampler",
        "system": [{"type": "text", "text": "Use the shell."}],
        "messages": [
            {"role": "user", "content": "List files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect files.", "signature": "signed"},
                    {"type": "tool_use", "id": "tool-1", "name": "shell", "input": {"cmd": "ls"}},
                ],
            },
        ],
        "tools": [{"name": "shell", "input_schema": {"type": "object"}}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }


def _anthropic_next(payload):
    return [
        *deepcopy(payload["messages"]),
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "main.py"}],
        },
    ]


def test_protocols_keep_identical_histories_separate():
    history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    payload = {"model": "fixed-sampler", "messages": history}
    following = [*history, {"role": "user", "content": "Continue"}]
    openai_context = request_context(payload, api="openai")
    anthropic_context = request_context(payload, api="anthropic")
    cache = TokenPrefixCache()
    cache.remember(history, [10, 20], openai_context)
    assert cache.match(following, anthropic_context) is None
    cache.remember(history, [30, 40], anthropic_context)
    assert cache.match(following, openai_context) == [10, 20]
    assert cache.match(following, anthropic_context) == [30, 40]


@pytest.mark.parametrize(
    "update",
    [
        {"system": "Use another shell."},
        {"tools": []},
        {"tool_choice": {"type": "any"}},
        {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        {"output_config": {"effort": "high"}},
    ],
)
def test_anthropic_settings_isolate_history_without_counting_system(update):
    payload = _anthropic_payload()
    context = request_context(payload, api="anthropic")
    cache = TokenPrefixCache()
    cache.remember(payload["messages"], [10, 20], context)
    following = _anthropic_next(payload)
    assert cache.match(following, context) == [10, 20]
    changed = request_context({**payload, **update}, api="anthropic")
    assert cache.match(following, changed) is None


@pytest.mark.parametrize("header", ["anthropic-version", "anthropic-beta"])
@pytest.mark.parametrize("value", [None, "", "changed", "Feature-A,feature-b"])
def test_anthropic_protocol_header_changes_prevent_reuse(header, value):
    payload = _anthropic_payload()
    headers = {header: "feature-a,feature-b"}
    context = request_context(payload, api="anthropic", headers=headers)
    cache = TokenPrefixCache()
    cache.remember(payload["messages"], [10, 20], context)
    changed_headers = {} if value is None else {header: value}
    changed = request_context(payload, api="anthropic", headers=changed_headers)
    assert cache.match(_anthropic_next(payload), changed) is None


def test_anthropic_header_names_ignore_case_and_request_metadata():
    payload = _anthropic_payload()
    headers = {
        "Anthropic-Version": "2023-06-01",
        "Anthropic-Beta": "feature-a,feature-b",
        "X-Request-Id": "first-request",
        "Authorization": "Bearer first-key",
    }
    changed = {
        "anthropic-version": "2023-06-01",
        "ANTHROPIC-BETA": "feature-a,feature-b",
        "X-Request-Id": "second-request",
        "Authorization": "Bearer second-key",
    }
    assert request_context(payload, api="anthropic", headers=headers) == request_context(
        payload, api="anthropic", headers=changed
    )
