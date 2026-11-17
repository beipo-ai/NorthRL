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

"""Claude Code newline-block compatibility must not weaken other cache checks."""

from copy import deepcopy

import pytest

from northrl.remote_rollout.token_prefix import (
    TokenPrefixCache,
    prefix_cache_messages,
    request_context,
)


def _history(text="\n\n"):
    return [
        {"role": "user", "content": "Run the command.\n\n"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "\nPlan\n", "signature": "signed"},
                {"type": "text", "text": text},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "shell",
                    "input": {"command": "\npwd\n", "text": "\n\n"},
                },
            ],
        },
    ]


def _without_text(history):
    copied = deepcopy(history)
    copied[-1]["content"] = [
        block for block in copied[-1]["content"] if block.get("type") != "text"
    ]
    return copied


def _next(history):
    return [
        *deepcopy(history),
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "\n\n"}],
        },
    ]


@pytest.mark.parametrize("text", ["\n", "\n\n", "\r", "\r\n", "\r\n\n\r"])
@pytest.mark.parametrize("thinking", [False, True])
def test_client_removed_newlines_reuse_original_tokens_without_mutating_history(text, thinking):
    history = _history(text)
    if not thinking:
        history[-1]["content"].pop(0)
    original = deepcopy(history)
    following = _next(_without_text(history))
    original_following = deepcopy(following)
    tokens = [10, 20, 30, 40]  # Includes the original sampled newline tokens.
    context = request_context({"model": "sampler", "messages": history}, api="anthropic")
    cache = TokenPrefixCache()
    cache.remember(prefix_cache_messages(history, api="anthropic"), tokens, context)
    assert cache.match(prefix_cache_messages(following, api="anthropic"), context) == tokens
    # Clients that keep the block can reuse the same token history as well.
    assert cache.match(prefix_cache_messages(_next(history), api="anthropic"), context) == tokens
    assert history == original
    assert following == original_following


def test_multiple_newline_blocks_and_parallel_tools_share_the_same_cache_view():
    history = _history()
    history[-1]["content"].insert(2, {"type": "text", "text": "\r\n"})
    other_tool = deepcopy(history[-1]["content"][-1])
    other_tool["id"] = "call-2"
    history[-1]["content"].append(other_tool)
    assert prefix_cache_messages(history, api="anthropic") == _without_text(history)


@pytest.mark.parametrize("text", ["", "\n \n", "\n\t\n", "\n\u00a0\n", "\nAnswer\n", None, 1])
def test_non_newline_content_remains_an_exact_cache_key(text):
    history = _history(text)
    cache = TokenPrefixCache()
    cache.remember(prefix_cache_messages(history, api="anthropic"), [10, 20], "context")
    assert prefix_cache_messages(history, api="anthropic") == history
    assert (
        cache.match(
            prefix_cache_messages(_next(_without_text(history)), api="anthropic"), "context"
        )
        is None
    )


@pytest.mark.parametrize(
    "boundary",
    ["user", "no-tool", "after-tool", "metadata", "missing-text", "other-text", "non-object"],
)
def test_only_plain_text_before_tools_in_assistant_messages_is_ignored(boundary):
    history = _history()
    blocks = history[-1]["content"]
    if boundary == "user":
        history[-1]["role"] = "user"
    elif boundary == "no-tool":
        blocks.pop()
    elif boundary == "after-tool":
        blocks.append(blocks.pop(1))
    elif boundary == "metadata":
        blocks[1]["citations"] = []
    elif boundary == "missing-text":
        blocks[1].pop("text")
    elif boundary == "other-text":
        blocks.insert(1, {"type": "text", "text": "Actual body"})
    else:
        blocks.append(None)
    assert prefix_cache_messages(history, api="anthropic") == history


@pytest.mark.parametrize("field", ["thinking", "signature", "id", "name", "input"])
def test_reasoning_and_tool_fields_still_require_exact_equality(field):
    history = _history()
    cache = TokenPrefixCache()
    cache.remember(prefix_cache_messages(history, api="anthropic"), [10, 20], "context")
    following = _next(_without_text(history))
    blocks = following[1]["content"]
    target = blocks[0] if field in {"thinking", "signature"} else blocks[1]
    target[field] = {"command": "pwd", "text": "\n\n"} if field == "input" else "changed"
    assert cache.match(prefix_cache_messages(following, api="anthropic"), "context") is None


def test_openai_and_non_block_content_are_not_normalized():
    history = _history()
    assert prefix_cache_messages(history, api="openai") == history
    cache = TokenPrefixCache()
    cache.remember(prefix_cache_messages(history, api="openai"), [10, 20], "context")
    assert (
        cache.match(prefix_cache_messages(_next(_without_text(history)), api="openai"), "context")
        is None
    )
    for content in (None, "\n\n", [], {"type": "text", "text": "\n\n"}):
        history[-1]["content"] = content
        assert prefix_cache_messages(history, api="anthropic") == history


def test_normalized_token_collision_stays_ambiguous_without_shorter_fallback():
    history = _history()
    longer = [*_next(history), deepcopy(history[-1])]
    stripped = _without_text(longer)
    cache = TokenPrefixCache()
    cache.remember(prefix_cache_messages(history, api="anthropic"), [10, 20], "context")
    cache.remember(prefix_cache_messages(longer, api="anthropic"), [10, 20, 30, 40], "context")
    cache.remember(prefix_cache_messages(stripped, api="anthropic"), [10, 20, 30, 41], "context")
    # Repeating either candidate must not erase the persistent ambiguity marker.
    cache.remember(prefix_cache_messages(longer, api="anthropic"), [10, 20, 30, 40], "context")
    assert cache.match(prefix_cache_messages(_next(stripped), api="anthropic"), "context") is None
    assert cache.match(prefix_cache_messages(_next(history), api="anthropic"), "context") == [
        10,
        20,
    ]
