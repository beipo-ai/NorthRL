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

"""Match completed native message histories to original tokens within one rollout."""

import json
from collections.abc import Mapping
from typing import Any, Literal

_SAMPLING_ONLY_FIELDS = frozenset({"temperature", "top_p", "top_k", "seed"})
_CONTEXT_EXCLUDED_FIELDS = _SAMPLING_ONLY_FIELDS | {
    "messages",
    "history_tokens",
    "stream",
    "stream_options",
}


def _canonical(value: Any) -> str:
    # Inputs are parsed JSON. Canonical JSON ignores object key order while
    # distinguishing true/1, 1/1.0, missing/null, arrays and exact strings.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def request_context(
    payload: dict[str, Any],
    *,
    api: Literal["openai", "anthropic"],
    headers: Mapping[str, str] | None = None,
) -> str:
    """Compare protocol and effective request settings, including unknown fields."""
    context_body = {
        key: value for key, value in payload.items() if key not in _CONTEXT_EXCLUDED_FIELDS
    }
    if "sampling_params" in context_body:
        context_body["sampling_params"] = {
            key: value
            for key, value in context_body["sampling_params"].items()
            if key not in _SAMPLING_ONLY_FIELDS
        }
    protocol_headers = {
        key.lower(): value
        for key, value in (headers or {}).items()
        if api == "anthropic" and key.lower() in {"anthropic-version", "anthropic-beta"}
    }
    return _canonical({"api": api, "protocol_headers": protocol_headers, "request": context_body})


def _message_key(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(_canonical(message) for message in messages)


def prefix_cache_messages(
    messages: list[dict[str, Any]], *, api: Literal["openai", "anthropic"]
) -> list[dict[str, Any]]:
    """Build a read-only cache view, preserving the original native messages.

    SGLang's qwen3 reasoning + qwen3_coder tool parser preserves the CR/LF
    prefix before a tool call as content. Claude Code removes the resulting
    newline-only text block from subsequent history. Ignore that block only
    for cache comparison; responses, transcripts and original tokens retain it.
    Both cache writes and lookups must use this view, including earlier turns.
    """
    if api != "anthropic":
        return messages
    normalized = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if (
            not isinstance(content, list)
            or message.get("role") != "assistant"
            or any(not isinstance(block, dict) for block in content)
        ):
            normalized.append(message)
            continue
        first_tool = next(
            (index for index, block in enumerate(content) if block.get("type") == "tool_use"),
            len(content),
        )
        text_blocks = [
            (index, block) for index, block in enumerate(content) if block.get("type") == "text"
        ]
        # Do not erase actual body text, mixed whitespace, empty blocks, text
        # after a tool, or metadata we do not understand. Never recurse into
        # thinking, tool inputs/results or other nested message fields.
        if (
            first_tool < len(content)
            and text_blocks
            and all(
                index < first_tool
                and set(block) == {"type", "text"}
                and isinstance(block["text"], str)
                and block["text"]
                and not block["text"].strip("\r\n")
                for index, block in text_blocks
            )
        ):
            normalized.append(
                {**message, "content": [block for block in content if block.get("type") != "text"]}
            )
        else:
            normalized.append(message)
    return normalized


class TokenPrefixCache:
    """Keep immutable histories and persistent ambiguity markers until clear().

    An ambiguous longest match must not fall back to a shorter history. There
    is deliberately no eviction: forgetting one branch could make another
    branch appear unambiguous. The owning relay clears this cache at rollout end.
    """

    def __init__(self) -> None:
        self._histories: dict[str, dict[tuple[str, ...], tuple[int, ...] | None]] = {}

    def remember(self, messages: list[dict[str, Any]], tokens: list[int], context: str) -> None:
        """Remember a completed assistant turn after its response was validated."""
        if not messages or not tokens:
            return
        if messages[-1].get("role") != "assistant":
            return

        key = _message_key(messages)
        original_tokens = tuple(tokens)
        histories = self._histories.setdefault(context, {})
        if key not in histories:
            histories[key] = original_tokens
        elif histories[key] != original_tokens:
            # This also leaves an existing ambiguity marker in place.
            histories[key] = None

    def match(self, messages: list[dict[str, Any]], context: str) -> list[int] | None:
        """Return independent history_tokens for the longest unambiguous history."""
        histories = self._histories.get(context)
        if not histories:
            return None
        key = _message_key(messages)
        # At least one message must follow the completed historical assistant.
        for count in range(len(key) - 1, 0, -1):
            prefix = key[:count]
            if prefix in histories:
                tokens = histories[prefix]
                if tokens is None:
                    return None
                return list(tokens)
        return None

    def clear(self) -> None:
        self._histories.clear()
