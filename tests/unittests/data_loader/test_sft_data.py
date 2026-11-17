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

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from northrl.data_loader.sft_data import default_render_chat, load_render_chat, prepare_sft_item


class FakeTokenizer:
    bos_token_id = 1

    def __init__(self) -> None:
        self.calls = []
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        key = tuple((message["role"], message["content"]) for message in messages)
        add_generation_prompt = kwargs["add_generation_prompt"]
        self.calls.append((key, add_generation_prompt, kwargs.get("tools")))
        rendered = {
            (("system", "rules"), ("user", "question one")): "abcdef",
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
            ): "abcdefghi",
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
            ): "abcdefghijkl",
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
                ("assistant", "answer two"),
            ): "abcdefghijklmn",
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
                ("assistant", "answer two"),
                ("user", "unfinished"),
            ): "abcdefghijklmnop",
            (("user", "question"),): "qr",
            (("user", "question"), ("assistant", "first")): "qrs",
        }
        return rendered[key]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.encode_calls.append(text)
        return {
            "abcdef": [1, 10, 11, 12, 13, 14],
            "ghi": [20, 21, 22],
            "jkl": [30, 31, 14],
            "mn": [40, 22],
            "op": [50, 51],
            "qr": [1, 60, 14],
            "s": [61, 22],
        }[text]


def test_default_render_chat_masks_everything_except_assistant_completions() -> None:
    tokenizer = FakeTokenizer()
    tools = [{"type": "function", "function": {"name": "search"}}]
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "answer two"},
        {"role": "user", "content": "unfinished"},
    ]

    tokens, mask = default_render_chat({"messages": messages, "tools": tools}, tokenizer)

    assert tokens == [1, 10, 11, 12, 13, 14, 20, 21, 22, 30, 31, 14, 40, 22, 50, 51]
    assert mask == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    assert tokenizer.calls == [
        ((("system", "rules"), ("user", "question one")), True, tools),
        (
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
            ),
            False,
            tools,
        ),
        (
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
            ),
            True,
            tools,
        ),
        (
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
                ("assistant", "answer two"),
            ),
            False,
            tools,
        ),
        (
            (
                ("system", "rules"),
                ("user", "question one"),
                ("assistant", "answer one"),
                ("tool", "tool result"),
                ("assistant", "answer two"),
                ("user", "unfinished"),
            ),
            False,
            tools,
        ),
    ]
    assert tokenizer.encode_calls == ["abcdef", "ghi", "jkl", "mn", "op"]


def test_default_render_chat_splits_tokens_at_the_assistant_text_boundary() -> None:
    class _BoundaryMergingTokenizer:
        def __init__(self) -> None:
            self.encode_calls = []

        def apply_chat_template(self, messages, **kwargs):
            del messages
            assert kwargs["tokenize"] is False
            prompt = "<user>question</user><assistant><think>\n"
            if kwargs["add_generation_prompt"]:
                return prompt
            return prompt + "\n</think>\n\n<tool_call>search</tool_call>"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            self.encode_calls.append(text)
            token_ids = []
            index = 0
            while index < len(text):
                if text.startswith("\n\n", index):
                    token_ids.append(271)
                    index += 2
                elif text[index] == "\n":
                    token_ids.append(198)
                    index += 1
                else:
                    token_ids.append(1000 + ord(text[index]))
                    index += 1
            return token_ids

    tokenizer = _BoundaryMergingTokenizer()
    tokens, mask = default_render_chat(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "", "tool_calls": []},
            ]
        },
        tokenizer,
    )

    boundary = next(
        index for index in range(len(tokens) - 1) if tokens[index : index + 2] == [198, 198]
    )
    assert mask[boundary : boundary + 2] == [0, 1]
    assert tokenizer.encode_calls == [
        "<user>question</user><assistant><think>\n",
        "\n</think>\n\n<tool_call>search</tool_call>",
    ]


class _ToolCallTokenizer:
    bos_token_id = None

    def __init__(self) -> None:
        self.arguments = []

    def apply_chat_template(self, messages, **kwargs):
        for message in messages:
            for tool_call in message.get("tool_calls", []):
                self.arguments.append(tool_call["function"]["arguments"])
        return "P" if kwargs["add_generation_prompt"] else "PA"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"P": [10], "A": [20]}[text]


@pytest.mark.parametrize(
    "arguments",
    ['{"query": "northrl", "limit": 2}', {"query": "northrl", "limit": 2}],
)
def test_default_render_chat_normalizes_tool_call_arguments(arguments: object) -> None:
    tokenizer = _ToolCallTokenizer()
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        },
    ]
    original_arguments = messages[1]["tool_calls"][0]["function"]["arguments"]

    tokens, mask = default_render_chat({"messages": messages}, tokenizer)

    assert tokens == [10, 20]
    assert mask == [0, 1]
    assert tokenizer.arguments == [{"query": "northrl", "limit": 2}]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] is original_arguments


def test_default_render_chat_rejects_invalid_json_tool_arguments() -> None:
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search", "arguments": "{not-json}"}}],
        },
    ]

    with pytest.raises(
        ValueError,
        match=r"messages\[1\]\.tool_calls\[0\]\.function\.arguments must contain valid JSON",
    ):
        default_render_chat({"messages": messages}, _ToolCallTokenizer())


@pytest.mark.parametrize("arguments", ["[]", "null", "42", '"text"'])
def test_default_render_chat_rejects_non_object_json_tool_arguments(
    arguments: str,
) -> None:
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search", "arguments": arguments}}],
        },
    ]

    with pytest.raises(
        TypeError,
        match=r"messages\[1\]\.tool_calls\[0\]\.function\.arguments must be a JSON object",
    ):
        default_render_chat({"messages": messages}, _ToolCallTokenizer())


@pytest.mark.parametrize(
    "messages, error",
    [
        ([{"role": "assistant", "content": "answer"}], "must not start"),
        (
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            "consecutive assistant",
        ),
    ],
)
def test_default_render_chat_rejects_assistant_without_a_prompt(
    messages: list[dict[str, str]], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        default_render_chat({"messages": messages}, FakeTokenizer())


def test_default_render_chat_requires_assistant_completion_tokens() -> None:
    class _Tokenizer:
        bos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            del messages
            return "prompt" if kwargs["add_generation_prompt"] else "prompt"

        def encode(self, text, *, add_special_tokens):
            del add_special_tokens
            return list(range(len(text)))

    with pytest.raises(ValueError, match="rendered no completion"):
        default_render_chat(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            _Tokenizer(),
        )


def test_default_render_chat_requires_a_character_prefix() -> None:
    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del messages
            return "prompt" if kwargs["add_generation_prompt"] else "different"

        def encode(self, text, *, add_special_tokens):
            del add_special_tokens
            return list(range(len(text)))

    with pytest.raises(ValueError, match="preserve previously rendered text"):
        default_render_chat(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            _Tokenizer(),
        )


def test_prepare_sft_item_requires_loss_mask_for_tokenized_data() -> None:
    with pytest.raises(ValueError, match="requires loss_mask"):
        prepare_sft_item(
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            tokenizer=None,
            render_chat=None,
            max_sequence_length=8,
        )


def test_load_custom_render_chat(tmp_path: Path) -> None:
    module_path = tmp_path / "custom_sft.py"
    module_path.write_text(
        "def render_chat(row, tokenizer):\n"
        "    assert row['source'] == 'unit-test'\n"
        "    del tokenizer\n"
        "    return [1, 2, 3], [0, 1, 1]\n",
        encoding="utf-8",
    )

    callback = load_render_chat(f"{module_path}:render_chat", None)
    item = prepare_sft_item(
        {"prompt": "question", "answer": "answer", "source": "unit-test"},
        tokenizer=object(),
        render_chat=callback,
        max_sequence_length=8,
    )

    assert item["input_ids"] == [1, 2, 3]
    assert item["loss_mask"] == [0, 1, 1]


def test_prepare_sft_item_rejects_misaligned_result() -> None:
    with pytest.raises(ValueError, match="equal length"):
        prepare_sft_item(
            {"messages": [{"role": "assistant", "content": "answer"}]},
            tokenizer=object(),
            render_chat=lambda row, tokenizer: ([1, 2], [0]),
            max_sequence_length=8,
        )


def test_prepare_sft_item_truncates_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        item = prepare_sft_item(
            {
                "index": "source-7",
                "input_ids": [1, 2, 3, 4, 5],
                "loss_mask": [0, 1, 1, 1, 1],
            },
            tokenizer=None,
            render_chat=None,
            max_sequence_length=4,
        )

    assert item["input_ids"] == [1, 2, 3, 4]
    assert item["loss_mask"] == [0, 1, 1, 1]
    assert "Truncated SFT sample source-7 from 5 to max_sequence_length=4" in caplog.text
