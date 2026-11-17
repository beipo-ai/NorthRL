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

"""Unit tests for built-in and recipe-local response parsers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from northrl.utils import reasoning_parser as reasoning_parser_module
from northrl.utils import tool_parser as tool_parser_module
from northrl.utils.response_parser import create_response_parser


def _write_response_parser(tmp_path: Path, source: str) -> Path:
    parser_path = tmp_path / "custom_response_parser.py"
    parser_path.write_text(source, encoding="utf-8")
    return parser_path


def _custom_parser_config() -> dict[str, str]:
    return {
        "type": "custom",
        "module_path": "custom_response_parser.py",
    }


def _create_custom_parser(tmp_path: Path, source: str):
    _write_response_parser(tmp_path, source)
    return create_response_parser(
        _custom_parser_config(),
        config_file_path=str(tmp_path / "recipe.yaml"),
    )


def test_builtin_response_parser_can_disable_all_parsing() -> None:
    raw_response = (
        "<think>private reasoning</think>visible text\n"
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    )
    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": None,
            "tool": None,
        }
    )

    assert parser.parse(raw_response) == {
        "role": "assistant",
        "content": raw_response,
    }


def test_builtin_qwen3_thinking_parses_closing_only_reasoning_and_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDetector:
        @staticmethod
        def detect_and_parse(text: str) -> SimpleNamespace:
            reasoning, marker, content = text.partition("</think>")
            assert marker
            return SimpleNamespace(reasoning_text=reasoning, normal_text=content)

    class _FakeSglangReasoningParser:
        def __init__(
            self,
            *,
            model_type: str,
            stream_reasoning: bool,
            force_reasoning: bool | None,
        ) -> None:
            assert model_type == "qwen3-thinking"
            assert stream_reasoning is False
            assert force_reasoning is None
            self.detector = _FakeDetector()

    monkeypatch.setattr(reasoning_parser_module, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        reasoning_parser_module,
        "SglangReasoningParser",
        _FakeSglangReasoningParser,
        raising=False,
    )
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)

    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": "qwen3-thinking",
            "tool": "qwen25",
        }
    )
    message = parser.parse(
        'reasoning</think><tool_call>{"name":"lookup",' + '"arguments":{"q":"northrl"}}</tool_call>'
    )

    assert message["role"] == "assistant"
    assert message["reasoning_content"] == "reasoning"
    assert message["content"] == ""
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    assert tool_call["id"]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "lookup"
    assert json.loads(tool_call["function"]["arguments"]) == {"q": "northrl"}


def test_builtin_qwen3_coder_uses_lightweight_xml_parser_without_sglang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)

    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": None,
            "tool": "qwen3_coder",
        }
    )
    message = parser.parse(
        "I'll inspect the repository.\n\n"
        "<tool_call>\n"
        "<function=Bash>\n"
        "<parameter=command>\n"
        'find /workspace -name "*.py" | head -50\n'
        "</parameter>\n"
        "<parameter=description>\n"
        "List Python files\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=Read>\n"
        "<parameter=file_path>\n"
        "/workspace/README.md\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert message["content"] == "I'll inspect the repository."
    assert len(message["tool_calls"]) == 2
    bash_call, read_call = message["tool_calls"]
    assert bash_call["function"]["name"] == "Bash"
    assert json.loads(bash_call["function"]["arguments"]) == {
        "command": 'find /workspace -name "*.py" | head -50',
        "description": "List Python files",
    }
    assert read_call["function"]["name"] == "Read"
    assert json.loads(read_call["function"]["arguments"]) == {"file_path": "/workspace/README.md"}


def test_builtin_qwen3_coder_preserves_typed_parameter_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)
    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": None,
            "tool": "qwen3_coder",
        }
    )

    message = parser.parse(
        "<tool_call><function=Run>"
        "<parameter=timeout>100</parameter>"
        "<parameter=enabled>true</parameter>"
        '<parameter=paths>["a.py", "b.py"]</parameter>'
        "<parameter=options>{'limit': 3}</parameter>"
        "<parameter=command>pwd && ls</parameter>"
        "</function></tool_call>"
    )

    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "timeout": 100,
        "enabled": True,
        "paths": ["a.py", "b.py"],
        "options": {"limit": 3},
        "command": "pwd && ls",
    }


def test_builtin_qwen3_coder_uses_schema_and_preserves_string_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)
    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": None,
            "tool": "qwen3_coder",
        }
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "old_string": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                },
            },
        }
    ]

    message = parser.parse(
        "<tool_call><function=Edit>"
        "<parameter=command>true</parameter>"
        "<parameter=old_string>\n    return true\n\n</parameter>"
        "<parameter=timeout>100</parameter>"
        "<parameter=enabled>true</parameter>"
        "</function></tool_call>",
        tools=tools,
    )

    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "command": "true",
        "old_string": "    return true\n",
        "timeout": 100,
        "enabled": True,
    }


def test_builtin_qwen3_coder_rejects_unparsed_function_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)

    parser = create_response_parser(
        {
            "type": "builtin",
            "reasoning": None,
            "tool": "qwen3_coder",
        }
    )
    response = "\n".join(
        ["<tool_call>", "<function=Bash>unexpected content</function>", "</tool_call>"]
    )

    assert parser.parse(response) == {"role": "assistant", "content": response}


@pytest.mark.parametrize(
    "parser_config",
    [
        {"reasoning": None, "tool": None},
        {
            "type": "builtin",
            "reasoning": " think_tag ",
            "tool": " qwen25 ",
        },
    ],
)
def test_builtin_config_normalizes_names_and_keeps_legacy_defaults(parser_config) -> None:
    raw_response = (
        "<think>legacy reasoning</think>visible text\n"
        '<tool_call>\n{"name":"lookup","arguments":{"q":"northrl"}}\n</tool_call>'
    )
    parser = create_response_parser(parser_config)

    message = parser.parse(
        raw_response,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    )

    assert message["content"] == "visible text"
    assert message["reasoning_content"] == "legacy reasoning"
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "lookup"
    assert tool_call["function"]["arguments"] == '{"q": "northrl"}'


@pytest.mark.parametrize("invalid_name", ["   ", 1])
def test_explicit_builtin_config_rejects_invalid_names(invalid_name) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_response_parser(
            {
                "type": "builtin",
                "reasoning": invalid_name,
                "tool": None,
            }
        )


def test_custom_runtime_config_rejects_extra_fields(tmp_path: Path) -> None:
    _write_response_parser(
        tmp_path,
        """
def parse_response(*, response, tools):
    return {"content": response}
""",
    )

    with pytest.raises(ValueError):
        create_response_parser(
            {
                **_custom_parser_config(),
                "tool": "qwen25",
            },
            config_file_path=str(tmp_path / "recipe.yaml"),
        )


def test_custom_parser_rejects_incompatible_signature_at_startup(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="must accept keyword arguments"):
        _create_custom_parser(
            tmp_path,
            """
def parse_response(response, /, tools):
    return {"content": response}
""",
        )


def test_custom_parser_rejects_async_callback_at_startup(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a synchronous function"):
        _create_custom_parser(
            tmp_path,
            """
async def parse_response(*, response, tools):
    return {"content": response}
""",
        )


def test_custom_parser_defaults_tools_and_preserves_extra_output_fields(
    tmp_path: Path,
) -> None:
    parser = _create_custom_parser(
        tmp_path,
        """
def parse_response(*, response, tools):
    assert tools == []
    tools.append({"seen": True})
    return {
        "content": response,
        "parser_metadata": {"tools": tools},
    }
""",
    )

    message = parser.parse("raw response")

    assert message == {
        "role": "assistant",
        "content": "raw response",
        "parser_metadata": {"tools": [{"seen": True}]},
    }


def test_custom_parser_failure_is_not_silently_ignored(tmp_path: Path) -> None:
    parser = _create_custom_parser(
        tmp_path,
        """
def parse_response(*, response, tools):
    raise RuntimeError("parser boom")
""",
    )

    with pytest.raises(RuntimeError, match="Custom response parser") as exc_info:
        parser.parse("raw response")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "parser boom"


def test_custom_parser_rejects_non_mapping_result(tmp_path: Path) -> None:
    parser = _create_custom_parser(
        tmp_path,
        """
def parse_response(*, response, tools):
    return "not a message"
""",
    )

    with pytest.raises(TypeError, match="must return a mapping"):
        parser.parse("raw response")


def test_custom_parser_missing_function_fails_at_startup(tmp_path: Path) -> None:
    _write_response_parser(tmp_path, "NOT_A_PARSER = True\n")

    with pytest.raises(AttributeError, match="has no function 'parse_response'"):
        create_response_parser(
            _custom_parser_config(),
            config_file_path=str(tmp_path / "recipe.yaml"),
        )


@pytest.mark.parametrize(
    "bad_suffix",
    [
        "<tool_call>{not-json}</tool_call>",
        "<tool_call>{unfinished",
    ],
)
def test_builtin_diagnostics_include_bad_calls_alongside_valid_calls(
    monkeypatch: pytest.MonkeyPatch, bad_suffix: str
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)
    parser = create_response_parser({"type": "builtin", "reasoning": None, "tool": "qwen25"})
    diagnostics: dict[str, int] = {}
    message = parser.parse(
        '<tool_call>{"name":"Read","arguments":{}}</tool_call>' + bad_suffix,
        diagnostics=diagnostics,
    )
    assert len(message["tool_calls"]) == 1
    assert diagnostics == {
        "tool_calls_detected": 2,
        "tool_calls_invalid": 1,
        "tool_format_error": 1,
        "tool_format_checked": 1,
    }
    assert "northrl_parse_diagnostics" not in message


def test_tool_marker_in_arguments_does_not_count_as_another_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", False)
    parser = create_response_parser({"type": "builtin", "reasoning": None, "tool": "qwen25"})
    diagnostics: dict[str, int] = {}
    parser.parse(
        '<tool_call>{"name":"Write","arguments":{"text":"<tool_call>"}}</tool_call>',
        diagnostics=diagnostics,
    )
    assert diagnostics == {
        "tool_calls_detected": 1,
        "tool_calls_invalid": 0,
        "tool_format_error": 0,
        "tool_format_checked": 1,
    }


@pytest.mark.parametrize(
    "tokens",
    [
        {"bot_token": "<tool_call>\n", "eot_token": "\n</tool_call>"},
        {"tool_call_start_token": "<tool_call>", "tool_call_end_token": "</tool_call>"},
    ],
)
def test_sglang_diagnostics_preserve_partial_parse_failures(tokens: dict[str, str]) -> None:
    adapter = object.__new__(tool_parser_module.SglangToolParserAdapter)
    adapter.parser_type = "qwen25"
    adapter.detector = SimpleNamespace(
        **tokens,
        detect_and_parse=lambda text, tools: SimpleNamespace(
            calls=[
                SimpleNamespace(name="Read", parameters="{}"),
            ]
        ),
    )
    parsed = adapter.parse(
        '<tool_call>{"name":"Read","arguments":{}}</tool_call>' "<tool_call>broken</tool_call>"
    )
    assert parsed.is_valid
    assert parsed.tool_calls is not None and len(parsed.tool_calls) == 1
    assert parsed.detected_count == 2
    assert parsed.invalid_count == 1


@pytest.mark.parametrize(
    "response,has_call",
    [
        ("[Read(text='<tool_call> <tool_call>')]", True),
        ("The literal marker is <tool_call>.", False),
    ],
)
def test_sglang_pythonic_diagnostics_ignore_qwen_tags(
    monkeypatch: pytest.MonkeyPatch, response: str, has_call: bool
) -> None:
    class PythonicDetector:
        # SGLang's Pythonic detector inherits empty sentinel tokens from its base.
        bot_token = ""
        eot_token = ""
        tool_call_regex = re.compile(r"\[.*\]", re.DOTALL)
        _text_strip = staticmethod(str.strip)
        _get_parameter_value = staticmethod(ast.literal_eval)

        def detect_and_parse(self, text, tools):
            assert text == response
            return SimpleNamespace(
                calls=(
                    [SimpleNamespace(name="Read", parameters='{"text":"<tool_call> <tool_call>"}')]
                    if has_call
                    else []
                )
            )

    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        tool_parser_module,
        "FunctionCallParser",
        SimpleNamespace(ToolCallParserEnum={"pythonic": PythonicDetector}),
        raising=False,
    )
    parser = create_response_parser({"type": "builtin", "reasoning": None, "tool": "pythonic"})
    diagnostics: dict[str, int] = {}
    message = parser.parse(response, diagnostics=diagnostics)
    assert message["content"] == response
    calls = message.get("tool_calls", [])
    assert len(calls) == int(has_call)
    if has_call:
        assert json.loads(calls[0]["function"]["arguments"]) == {"text": "<tool_call> <tool_call>"}
    assert diagnostics == {
        "tool_calls_detected": int(has_call),
        "tool_calls_invalid": 0,
        "tool_format_error": 0,
        "tool_format_checked": int(has_call),
    }


@pytest.mark.parametrize("parser_kind", ["disabled", "custom"])
def test_response_without_raw_syntax_diagnostics_is_explicitly_unknown(
    tmp_path: Path, parser_kind: str
) -> None:
    if parser_kind == "custom":
        parser = _create_custom_parser(
            tmp_path,
            "def parse_response(*, response, tools): return {'content': response}",
        )
    else:
        parser = create_response_parser({"type": "builtin", "reasoning": None, "tool": None})
    # Reusing a diagnostics mapping must not retain the previous reply's verdict.
    diagnostics = {"tool_format_checked": 1, "tool_format_error": 1, "tool_calls_invalid": 2}
    message = parser.parse("unexamined tool syntax", diagnostics=diagnostics)
    assert message == {"role": "assistant", "content": "unexamined tool syntax"}
    assert diagnostics == {
        "tool_calls_detected": 0,
        "tool_calls_invalid": 0,
        "tool_format_error": 0,
        "tool_format_checked": 0,
    }


def test_sglang_without_syntax_diagnostics_does_not_claim_empty_result_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedDetector:
        bot_token = "<different-tool>"
        eot_token = "</different-tool>"

        def detect_and_parse(self, text, tools):
            return SimpleNamespace(calls=[])

    monkeypatch.setattr(tool_parser_module, "SGLANG_AVAILABLE", True)
    monkeypatch.setattr(
        tool_parser_module,
        "FunctionCallParser",
        SimpleNamespace(ToolCallParserEnum={"unobserved": UnsupportedDetector}),
        raising=False,
    )
    parser = create_response_parser({"type": "builtin", "reasoning": None, "tool": "unobserved"})
    diagnostics: dict[str, int] = {}
    response = "<different-tool>bad call"
    assert parser.parse(response, diagnostics=diagnostics) == {
        "role": "assistant",
        "content": response,
    }
    assert diagnostics == {
        "tool_calls_detected": 0,
        "tool_calls_invalid": 0,
        "tool_format_error": 0,
        "tool_format_checked": 0,
    }
