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

"""Known, invalid and unsupported tool syntax must stay distinguishable."""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace

import pytest

from northrl.utils.tool_parser import ParseResult, SglangToolParserAdapter
from northrl.utils.tool_parser.minimal import MinimalToolParser


def _native_call(name="Read", parameters="{}"):
    return SimpleNamespace(name=name, parameters=parameters)


def _adapter(parser_type, calls, *, error=None):
    def detect_and_parse(text, tools):
        if error is not None:
            raise error
        return SimpleNamespace(calls=calls)

    def literal(value):
        if not isinstance(value, ast.Constant):
            raise ValueError("Tool call arguments must be literals")
        return value.value

    adapter = object.__new__(SglangToolParserAdapter)
    adapter.parser_type = parser_type
    adapter.detector = SimpleNamespace(
        detect_and_parse=detect_and_parse,
        tool_call_regex=re.compile(r"\[.*\]", re.DOTALL),
        _text_strip=lambda text: text,
        _get_parameter_value=literal,
    )
    return adapter


def test_uninstrumented_parse_result_defaults_to_unknown():
    assert ParseResult(tool_calls=None, is_valid=True).format_status == "unknown"


@pytest.mark.parametrize("parser_type", ["qwen", "qwen25", "qwen3_coder"])
def test_minimal_known_formats_distinguish_plain_text_and_bad_block(parser_type):
    parser = MinimalToolParser(parser_type)
    assert parser.parse("ordinary answer").format_status == "valid"
    result = parser.parse("<tool_call>broken</tool_call>")
    assert result.format_status == "invalid"
    assert result.invalid_count == 1


@pytest.mark.parametrize("text", ["ordinary answer", "<tool_call>broken</tool_call>"])
def test_unsupported_minimal_fallback_does_not_claim_syntax_coverage(text):
    result = MinimalToolParser("deepseekv3").parse(text)
    assert result.format_status == "unknown"
    assert result.invalid_count == result.detected_count == 0


def test_qwen_partial_bad_blocks_count_only_dropped_calls():
    result = _adapter("qwen25", [_native_call()]).parse(
        '<tool_call>{"name":"Read","arguments":{}}</tool_call>' "<tool_call>broken</tool_call>"
    )
    assert result.is_valid and len(result.tool_calls) == 1
    assert result.format_status == "invalid"
    assert result.detected_count == 2
    assert result.invalid_count == 1


def test_qwen_coder_recovered_unclosed_parameter_is_invalid_without_double_counting():
    result = _adapter("qwen3_coder", [_native_call(parameters='{"text":"hello"}')]).parse(
        "<tool_call><function=Read><parameter=text>hello</function></tool_call>"
    )
    assert result.is_valid and len(result.tool_calls) == 1
    assert result.format_status == "invalid"
    assert result.detected_count == 1
    assert result.invalid_count == 0


@pytest.mark.parametrize(
    "parser_type,text",
    [
        (
            "qwen3_coder",
            "<tool_call><function=Read></function><function=Read></function></tool_call>",
        ),
        (
            "qwen25",
            '<tool_call>[{"name":"Read"},{"name":"Read"}]</tool_call>',
        ),
    ],
)
def test_native_multi_call_extensions_remain_accepted_with_unknown_diagnostics(parser_type, text):
    result = _adapter(parser_type, [_native_call(), _native_call()]).parse(text)
    assert result.is_valid and len(result.tool_calls) == 2
    assert result.format_status == "unknown"
    assert result.detected_count == 2
    assert result.invalid_count == 0


def test_sglang_unknown_format_does_not_treat_empty_output_as_successful_validation():
    result = _adapter("deepseekv3", []).parse("<｜tool▁calls▁begin｜>broken")
    assert result.is_valid  # Preserve the native detector's acceptance behavior.
    assert result.format_status == "unknown"
    assert result.detected_count == result.invalid_count == 0


def test_sglang_whole_response_failure_has_no_invented_call_count():
    result = _adapter("deepseekv3", [], error=ValueError("cannot parse")).parse("broken")
    assert not result.is_valid
    assert result.format_status == "invalid"
    assert result.detected_count == result.invalid_count == 0


def test_known_conversion_failure_counts_one_call_without_claiming_complete_coverage():
    result = _adapter("deepseekv3", [SimpleNamespace(name="Read")]).parse("native call")
    assert not result.is_valid
    assert result.format_status == "invalid"
    assert result.detected_count == result.invalid_count == 1


@pytest.mark.parametrize(
    "text,calls,status,invalid_count",
    [
        ("[Read(text='<tool_call> <tool_call>')]", [_native_call()], "valid", 0),
        ("[Read(text='one'), Read(text='two')]", [_native_call(), _native_call()], "valid", 0),
        ("[Read(text=value)]", [], "invalid", 1),
        ("[Read(text=)]", [], "invalid", 0),
        ("[Read(text='unfinished'", [], "unknown", 0),
        ("The literal marker is <tool_call>.", [], "unknown", 0),
    ],
)
def test_pythonic_uses_its_own_candidate_and_ast_literal_grammar(
    text, calls, status, invalid_count
):
    result = _adapter("pythonic", calls).parse(text)
    assert result.is_valid
    assert len(result.tool_calls or []) == len(calls)
    assert result.format_status == status
    assert result.invalid_count == invalid_count


def test_diagnostics_failure_cannot_change_native_accepted_output():
    adapter = _adapter("pythonic", [_native_call()])
    adapter.detector.tool_call_regex = None
    result = adapter.parse("native call")
    assert result.is_valid and len(result.tool_calls) == 1
    assert result.format_status == "unknown"
    assert result.invalid_count == 0


@pytest.mark.parametrize(
    "parser_type,text",
    [
        ("qwen3_coder", "<function=Read></function>"),
        ("qwen3_coder", "<parameter=text>hello</parameter>"),
        ("qwen3_coder", "</function>"),
        ("qwen3_coder", "</tool_call>"),
        ("qwen25", "</tool_call>"),
    ],
)
def test_unframed_qwen_syntax_does_not_look_like_checked_plain_text(parser_type, text):
    result = _adapter(parser_type, []).parse(text)
    assert result.is_valid
    assert result.format_status == "unknown"
    assert result.invalid_count == result.detected_count == 0
    fallback = MinimalToolParser(parser_type).parse(text)
    assert fallback.is_valid
    assert fallback.format_status == "unknown"
    assert fallback.invalid_count == fallback.detected_count == 0


@pytest.mark.parametrize(
    "parser_type,valid_call,remainder",
    [
        ("qwen25", '<tool_call>{"name":"Read","arguments":{}}</tool_call>', "</tool_call>"),
        ("qwen3_coder", "<tool_call><function=Read></function></tool_call>", "<function=Read>"),
    ],
)
def test_valid_calls_with_unframed_remainder_have_unknown_coverage(
    parser_type, valid_call, remainder
):
    for parser in (_adapter(parser_type, [_native_call()]), MinimalToolParser(parser_type)):
        result = parser.parse(valid_call + remainder)
        assert result.is_valid and len(result.tool_calls) == 1
        assert result.format_status == "unknown"
        assert result.invalid_count == 0
        malformed = parser.parse(valid_call + "<tool_call>broken</tool_call>" + remainder)
        assert malformed.format_status == "invalid"
        assert malformed.invalid_count == 1


def test_tool_syntax_in_valid_arguments_does_not_create_unframed_remainder():
    text = '<tool_call>{"name":"Read","arguments":{"text":"<function=Read>"}}</tool_call>'
    for parser in (_adapter("qwen25", [_native_call()]), MinimalToolParser("qwen25")):
        result = parser.parse(text)
        assert result.format_status == "valid"
        assert result.invalid_count == 0
