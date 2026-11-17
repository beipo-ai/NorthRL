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

"""Conservative syntax observations independent of SGLang's accepted output."""

from __future__ import annotations

import ast
from typing import Any, Literal

from .minimal import MinimalToolParser, _has_unframed_tool_syntax, _tagged_call_count

# Reuse parser helpers rather than copying their grammar and conversion rules.
# pylint: disable=protected-access


FormatStatus = Literal["valid", "invalid", "unknown"]


def diagnose_sglang(
    parser_type: str,
    detector: Any,
    text: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[FormatStatus, int | None]:
    """Return format status and a candidate count only for verified grammar subsets.

    SGLang exposes accepted calls, not dropped-call diagnostics. Never infer
    successful validation from an empty result for an unsupported grammar.
    """
    if parser_type in {"qwen", "qwen25", "qwen3_coder"}:
        return _diagnose_qwen(parser_type, text, tools)
    if parser_type == "pythonic":
        return _diagnose_pythonic(detector, text)
    return "unknown", None


def _diagnose_qwen(
    parser_type: str, text: str, tools: list[dict[str, Any]] | None
) -> tuple[FormatStatus, int | None]:
    blocks = MinimalToolParser._TOOL_CALL_PATTERN.findall(text)
    # SGLang permits call arrays and multiple functions within one block. Those
    # extensions are outside the minimal parser's grammar, so do not reject them.
    if any(
        block.lstrip().startswith("[")
        or (parser_type == "qwen3_coder" and block.count("<function=") > 1)
        for block in blocks
    ):
        return "unknown", None
    if parser_type == "qwen3_coder" and not blocks and text.count("<function=") > 1:
        return "unknown", None
    count = _tagged_call_count(text)
    remainder = MinimalToolParser._TOOL_CALL_PATTERN.sub("", text)
    unframed = _has_unframed_tool_syntax(remainder, parser_type)
    invalid = count > len(blocks)
    for block in blocks:
        if parser_type == "qwen3_coder":
            # The native coder detector expects XML, whereas the lightweight
            # fallback additionally accepts JSON. Do not conflate their grammars.
            if block.lstrip().startswith("{"):
                return "unknown", None
            valid_call = (
                MinimalToolParser._parse_qwen3_coder_tool_call(block.strip(), tools) is not None
            )
        else:
            valid_call = MinimalToolParser._parse_json_tool_call(block.strip()) is not None
        invalid |= not valid_call
    if invalid:
        return "invalid", count
    return ("unknown", None) if unframed else ("valid", count)


def _diagnose_pythonic(detector: Any, text: str) -> tuple[FormatStatus, int | None]:
    # Use the detector's own candidate boundary and literal conversion rules.
    # No candidate (including incomplete calls the regex misses) is unknown.
    candidate = detector.tool_call_regex.search(detector._text_strip(text.strip()))
    if candidate is None:
        return "unknown", None
    try:
        expression = ast.parse(candidate.group(0), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return "invalid", None
    if not isinstance(expression, ast.List) or not all(
        isinstance(call, ast.Call) for call in expression.elts
    ):
        return "unknown", None
    for call in expression.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return "unknown", None
        # Positional/unpacked arguments are not covered by the native keyword
        # loop; do not assign a stronger validation result than it can provide.
        if call.args or any(keyword.arg is None for keyword in call.keywords):
            return "unknown", None
        try:
            for keyword in call.keywords:
                detector._get_parameter_value(keyword.value)
        except (ValueError, TypeError, AttributeError, RecursionError):
            return "invalid", len(expression.elts)
    return "valid", len(expression.elts)
