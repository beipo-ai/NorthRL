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

"""Lightweight tool-call parsing without the SGLang dependency."""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any, Literal

from .core_types import ParseResult, ToolCallItem

logger = logging.getLogger(__package__)


_TAGGED_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _tagged_call_count(text: str) -> int:
    """Count complete blocks and remaining starts, ignoring starts inside blocks."""
    return len(_TAGGED_TOOL_CALL_PATTERN.findall(text)) + _TAGGED_TOOL_CALL_PATTERN.sub(
        "", text
    ).count("<tool_call>")


def _has_unframed_tool_syntax(text: str, parser_type: str | None) -> bool:
    """Recognize orphan tool syntax when no outer call starts were found."""
    return "</tool_call>" in text or (
        parser_type == "qwen3_coder"
        and any(
            token in text for token in ("<function=", "</function>", "<parameter=", "</parameter>")
        )
    )


class MinimalToolParser:
    """
    Minimal fallback parser when SGLang is not available.

    Handles both the basic ``<tool_call>JSON</tool_call>`` format and Qwen3
    Coder's XML-style function and parameter format.
    """

    _TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _QWEN3_FUNCTION_PATTERN = re.compile(
        r"\A\s*<function=([^>\r\n]+)>(.*?)</function>\s*\Z",
        re.DOTALL,
    )
    _QWEN3_PARAMETER_PATTERN = re.compile(
        r"<parameter=([^>\r\n]+)>(.*?)</parameter>",
        re.DOTALL,
    )

    def __init__(self, parser_type: str | None = None) -> None:
        self.parser_type = parser_type

    @staticmethod
    def _parameter_schemas(
        tools: list[dict[str, Any]] | None,
        function_name: str,
    ) -> dict[str, dict[str, Any]]:
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                parameters = function.get("parameters")
            else:
                name = tool.get("name")
                parameters = tool.get("input_schema")
            if name != function_name or not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                return {}
            return {
                name: schema
                for name, schema in properties.items()
                if isinstance(name, str) and isinstance(schema, dict)
            }
        return {}

    @staticmethod
    def _declared_schema_types(schema: dict[str, Any] | None) -> set[str]:
        if schema is None:
            return set()
        declared = schema.get("type")
        if isinstance(declared, str):
            return {declared}
        if isinstance(declared, list):
            return {item for item in declared if isinstance(item, str)}
        return set()

    @staticmethod
    def _matches_schema_type(value: Any, declared_types: set[str]) -> bool:
        if not declared_types:
            return True
        return any(
            (
                (declared == "null" and value is None)
                or (declared == "boolean" and isinstance(value, bool))
                or (
                    declared == "integer" and isinstance(value, int) and not isinstance(value, bool)
                )
                or (
                    declared == "number"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
                or (declared == "array" and isinstance(value, list))
                or (declared == "object" and isinstance(value, dict))
                or (declared == "string" and isinstance(value, str))
            )
            for declared in declared_types
        )

    @staticmethod
    def _unwrap_parameter_value(value: str) -> str:
        """Remove only the XML formatting newlines around a parameter value."""

        if value.startswith("\r\n"):
            value = value[2:]
        elif value.startswith("\n"):
            value = value[1:]
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]
        return value

    @classmethod
    def _parse_parameter_value(
        cls,
        raw_value: str,
        schema: dict[str, Any] | None,
    ) -> Any:
        declared_types = cls._declared_schema_types(schema)
        if "string" in declared_types:
            return raw_value

        for value_parser in (json.loads, ast.literal_eval):
            try:
                candidate = value_parser(raw_value)
                json.dumps(candidate)
            except (json.JSONDecodeError, SyntaxError, TypeError, ValueError):
                continue
            if cls._matches_schema_type(candidate, declared_types):
                return candidate
        return raw_value

    @staticmethod
    def _parse_json_tool_call(text: str) -> tuple[str, str, str | None] | None:
        try:
            tool_call = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None

        name = tool_call.get("name")
        if not isinstance(name, str) or not name.strip():
            return None

        args = tool_call.get("arguments") or tool_call.get("parameters")
        if args is None:
            arguments_str = "{}"
        elif isinstance(args, str):
            arguments_str = args
        elif isinstance(args, dict):
            arguments_str = json.dumps(args)
        else:
            arguments_str = str(args)

        tool_id = tool_call.get("id")
        return name.strip(), arguments_str, tool_id if isinstance(tool_id, str) else None

    @classmethod
    def _parse_qwen3_coder_tool_call(
        cls,
        text: str,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, str, None] | None:
        function_match = cls._QWEN3_FUNCTION_PATTERN.fullmatch(text)
        if function_match is None:
            return None

        function_name = function_match.group(1).strip()
        if not function_name:
            return None

        function_body = function_match.group(2)
        parameter_schemas = cls._parameter_schemas(tools, function_name)
        arguments: dict[str, Any] = {}
        cursor = 0
        for parameter_match in cls._QWEN3_PARAMETER_PATTERN.finditer(function_body):
            if function_body[cursor : parameter_match.start()].strip():
                return None

            parameter_name = parameter_match.group(1).strip()
            if not parameter_name or parameter_name in arguments:
                return None
            raw_value = cls._unwrap_parameter_value(parameter_match.group(2))
            arguments[parameter_name] = cls._parse_parameter_value(
                raw_value,
                parameter_schemas.get(parameter_name),
            )
            cursor = parameter_match.end()

        if function_body[cursor:].strip():
            return None

        return function_name, json.dumps(arguments), None

    def parse(self, text: str, tools: list[dict[str, Any]] | None = None) -> ParseResult:
        """
        Parse tool calls from text using simple pattern matching.

        Args:
            text: The full response text
            tools: Optional list of available tools used for parameter schemas

        Returns:
            ParseResult with parsed tool calls and validation status
        """
        diagnostics_supported = self.parser_type in {None, "qwen", "qwen25", "qwen3_coder"}
        if "<tool_call>" not in text:
            return ParseResult(
                tool_calls=None,
                is_valid=True,
                format_status=(
                    "valid"
                    if diagnostics_supported
                    and not _has_unframed_tool_syntax(text, self.parser_type)
                    else "unknown"
                ),
            )

        matches = self._TOOL_CALL_PATTERN.findall(text)
        detected_count = _tagged_call_count(text)
        if not matches:
            return ParseResult(
                tool_calls=None,
                is_valid=False,
                detected_count=detected_count if diagnostics_supported else 0,
                invalid_count=detected_count if diagnostics_supported else 0,
                format_status="invalid" if diagnostics_supported else "unknown",
            )

        tool_calls = []
        for idx, match in enumerate(matches):
            match = match.strip()
            parsed = self._parse_json_tool_call(match)
            if parsed is None and self.parser_type == "qwen3_coder":
                parsed = self._parse_qwen3_coder_tool_call(match, tools)
            if parsed is None:
                logger.warning(
                    "Failed to parse %s tool call: %s...",
                    self.parser_type or "minimal",
                    match[:100],
                )
                continue

            name, arguments_str, configured_tool_id = parsed
            tool_id = configured_tool_id or f"call-{uuid.uuid4().hex}-{idx}"
            tool_calls.append(
                ToolCallItem(
                    id=tool_id,
                    type="function",
                    function={"name": name, "arguments": arguments_str},
                )
            )

        format_status: Literal["valid", "invalid", "unknown"] = "unknown"
        if diagnostics_supported:
            if detected_count > len(tool_calls):
                format_status = "invalid"
            elif not _has_unframed_tool_syntax(
                self._TOOL_CALL_PATTERN.sub("", text), self.parser_type
            ):
                format_status = "valid"
        return ParseResult(
            tool_calls=tool_calls or None,
            is_valid=bool(tool_calls),
            detected_count=detected_count if diagnostics_supported else 0,
            invalid_count=detected_count - len(tool_calls) if diagnostics_supported else 0,
            format_status=format_status,
        )
