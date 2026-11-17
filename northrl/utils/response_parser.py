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

"""Build assistant messages from raw remote inference responses."""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import re
import sys
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from northrl.utils.path_utils import resolve_path_from_config
from northrl.utils.reasoning_parser import create_reasoning_parser
from northrl.utils.tool_parser import create_tool_parser

logger = logging.getLogger(__name__)

_CUSTOM_PARSER_FUNCTION = "parse_response"


def _normalize_builtin_parser_name(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"rollout.parsers.{field_name} must be a string or null")
    value = value.strip()
    if not value:
        raise ValueError(f"rollout.parsers.{field_name} must not be blank")
    return value


class ResponseParser(Protocol):
    """Common interface for built-in and custom response parsers."""

    def parse(
        self,
        response: str,
        tools: list[dict[str, Any]] | None = None,
        *,
        diagnostics: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Return the final OpenAI-compatible assistant message."""


class _BuiltinResponseParser:  # pylint: disable=too-few-public-methods
    def __init__(self, reasoning_parser: Any | None, tool_parser: Any | None) -> None:
        self._reasoning_parser = reasoning_parser
        self._tool_parser = tool_parser

    def parse(
        self,
        response: str,
        tools: list[dict[str, Any]] | None = None,
        *,
        diagnostics: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        response_content = response
        reasoning_content = None

        if self._reasoning_parser is not None:
            reasoning_result = self._reasoning_parser.parse(response_content)
            if reasoning_result.is_valid:
                response_content = reasoning_result.cleaned_content
                reasoning_content = reasoning_result.reasoning_content
            else:
                logger.warning("Failed to parse reasoning content")

        tool_calls = None
        if diagnostics is not None:
            diagnostics.update(
                tool_calls_detected=0,
                tool_calls_invalid=0,
                tool_format_error=0,
                tool_format_checked=0,
            )
        if self._tool_parser is not None:
            parse_result = self._tool_parser.parse(response_content, tools=tools)
            if diagnostics is not None:
                diagnostics.update(
                    tool_calls_detected=parse_result.detected_count,
                    tool_calls_invalid=parse_result.invalid_count,
                    tool_format_error=int(
                        parse_result.format_status == "invalid" or parse_result.invalid_count > 0
                    ),
                    tool_format_checked=int(parse_result.format_status != "unknown"),
                )
            if parse_result.is_valid and parse_result.tool_calls:
                tool_calls = []
                for tool_call in parse_result.tool_calls:
                    function = dict(tool_call.function)
                    if isinstance(function.get("arguments"), dict):
                        function["arguments"] = json.dumps(function["arguments"])
                    elif "arguments" not in function:
                        function["arguments"] = "{}"

                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": function,
                        }
                    )

                response_content = re.sub(
                    r"<tool_call>.*?</tool_call>\s*",
                    "",
                    response_content,
                    flags=re.DOTALL,
                )
                logger.debug("Parsed tool calls: %s", json.dumps(tool_calls, indent=2))
            elif not parse_result.is_valid:
                logger.warning("Failed to parse tool calls from: %s...", response_content[:100])

        message: dict[str, Any] = {
            "role": "assistant",
            "content": response_content.strip(),
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message


class _CustomResponseParser:  # pylint: disable=too-few-public-methods
    def __init__(self, callback: Callable[..., Any], description: str) -> None:
        self._callback = callback
        self._description = description

    def parse(
        self,
        response: str,
        tools: list[dict[str, Any]] | None = None,
        *,
        diagnostics: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        # Custom callbacks return messages without a raw-syntax diagnosis.
        if diagnostics is not None:
            diagnostics.update(
                tool_calls_detected=0,
                tool_calls_invalid=0,
                tool_format_error=0,
                tool_format_checked=0,
            )
        try:
            parsed = self._callback(
                response=response,
                tools=deepcopy(tools or []),
            )
        except Exception as exc:
            raise RuntimeError(f"Custom response parser '{self._description}' failed") from exc

        if inspect.isawaitable(parsed):
            close = getattr(parsed, "close", None)
            if callable(close):
                close()
            raise TypeError(
                f"Custom response parser '{self._description}' returned an awaitable; "
                "response parsers must be synchronous"
            )

        return _validate_assistant_message(parsed, self._description)


def _validate_assistant_message(parsed: Any, description: str) -> dict[str, Any]:
    if not isinstance(parsed, Mapping):
        raise TypeError(f"Custom response parser '{description}' must return a mapping")

    message = deepcopy(dict(parsed))
    role = message.setdefault("role", "assistant")
    if role != "assistant":
        raise ValueError(f"Custom response parser '{description}' field 'role' must be 'assistant'")

    content = message.setdefault("content", None)
    if content is not None and not isinstance(content, str):
        raise TypeError(
            f"Custom response parser '{description}' field 'content' must be a string or null"
        )

    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise TypeError(
            f"Custom response parser '{description}' field 'reasoning_content' "
            "must be a string or null"
        )

    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list) or not all(
            isinstance(tool_call, Mapping) for tool_call in tool_calls
        ):
            raise TypeError(
                f"Custom response parser '{description}' field 'tool_calls' "
                "must be a list of mappings or null"
            )
        message["tool_calls"] = [deepcopy(dict(tool_call)) for tool_call in tool_calls]

    if content is None and not tool_calls:
        raise ValueError(
            f"Custom response parser '{description}' must return content or tool_calls"
        )
    return message


def _load_custom_response_parser(
    module_path: str,
    config_file_path: str | None,
) -> _CustomResponseParser:
    resolved_path = resolve_path_from_config(module_path, config_file_path)
    if resolved_path is None:
        raise ValueError(f"Could not resolve custom response parser path: {module_path}")

    path = Path(resolved_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Custom response parser module not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Custom response parser module must be a .py file: {path}")

    module_name = f"_northrl_custom_response_parser_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom response parser module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(f"Failed to load custom response parser module: {path}") from exc

        callback = getattr(module, _CUSTOM_PARSER_FUNCTION, None)
        description = f"{module_path}:{_CUSTOM_PARSER_FUNCTION}"
        if callback is None:
            raise AttributeError(
                f"Custom response parser module '{module_path}' "
                f"has no function '{_CUSTOM_PARSER_FUNCTION}'"
            )
        if not callable(callback):
            raise TypeError(f"Configured custom response parser '{description}' is not callable")
        if inspect.iscoroutinefunction(callback):
            raise TypeError(
                f"Custom response parser '{description}' must be a synchronous function"
            )
        try:
            inspect.signature(callback).bind(response="", tools=[])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Custom response parser '{description}' must accept keyword arguments "
                "'response' and 'tools'"
            ) from exc
    finally:
        sys.modules.pop(module_name, None)

    logger.info("Loaded custom response parser: %s", description)
    return _CustomResponseParser(callback, description)


def create_response_parser(
    parser_config: Mapping[str, Any],
    *,
    config_file_path: str | None = None,
) -> ResponseParser:
    """Create the configured built-in or custom response parser."""
    explicit_type = "type" in parser_config
    parser_type = parser_config.get("type", "builtin")
    if parser_type == "custom":
        unknown_fields = set(parser_config) - {"type", "module_path"}
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown custom response parser fields: {fields}")
        module_path = parser_config.get("module_path")
        if not isinstance(module_path, str) or not module_path.strip():
            raise ValueError("rollout.parsers.module_path must be a non-empty string")
        return _load_custom_response_parser(module_path.strip(), config_file_path)

    if parser_type != "builtin":
        raise ValueError(f"Unknown response parser type: {parser_type}")

    allowed_fields = {"type", "reasoning", "tool"} if explicit_type else {"reasoning", "tool"}
    unknown_fields = set(parser_config) - allowed_fields
    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown built-in response parser fields: {fields}")

    if explicit_type:
        reasoning_type = parser_config.get("reasoning", "think_tag")
        tool_type = parser_config.get("tool", "qwen25")
    else:
        reasoning_type = parser_config.get("reasoning") or "think_tag"
        tool_type = parser_config.get("tool") or "qwen25"
    reasoning_type = _normalize_builtin_parser_name(reasoning_type, "reasoning")
    tool_type = _normalize_builtin_parser_name(tool_type, "tool")
    reasoning_parser = (
        create_reasoning_parser(reasoning_type) if reasoning_type is not None else None
    )
    tool_parser = create_tool_parser(tool_type) if tool_type is not None else None
    return _BuiltinResponseParser(reasoning_parser, tool_parser)
