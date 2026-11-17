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

"""
Tool Parser - Framework for parsing tool calls from model responses

This module wraps SGLang's function call parser to provide NorthRL's interface.
Falls back to a minimal implementation if sglang is not installed.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .core_types import ParseResult, ToolCallItem
from .diagnostics import diagnose_sglang
from .minimal import MinimalToolParser

logger = logging.getLogger(__name__)

__all__ = ["ToolCallItem", "ParseResult", "create_tool_parser"]

# Try to import from SGLang
try:
    from sglang.srt.entrypoints.openai.protocol import Function, Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    logger.warning(
        "SGLang not available. Install with: pip install 'NorthRL[sglang]'. "
        "Falling back to minimal tool parser implementation."
    )


class SglangToolParserAdapter:
    """
    Adapter that wraps SGLang's FunctionCallParser to provide NorthRL's interface.

    Handles conversion between:
    - NorthRL tool dicts <-> SGLang Tool objects
    - SGLang ToolCallItem(tool_index, name, parameters) <-> NorthRL ToolCallItem(id, type, function)
    """

    def __init__(self, parser_type: str):
        """
        Initialize the adapter.

        Args:
            parser_type: SGLang parser type (e.g., "qwen25", "deepseekv31")
        """
        self.parser_type = parser_type
        # Get the detector class from SGLang's parser enum
        detector_class = FunctionCallParser.ToolCallParserEnum.get(parser_type)
        if not detector_class:
            raise ValueError(
                f"Unknown tool parser type: {parser_type}. "
                f"Supported types: {', '.join(FunctionCallParser.ToolCallParserEnum.keys())}"
            )
        self.detector = detector_class()

    def _convert_tool_dict_to_sglang(self, tool_dict: dict[str, Any]) -> Tool:
        """
        Convert NorthRL/OpenAI tool dict to SGLang Tool object.

        Args:
            tool_dict: Tool dictionary with OpenAI format

        Returns:
            SGLang Tool object
        """
        function_dict = tool_dict.get("function", {})
        return Tool(
            type=tool_dict.get("type", "function"),
            function=Function(
                name=function_dict.get("name"),
                description=function_dict.get("description"),
                parameters=function_dict.get("parameters"),
            ),
        )

    def _convert_sglang_tool_call_to_northrl(self, sglang_item, idx: int) -> ToolCallItem:
        """
        Convert SGLang ToolCallItem to NorthRL ToolCallItem.

        Args:
            sglang_item: SGLang ToolCallItem with tool_index, name, parameters
            idx: Index for generating unique ID

        Returns:
            NorthRL ToolCallItem with id, type, function
        """
        # Parse parameters if it's a string
        if isinstance(sglang_item.parameters, str):
            try:
                # Try to parse as JSON to ensure it's valid
                params_dict = json.loads(sglang_item.parameters)
                arguments_str = json.dumps(params_dict)
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it as is
                arguments_str = sglang_item.parameters
        else:
            # If it's already a dict, convert to JSON string
            arguments_str = json.dumps(sglang_item.parameters)

        # Generate unique ID
        tool_id = f"call-{uuid.uuid4().hex}-{idx}"

        return ToolCallItem(
            id=tool_id,
            type="function",
            function={"name": sglang_item.name, "arguments": arguments_str},
        )

    def parse(self, text: str, tools: list[dict[str, Any]] | None = None) -> ParseResult:
        """
        Parse tool calls from text.

        Args:
            text: The full response text
            tools: Optional list of available tools (for validation)

        Returns:
            ParseResult with parsed tool calls and validation status
        """
        # Convert tools to SGLang format
        sglang_tools = []
        if tools:
            for tool_dict in tools:
                try:
                    sglang_tools.append(self._convert_tool_dict_to_sglang(tool_dict))
                except Exception as e:
                    logger.warning(f"Failed to convert tool dict: {e}")
                    continue

        try:
            format_status, candidate_count = diagnose_sglang(
                self.parser_type, self.detector, text, tools
            )
        except Exception:  # Diagnostics must never change the accepted output.
            format_status, candidate_count = "unknown", None
        # Use SGLang detector's detect_and_parse
        try:
            sglang_result = self.detector.detect_and_parse(text, sglang_tools)
        except Exception as e:
            logger.error(f"SGLang detector failed: {e}")
            return ParseResult(
                tool_calls=None,
                is_valid=False,
                detected_count=candidate_count or 0,
                invalid_count=candidate_count or 0,
                format_status="invalid",
            )

        # Check if any tool calls were found
        if not sglang_result.calls:
            return ParseResult(
                tool_calls=None,
                is_valid=True,
                detected_count=candidate_count or 0,
                invalid_count=candidate_count or 0,
                format_status="invalid" if candidate_count else format_status,
            )

        # Convert SGLang tool calls to NorthRL format
        northrl_tool_calls = []
        for idx, sglang_item in enumerate(sglang_result.calls):
            try:
                northrl_item = self._convert_sglang_tool_call_to_northrl(sglang_item, idx)
                northrl_tool_calls.append(northrl_item)
            except Exception as e:
                logger.warning(f"Failed to convert tool call: {e}")
                continue

        detected_count = max(candidate_count or 0, len(sglang_result.calls))
        invalid_count = detected_count - len(northrl_tool_calls)
        return ParseResult(
            tool_calls=northrl_tool_calls or None,
            is_valid=bool(northrl_tool_calls),
            detected_count=detected_count,
            invalid_count=invalid_count,
            format_status="invalid" if invalid_count else format_status,
        )


def create_tool_parser(parser_type: str):
    """
    Factory function to create tool parsers based on type.

    Args:
        parser_type: Type of parser to create. Must match SGLang's parser names exactly.
            Supported values (from SGLang's FunctionCallParser.ToolCallParserEnum):
            - "qwen25" or "qwen": Qwen 2.5/3.0 format
            - "qwen3_coder": Qwen 3 Coder format with XML-style parameters
            - "deepseekv3": DeepSeek V3 format
            - "deepseekv31": DeepSeek V3.1 format
            - "deepseekv32": DeepSeek V3.2 format
            - "gpt-oss": GPT OSS/Harmony format
            - "llama3": Llama 3.x format
            - "mistral": Mistral format
            - "pythonic": Pythonic tool call format
            - "kimi_k2": Kimi K2 format
            - "step3": Step3 format
            - "step3p5": Step3.5 format
            - "glm" or "glm45": GLM format
            - "glm47": GLM 4.7 format
            - "hermes": Hermes format
            - "interns1": InternLM S1 format
            - "minimax-m2": MiniMax M2 format
            - "trinity": Trinity format
            - "gigachat3": GigaChat3 format
            - "mimo": MiMo format
            - "lfm2": LFM2 format

    Returns:
        Tool parser instance (SglangToolParserAdapter or MinimalToolParser)

    Raises:
        ValueError: If parser_type is not recognized (when SGLang is available)
    """
    if not SGLANG_AVAILABLE:
        if parser_type == "qwen3_coder":
            logger.warning("SGLang not available, using lightweight qwen3_coder parser.")
        else:
            logger.warning(
                "SGLang not available, using minimal parser for requested type '%s'.",
                parser_type,
            )
        return MinimalToolParser(parser_type)

    # Use parser_type directly - no mapping needed
    # SGLang will validate if it's in FunctionCallParser.ToolCallParserEnum
    try:
        return SglangToolParserAdapter(parser_type)
    except ValueError as e:
        logger.error(f"Failed to create SGLang tool parser: {e}")
        logger.warning("Falling back to minimal parser")
        return MinimalToolParser(parser_type)
