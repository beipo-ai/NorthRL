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
Core types for tool parsing
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ToolCallItem:
    """
    Represents a parsed tool call.

    Attributes:
        id: Unique identifier for the tool call
        type: Type of the call (typically "function")
        function: Dictionary containing function name and arguments
    """

    id: str
    type: str  # typically "function"
    function: dict[str, Any]  # {"name": str, "arguments": str (JSON)}


@dataclass
class ParseResult:
    """
    Result of parsing a tool call string.

    Attributes:
        tool_calls: List of parsed tool calls (None if no valid tool calls found)
        is_valid: Whether parsing retained a usable result, independent of diagnostics.
        invalid_count: Countable calls discarded by parsing or conversion.
        format_status: "valid" means the supported grammar was checked; "invalid"
            means a definite parse/format error; "unknown" means coverage is insufficient.
            A result can retain usable calls while still having an invalid format.
    """

    tool_calls: list[ToolCallItem] | None
    is_valid: bool
    # A response may contain usable calls alongside malformed skipped blocks.
    detected_count: int = 0
    invalid_count: int = 0
    format_status: Literal["valid", "invalid", "unknown"] = "unknown"
