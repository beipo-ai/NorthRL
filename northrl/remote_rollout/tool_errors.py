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

"""Conservative, fixed-cardinality classification of observed tool failures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_ERROR_CATEGORIES = (
    "command_not_found",
    "dependency_missing",
    "network",
    "permission",
    "resource",
    "file_not_found",
    "edit",
    "test_or_compile",
    "other",
)
# These categories are environment *signals*, not proof of rollout failure causes.
ENV_TOOL_ERROR_CATEGORIES = TOOL_ERROR_CATEGORIES[:5]
_SHELL_TOOLS = frozenset(("bash", "shell", "terminal", "exec_command", "run_command"))
_EDIT_TOOLS = frozenset(("edit", "multiedit", "apply_patch"))
_EXIT_PREFIX = re.compile(r"^\s*Exit code[: ]+(-?\d+)(?:\s|$)", re.IGNORECASE)
_TEST_COMMAND = re.compile(
    r"(?:^\s*|[;&|]\s*)(?:pytest|py\.test|cargo\s+(?:test|build|check)|"
    r"python[\d.]*\s+-m\s+(?:pytest|unittest)|npm\s+(?:test|run\s+(?:test|build))|"
    r"(?:go|mvn|gradle)\s+(?:test|build)|(?:gcc|g\+\+|clang\+\+|clang|rustc|tsc))"
    r"(?:\s|$)",
    re.IGNORECASE,
)
# Priority matters: missing dependencies during a test are not assertion failures.
_ERROR_PATTERNS = tuple(
    (category, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for category, pattern in (
        (
            "command_not_found",
            r"\bcommand not found\b|\b(?:bash|sh):[^\n]*: not found\b|"
            r"\bis not recognized as an internal or external command\b|"
            r"\bexecutable file not found in \$PATH\b",
        ),
        (
            "dependency_missing",
            r"\bModuleNotFoundError\b|\bImportError: No module named\b|"
            r"\bCannot find (?:module|package)\b|\berror while loading shared libraries\b|"
            r"\bNo module named\b|\bPackageNotFoundError\b",
        ),
        (
            "network",
            r"\b(?:connection refused|connection reset|network is unreachable|"
            r"temporary failure in name resolution|could not resolve host|"
            r"failed to establish a new connection|connection timed out|"
            r"connecttimeout|readtimeout|sslerror|certificate verify failed)\b",
        ),
        ("permission", r"\b(?:permission denied|operation not permitted|PermissionError)\b"),
        (
            "resource",
            r"\b(?:no space left on device|disk quota exceeded|out of memory|"
            r"cannot allocate memory|MemoryError|too many open files)\b",
        ),
        (
            "file_not_found",
            r"\b(?:no such file or directory|FileNotFoundError|file (?:not found|does not exist))\b",
        ),
        (
            "edit",
            r"\b(?:old_string|new_string)\b[^\n]*\b(?:same|identical|not found)\b|"
            r"\b(?:no changes to make|string to replace not found|"
            r"patch failed|patch does not apply|failed to find expected lines)\b",
        ),
        (
            "test_or_compile",
            r"\b(?:AssertionError|compilation failed|build failed|could not compile)\b|"
            r"^\s*(?:FAILED\s+\S+|error\[E\d+\]:|test result: FAILED)",
        ),
    )
)


@dataclass(frozen=True)
class ToolCallContext:
    """Retain only classification hints, never command text or file contents."""

    name: str = ""
    test_or_compile: bool = False


def tool_call_context(name: Any, arguments: Any) -> ToolCallContext:
    """Read tool metadata in either provider format, tolerating invalid arguments."""
    name = name.lower().rsplit(".", 1)[-1] if isinstance(name, str) else ""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            arguments = None
    command = (
        arguments.get("command", arguments.get("cmd")) if isinstance(arguments, dict) else None
    )
    return ToolCallContext(
        name=name,
        test_or_compile=(
            name in _SHELL_TOOLS
            and isinstance(command, str)
            and _TEST_COMMAND.search(command) is not None
        ),
    )


def classify_tool_result(result: dict[str, Any], context: ToolCallContext) -> str | None:
    """Return one category only after an explicit failure signal is observed.

    Text inside successful Read/search results is never interpreted as a failure.
    The textual exit-code header is accepted only for a known shell tool. Unknown
    or incomplete failures remain ``other``; a nonzero code alone is not proof of
    an environment problem (for example, grep uses 1 for no matches).
    """
    content = result.get("content")
    structured = content if isinstance(content, dict) else {}
    if context.name in _SHELL_TOOLS and isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (ValueError, RecursionError):
            decoded = None
        if isinstance(decoded, dict):
            structured = decoded
    text = _result_text(structured if structured else content)
    failed = (
        result.get("is_error") is True
        or is_tool_timeout(result)
        or any(_nonzero_exit_code(source.get("exit_code")) for source in (result, structured))
    )
    if not failed and context.name in _SHELL_TOOLS:
        match = _EXIT_PREFIX.match(text)
        failed = match is not None and bool(match.group(1).lstrip("-0"))
    if not failed:
        return None
    for category, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return category
    if context.name in _EDIT_TOOLS:
        return "edit"
    if context.test_or_compile:
        return "test_or_compile"
    return "other"


def _nonzero_exit_code(value: Any) -> bool:
    # Strings, booleans, NaN, and nested arbitrary content are not exit statuses.
    return isinstance(value, int) and not isinstance(value, bool) and value != 0


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    if isinstance(content, dict):
        return "\n".join(
            content[key]
            for key in ("stderr", "stdout", "error", "message", "text", "output")
            if isinstance(content.get(key), str)
        )
    return ""


def is_tool_timeout(result: dict[str, Any]) -> bool:
    """Require an explicit execution receipt, not missing output or quoted text.

    Structured timeout flags/codes are authoritative. Free text is accepted only
    when the harness explicitly marks the receipt as an error and its message
    begins with a timeout status, so a successful file read mentioning timeouts
    cannot create events.
    """
    content = result.get("content")
    structured = content if isinstance(content, dict) else {}
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (ValueError, RecursionError):
            decoded = None
        # A JSON file read is still content, not an execution receipt. Require
        # an explicit error or execution-status envelope before trusting flags.
        if isinstance(decoded, dict) and (
            result.get("is_error") is True
            or decoded.get("is_error") is True
            or "exit_code" in decoded
            or decoded.get("status") in ("timeout", "timed_out")
        ):
            structured = decoded
    for source in (result, structured):
        if source.get("timed_out") is True or source.get("timeout") is True:
            return True
        if source.get("status") in ("timeout", "timed_out"):
            return True
        if source.get("error_type") in ("TimeoutError", "TimeoutExpired", "ToolTimeoutError"):
            return True
    if result.get("is_error") is not True:
        return False
    return (
        re.match(
            r"^\s*(?:Error:\s*)?(?:tool|command|execution)?\s*(?:timed out|timeout(?:error|expired)?\b)",
            _result_text(content),
            re.IGNORECASE,
        )
        is not None
    )
