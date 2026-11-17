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

"""Forward native chat HTTP/SSE while observing exact NorthGate training data."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from copy import deepcopy
from email.utils import parsedate_to_datetime
from itertools import count
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from omegaconf import DictConfig, OmegaConf

from ..exceptions import ContextLengthExceededError
from ..utils.config_utils import get_algorithm_loss_mode, get_derived_sampling_flags
from .metrics import RolloutMetrics
from .token_prefix import TokenPrefixCache, prefix_cache_messages, request_context
from .trajectory_recorder import TrajectoryRecorder

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}
_AUTH_HEADERS = {"authorization", "x-api-key", "anthropic-auth-token", "cookie"}
_FRAME_END = re.compile(rb"\r\n\r\n|\n\n|\r\r")
# Transport recovery is an internal relay policy, not a rollout configuration.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_WINDOW_SECONDS = 300.0
# A client that gives up after this long without any upstream bytes is reported as
# a model-service failure; earlier disconnects from a healthy call are cancellations.
_UPSTREAM_STALL_SECONDS = 300.0
logger = logging.getLogger(__name__)

type _InputKeyOrder = tuple[tuple[str | int, _InputKeyOrder | None], ...]


def _strip_anthropic_cache_control(payload: dict[str, Any]) -> None:
    """Remove prompt-cache metadata from the relay's private request copy.

    Cache breakpoints move between turns and must not invalidate the token-history
    match. Only visit protocol containers: tool inputs, schemas, and other nested
    business data may legitimately contain a field named ``cache_control``.
    """
    payload.pop("cache_control", None)
    containers = [payload.get("system"), payload.get("tools")]
    containers.extend(
        message.get("content") for message in payload["messages"] if isinstance(message, dict)
    )
    for items in containers:
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item.pop("cache_control", None)


def _input_key_order(value: Any) -> _InputKeyOrder | None:
    """Keep object key order, without retaining command/code/string values."""
    if isinstance(value, dict):
        return tuple((key, _input_key_order(item)) for key, item in value.items())
    if isinstance(value, list):
        children = ((index, _input_key_order(item)) for index, item in enumerate(value))
        return tuple((index, child) for index, child in children if child is not None) or None
    return None


def _input_fingerprint(value: dict[str, Any]) -> bytes:
    # JSON distinguishes bool/int and int/float, unlike Python object equality.
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _apply_input_key_order(value: Any, order: _InputKeyOrder) -> Any:
    if isinstance(value, dict):
        return {
            key: _apply_input_key_order(value[key], child) if child is not None else value[key]
            for key, child in order
        }
    for index, child in order:
        if child is not None:
            value[index] = _apply_input_key_order(value[index], child)
    return value


def _tool_input_identity(block: Any) -> tuple[str, str] | None:
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return None
    if not isinstance(block.get("input"), dict):
        return None
    call_id, name = block.get("id"), block.get("name")
    if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
        return call_id, name
    return None


def _with_client_model(result: dict[str, Any], model: str) -> dict[str, Any]:
    # Claude Code treats a different response model as a model switch and may
    # discard historical thinking. Keep the upstream identity for recording.
    if "model" not in result or result["model"] == model:
        return result
    return {**result, "model": model}


def _client_sse_frame(
    frame: bytes,
    event: dict[str, Any] | None,
    api: Literal["openai", "anthropic"],
    model: str,
) -> bytes:
    if event is None:
        return frame
    if api == "anthropic":
        if event.get("type") != "message_start":
            return frame
        message = event["message"]
        rewritten = _with_client_model(message, model)
        if rewritten is message:
            return frame
        outgoing = {**event, "message": rewritten}
    else:
        outgoing = _with_client_model(event, model)
        if outgoing is event:
            return frame

    # Replace only the data payload; retain event/id/retry/comment lines and
    # framing, including when the upstream JSON spans several data lines.
    data = json.dumps(outgoing, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    lines = frame.splitlines(keepends=True)
    emitted = False
    for index, line in enumerate(lines):
        if line.startswith(b"data:"):
            ending = line[len(line.rstrip(b"\r\n")) :]
            lines[index] = b"data: " + data + ending if not emitted else b""
            emitted = True
    return b"".join(lines)


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _response_request_id(headers: httpx.Headers) -> str | None:
    for name in ("x-northgate-request-id", "x-request-id", "request-id"):
        if value := headers.get(name):
            return value
    return None


class _UpstreamFailure(RuntimeError):
    def __init__(
        self,
        status: int | None,
        body: Any,
        *,
        source: Literal["http", "sse"] = "http",
        request_id: str | None = None,
    ) -> None:
        # A streaming error travels inside a successful HTTP response. Its own
        # status/code must not turn that HTTP response into a fabricated 400.
        self.status = status
        self.http_status = status
        self.source = source
        self.body = body
        error = body.get("error", body) if isinstance(body, dict) else body
        self.code = error.get("code", error.get("type")) if isinstance(error, dict) else None
        self.sse_status: int | None = None
        if source == "sse":
            for item in (error, body):
                if not isinstance(item, dict):
                    continue
                for name in ("status", "status_code"):
                    value = item.get(name)
                    if isinstance(value, str) and value.isascii() and value.isdecimal():
                        value = int(value)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 100 <= value < 600
                    ):
                        self.sse_status = value
                        break
                if self.sse_status is not None:
                    break
        self.request_id = request_id
        if self.request_id is None:
            for item in (body, error):
                if not isinstance(item, dict):
                    continue
                value = item.get("request_id")
                if isinstance(value, str) and value:
                    self.request_id = value
                    break
        detail = error.get("message", "") if isinstance(error, dict) else error
        if not detail and isinstance(body, dict):
            detail = body.get("message", "")
        description = f"HTTP {status}" if status is not None else "HTTP status unknown"
        if source == "sse":
            description += f", SSE status={self.sse_status} code={self.code}"
        super().__init__(f"NorthGate request failed ({description}): {str(detail)[:2000]}")


def _context_error(body: Any, payload: dict[str, Any]) -> ContextLengthExceededError | None:
    error = body.get("error", body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {"code": error}
    message = str(error.get("message", body.get("message", "") if isinstance(body, dict) else body))
    code = error.get("code", error.get("type"))
    if code not in {"sequence_too_long", "context_length_exceeded"} and not re.search(
        r"sequence length exceeds model max_seq_len|sequence length.*exceeds max_seq_len|context[ _](?:length|window).*exceed|exceed.*context[ _](?:length|window)",
        message,
        re.IGNORECASE,
    ):
        return None
    details = error.get("details", body.get("details", {}) if isinstance(body, dict) else {})
    details = details if isinstance(details, dict) else {}

    def integer(*names: str) -> int | None:
        for name in names:
            value = details.get(name, error.get(name))
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    prompt = integer("prompt_tokens", "prompt_length")
    limit = integer("max_seq_len", "max_sequence_length")
    requested_max_tokens = integer("max_new_tokens", "max_tokens", "requested_max_tokens")
    if requested_max_tokens is None:
        maximum = payload.get("max_completion_tokens")
        if maximum is None:
            maximum = payload.get("max_tokens")
        params = {"max_tokens": maximum, **payload.get("sampling_params", {})}
        for name in ("max_new_tokens", "max_tokens"):
            value = params.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                requested_max_tokens = value
                break
    match = re.search(
        r"prompt\s+(\d+)\s*\+\s*max_new_tokens\s+(\d+)\s*=\s*\d+\s*>\s*(\d+)", message
    )
    if match:
        prompt, requested_max_tokens, limit = map(int, match.groups())
    elif limit is None:
        match = re.search(r"max_seq_len\s*[=:]?\s*(\d+)", message)
        if match:
            limit = int(match.group(1))
    return ContextLengthExceededError(prompt, requested_max_tokens, limit)


class _CallObservation:
    """Keep request timing alive through stream termination and explicit recovery."""

    # This private helper owns the relay's request lifecycle state.
    # pylint: disable=protected-access

    def __init__(
        self,
        relay: "NorthGateRelay",
        api: Literal["openai", "anthropic"],
        path: str,
        payload: dict[str, Any],
    ):
        self.relay = relay
        self.api = api
        self.path = path
        self.started_at = time.monotonic()
        self.http_status: int | None = None
        self.request_id: str | None = None
        self.response_emitted = False
        self.attempts = 0
        self._upstream_failure: _UpstreamFailure | None = None
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.key = (path, hashlib.sha256(encoded.encode()).digest())
        self.operation = f"northgate request {next(relay._call_ids)}"
        self.previous_failures = relay._failed_calls.get(self.key, set()).copy()
        self.context = relay._metrics.track_request(api)
        self.timing = self.context.__enter__()
        self.finished = False
        # Upstream failure this call is backing off from before a retry, if any.
        self.retrying: str | None = None
        self.last_upstream_activity = time.monotonic()
        relay._observations.add(self)

    def remember_failure(self, error: _UpstreamFailure) -> None:
        """Retain protocol diagnostics when context limits get a typed exception."""
        self._upstream_failure = error
        self.http_status = error.http_status
        self.request_id = error.request_id

    def _log_failure(self, error: BaseException) -> None:
        # The relay wraps a protocol error only for an exhausted retry window.
        # Implicit context may describe an unrelated, recovered attempt.
        cause = error if isinstance(error, _UpstreamFailure) else error.__cause__
        protocol_error = cause if isinstance(cause, _UpstreamFailure) else None
        if protocol_error is None and isinstance(error, ContextLengthExceededError):
            protocol_error = self._upstream_failure
        try:
            request_id = (
                protocol_error.request_id if protocol_error is not None else None
            ) or self.request_id
            # Never include request/response bodies, headers, or exception text.
            # JSON also escapes control characters in upstream diagnostic IDs.
            logger.warning(
                "NorthGate request failed: %s",
                json.dumps(
                    {
                        "api": self.api,
                        "path": self.path,
                        "payload_sha256": self.key[1].hex(),
                        "operation": self.operation,
                        "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                        "error_type": type(error).__name__,
                        "source": (
                            protocol_error.source
                            if protocol_error is not None
                            else "transport" if isinstance(error, httpx.TransportError) else "local"
                        ),
                        "http_status": (
                            protocol_error.http_status
                            if protocol_error is not None
                            else self.http_status
                        ),
                        "sse_status": protocol_error.sse_status if protocol_error else None,
                        "error_code": (
                            str(protocol_error.code)[:256]
                            if protocol_error is not None and protocol_error.code is not None
                            else None
                        ),
                        "request_id": str(request_id)[:256] if request_id else None,
                        "response_emitted": self.response_emitted,
                        "attempts": self.attempts,
                    },
                    sort_keys=True,
                ),
            )
        except Exception:  # Diagnostic sinks must not alter failure/recovery bookkeeping.
            pass

    def upstream_activity(self) -> None:
        self.last_upstream_activity = time.monotonic()

    def finish(self, error: BaseException | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        self.relay._observations.discard(self)
        self.context.__exit__(
            type(error) if error is not None else None,
            error,
            error.__traceback__ if error is not None else None,
        )
        if error is not None:
            silent_seconds = time.monotonic() - self.last_upstream_activity
            if isinstance(error, Exception):
                failure = error
            elif self.retrying is not None:
                failure = ConnectionError(
                    f"NorthGate response was interrupted while retrying {self.retrying}"
                )
            elif silent_seconds >= _UPSTREAM_STALL_SECONDS:
                failure = ConnectionError(
                    "NorthGate response was interrupted after "
                    f"{silent_seconds:.0f}s without upstream data"
                )
            else:
                # The client stopped waiting for a healthy upstream call, e.g. the
                # agent budget stopped the agent or the agent exited mid-request.
                # The model service did not fail, so the rollout stays evaluable.
                return
            self._log_failure(error)
            if failure is not error:
                failure.__cause__ = error
            self.relay._failed_calls.setdefault(self.key, set()).add(self.operation)
            self.relay._on_service_call(self.operation, failure)
            return
        failures = self.relay._failed_calls.get(self.key, set())
        recovered = self.previous_failures & failures
        failures.difference_update(recovered)
        if not failures:
            self.relay._failed_calls.pop(self.key, None)
        for operation in recovered:
            self.relay._on_service_call(operation, None)
        self.relay._on_service_call(self.operation, None)


class _StreamObservation:
    """Read provider-native deltas for transcripts; never generate chat events."""

    _OPENAI_TEXT_FIELDS = frozenset({"content", "refusal", "reasoning_content"})
    _ANTHROPIC_DELTAS = {
        "text_delta": ("text", "text"),
        "thinking_delta": ("thinking", "thinking"),
        "signature_delta": ("thinking", "signature"),
        "input_json_delta": ("tool_use", "partial_json"),
    }

    def __init__(
        self,
        api: Literal["openai", "anthropic"],
        *,
        http_status: int | None = None,
        request_id: str | None = None,
    ):
        self.api = api
        self.http_status = http_status
        self.request_id = request_id
        self.result: dict[str, Any] = {}
        self.assistant: dict[str, Any] = {"role": "assistant"}
        self.tools: dict[int, dict[str, Any]] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.arguments: dict[int, str] = {}
        self.open_blocks: set[int] = set()
        self.message_started = False
        self.finished = False
        self.finish_reason: str | None = None
        self.terminal = False
        self.metadata_seen = False
        # Unknown incremental fields may be forwarded and recorded, but we
        # cannot use an incomplete reconstruction to identify future history.
        self.prefix_reusable = True

    def observe(self, frame: bytes) -> dict[str, Any] | None:
        lines = frame.decode("utf-8").splitlines()
        data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
        if not data:
            return None
        if self.terminal:
            raise ValueError("NorthGate sent data after stream termination")
        if data == "[DONE]":
            if self.api != "openai" or not self.finished:
                raise ValueError("NorthGate stream ended without a finish reason")
            self.terminal = True
            return None
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("NorthGate SSE data must be an object")
        if "error" in event or event.get("type") == "error":
            raise _UpstreamFailure(
                self.http_status, event, source="sse", request_id=self.request_id
            )
        if "token_data" in event:
            if self.metadata_seen:
                raise ValueError("NorthGate stream contains duplicate token_data")
            self.metadata_seen = True
            self.result["token_data"] = event["token_data"]
            if "weight_version" in event:
                self.result["weight_version"] = event["weight_version"]
        if "tool_call_diagnostics" in event:
            self.result["tool_call_diagnostics"] = deepcopy(event["tool_call_diagnostics"])
        if self.api == "openai":
            self._openai(event)
        else:
            self._anthropic(event)
        return event

    def _openai(self, event: dict[str, Any]) -> None:
        if isinstance(event.get("usage"), dict):
            self.result["usage"] = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("NorthGate returned more than one completion")
            if choice.get("finish_reason") is not None:
                self.finished = True
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for key, value in delta.items():
                if key in self._OPENAI_TEXT_FIELDS:
                    if value is not None and not isinstance(value, str):
                        self.prefix_reusable = False
                elif key == "role":
                    if value not in (None, "assistant"):
                        self.prefix_reusable = False
                elif key != "tool_calls":
                    self.prefix_reusable = False
                if key == "tool_calls" and value is not None:
                    for call in value:
                        index = call["index"]
                        target = self.tools.setdefault(index, {"type": "function", "function": {}})
                        if set(call) - {"index", "id", "type", "function"}:
                            self.prefix_reusable = False
                        if "id" in call and "id" in target:
                            # The recorder overwrites IDs; fragmented IDs need
                            # explicit aggregation before this history is safe.
                            self.prefix_reusable = False
                        if call.get("type", "function") != "function":
                            self.prefix_reusable = False
                        for field, item in call.items():
                            if field == "index":
                                continue
                            if field == "function":
                                for name, part in item.items():
                                    if name not in {"name", "arguments"} or not isinstance(
                                        part, str
                                    ):
                                        self.prefix_reusable = False
                                    target["function"][name] = (
                                        target["function"].get(name, "") + part
                                        if isinstance(part, str)
                                        else part
                                    )
                            else:
                                target[field] = item
                elif isinstance(value, str) and key != "role":
                    self.assistant[key] = (self.assistant.get(key) or "") + value
                elif value is not None or key not in self.assistant:
                    self.assistant[key] = deepcopy(value)
        if self.tools:
            self.assistant["tool_calls"] = [self.tools[index] for index in sorted(self.tools)]
        self.result["choices"] = [{"message": self.assistant}]
        if self.finish_reason is not None:
            self.result["choices"][0]["finish_reason"] = self.finish_reason

    def _anthropic(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            if self.message_started or self.blocks or event["message"].get("content"):
                self.prefix_reusable = False
            self.message_started = True
            self.result.update(deepcopy(event["message"]))
            self.assistant = {"role": "assistant", "content": []}
        elif kind == "content_block_start":
            index, block = event["index"], event["content_block"]
            if (
                index in self.blocks
                or self.finished
                or block.get("type") not in {"text", "thinking", "redacted_thinking", "tool_use"}
            ):
                self.prefix_reusable = False
            self.open_blocks.add(index)
            self.blocks[index] = deepcopy(block)
        elif kind == "content_block_delta":
            index, delta = event["index"], event["delta"]
            block = self.blocks[index]
            expected = self._ANTHROPIC_DELTAS.get(delta.get("type"))
            if (
                index not in self.open_blocks
                or self.finished
                or expected is None
                or block.get("type") != expected[0]
                or set(delta) != {"type", expected[1]}
            ):
                self.prefix_reusable = False
            if delta.get("type") == "input_json_delta":
                self.arguments[index] = self.arguments.get(index, "") + delta["partial_json"]
            else:
                for field in ("text", "thinking", "signature"):
                    if field in delta:
                        block[field] = block.get(field, "") + delta[field]
        elif kind == "content_block_stop":
            index = event["index"]
            if index not in self.open_blocks:
                self.prefix_reusable = False
            self.open_blocks.discard(index)
            if index in self.arguments:
                self.blocks[index]["input"] = json.loads(self.arguments.pop(index))
        elif kind == "message_delta":
            if set(event.get("delta", {})) - {"stop_reason", "stop_sequence"}:
                self.prefix_reusable = False
            self.result.update(deepcopy(event.get("delta", {})))
            self.result.setdefault("usage", {}).update(event.get("usage", {}))
            self.finished = event.get("delta", {}).get("stop_reason") is not None
        elif kind == "message_stop":
            if not self.finished or self.arguments:
                raise ValueError("NorthGate message stream ended before completion")
            if not self.message_started or self.open_blocks:
                self.prefix_reusable = False
            self.terminal = True
        elif kind != "ping" and not (
            kind is None
            and "token_data" in event
            and not set(event) - {"token_data", "weight_version", "tool_call_diagnostics"}
        ):
            self.prefix_reusable = False
        self.assistant["content"] = [self.blocks[index] for index in sorted(self.blocks)]
        self.result["content"] = self.assistant["content"]


class NorthGateRelay:
    """Bind one rollout to one published sampler and observe its native responses."""

    def __init__(
        self,
        config: DictConfig,
        gateway: dict[str, Any],
        *,
        recorder: TrajectoryRecorder,
        metrics: RolloutMetrics,
        on_service_call: Callable[[str, Exception | None], None],
        is_validation: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        base_url = gateway.get("base_url", "")
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        ):
            raise ValueError("NorthGate gateway.base_url must be an HTTP service root")
        if not isinstance(gateway.get("model"), str) or not gateway["model"]:
            raise ValueError("NorthGate gateway.model must not be empty")
        api_key = os.getenv("NORTHGATE_API_KEY")
        if not api_key:
            raise ValueError("NORTHGATE_API_KEY is required inside the E2B sandbox")
        self._base_url = base_url.rstrip("/")
        self._model = gateway["model"]
        self._api_key = api_key
        self._recorder = recorder
        self._metrics = metrics
        self._on_service_call = on_service_call
        reuse_prefix = config.rollout.get("execution", {}).get("token_prefix_reuse", False)
        if not isinstance(reuse_prefix, bool):
            raise ValueError("rollout.execution.token_prefix_reuse must be a boolean")
        self._token_prefix_cache = TokenPrefixCache() if reuse_prefix else None
        if get_algorithm_loss_mode(config) == "score_centering":
            raise ValueError("E2B gateway trajectories do not support score_centering")
        self._flags = get_derived_sampling_flags(config)
        if self._flags["return_moe_topk_indices"]:
            raise ValueError("E2B gateway trajectories do not support R3 router_replay")
        sampling = config.validate.sampling if is_validation else config.rollout.sampling
        sampling_values = OmegaConf.to_container(sampling, resolve=True)
        if not isinstance(sampling_values, dict):
            raise ValueError("sampling configuration must be an object")
        self._sampling = {key: value for key, value in sampling_values.items() if value is not None}
        self._reasoning_effort = self._sampling.pop("reasoning_effort", None)
        if "max_tokens" in self._sampling:
            self._sampling["max_tokens"] = int(self._sampling["max_tokens"])
            if self._sampling["max_tokens"] <= 0:
                raise ValueError("sampling.max_tokens must be positive")
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=120.0)
        )
        self._call_ids = count()
        self._failed_calls: dict[tuple[str, bytes], set[str]] = {}
        self._observations: set[_CallObservation] = set()
        # One relay owns one rollout. A None entry marks an ambiguous reused ID.
        self._tool_input_orders: dict[tuple[str, str], tuple[bytes, _InputKeyOrder] | None] = {}
        recorder.set_metrics(metrics)
        self.router = APIRouter()

        @self.router.post("/v1/chat/completions")
        async def chat(request: Request):
            return await self._forward(request, "openai")

        @self.router.post("/v1/messages")
        async def messages(request: Request):
            return await self._forward(request, "anthropic")

    async def close(self) -> None:
        for observation in tuple(self._observations):
            observation.finish(
                asyncio.CancelledError("NorthGate relay closed before response completion")
            )
        self._tool_input_orders.clear()
        if self._token_prefix_cache is not None:
            self._token_prefix_cache.clear()
            self._token_prefix_cache = None
        await self._http_client.aclose()

    def _remember_tool_input_orders(self, calls: list[dict[str, Any]]) -> None:
        for call in calls:
            identity = _tool_input_identity(call)
            if identity is None:
                continue
            order = _input_key_order(call["input"])
            if order is None:
                continue
            entry = (_input_fingerprint(call["input"]), order)
            if identity not in self._tool_input_orders:
                self._tool_input_orders[identity] = entry
            elif self._tool_input_orders[identity] != entry:
                # Never guess which response an ID refers to after a collision.
                self._tool_input_orders[identity] = None

    def _restore_tool_input_orders(self, messages: list[Any]) -> None:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                identity = _tool_input_identity(block)
                entry = self._tool_input_orders.get(identity) if identity is not None else None
                if entry is not None and _input_fingerprint(block["input"]) == entry[0]:
                    # Read values from the request copy; only object key order changes.
                    block["input"] = _apply_input_key_order(block["input"], entry[1])

    def _payload(self, raw: Any, api: Literal["openai", "anthropic"]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("request body must be an object")
        if raw.get("model") not in {"northrl", self._model}:
            raise ValueError("this rollout can only call its fixed sampler")
        if not isinstance(raw.get("messages"), list) or not raw["messages"]:
            raise ValueError("messages must be a nonempty array")
        if not isinstance(raw.get("stream", False), bool):
            raise ValueError("stream must be a boolean")
        if isinstance(raw.get("n", 1), bool) or raw.get("n", 1) != 1:
            raise ValueError("training requires one completion per request")
        payload = deepcopy(raw)
        if api == "anthropic":
            _strip_anthropic_cache_control(payload)
        # Only this rollout's matched history can supply history_tokens.
        payload.pop("history_tokens", None)
        payload["model"] = self._model
        if self._reasoning_effort is not None:
            if api == "openai":
                payload["reasoning_effort"] = self._reasoning_effort
            else:
                output_config = payload.setdefault("output_config", {})
                if not isinstance(output_config, dict):
                    raise ValueError("output_config must be an object")
                output_config["effort"] = self._reasoning_effort
        if "max_tokens" in self._sampling:
            payload.pop("max_completion_tokens", None)
        payload.update(
            {
                key: value
                for key, value in self._sampling.items()
                if key in {"max_tokens", "temperature", "top_p", "top_k", "seed", "stop"}
            }
        )
        native = payload.get("sampling_params", {})
        options = payload.get("weaver_options", {})
        if not isinstance(native, dict) or not isinstance(options, dict):
            raise ValueError("sampling_params and weaver_options must be objects")
        if "sampling_params" in options:
            # Preserve the legacy sampling-object override, but use NorthGate's
            # current top-level field rather than forwarding the removed option.
            native = options.pop("sampling_params")
            if not isinstance(native, dict):
                raise ValueError("weaver_options.sampling_params must be an object")
        # Recipe values update agent settings; omitted values remain agent-controlled.
        if "max_tokens" in self._sampling:
            native.pop("max_new_tokens", None)
        payload["sampling_params"] = {**native, **self._sampling}
        options.update(self._flags)
        options.pop("num_samples", None)
        payload["weaver_options"] = options
        return payload

    async def _forward(self, request: Request, api: Literal["openai", "anthropic"]) -> Response:
        try:
            raw = await request.json()
            payload = self._payload(raw, api)
            client_model = raw["model"]
            if api == "anthropic":
                self._restore_tool_input_orders(payload["messages"])
        except (ValueError, TypeError):
            return self._error(api, 400, "Invalid request for this rollout")
        self._metrics.observe_messages(api, payload["messages"])
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_HEADERS | _AUTH_HEADERS
        }
        headers["authorization" if api == "openai" else "x-api-key"] = (
            "Bearer " if api == "openai" else ""
        ) + self._api_key
        prefix_context = None
        if self._token_prefix_cache is not None:
            prefix_context = request_context(payload, api=api, headers=headers)
            history_tokens = self._token_prefix_cache.match(
                prefix_cache_messages(payload["messages"], api=api), prefix_context
            )
            if history_tokens is not None:
                # Freeze the chosen history before any I/O or retry can yield.
                # NorthGate locates the boundary by counting the final token's
                # end marker in the rendered history; no message count is sent.
                payload["history_tokens"] = history_tokens
        observation = _CallObservation(self, api, request.url.path, payload)
        upstream: httpx.Response | None = None
        try:
            upstream = await self._request_with_retries(request, payload, headers, observation)
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in _HOP_HEADERS
            }
            if payload.get("stream", False):
                if "text/event-stream" not in upstream.headers.get("content-type", ""):
                    raise ValueError("NorthGate did not return an SSE response")
                response = StreamingResponse(
                    self._stream(upstream, api, payload, observation, client_model, prefix_context),
                    status_code=upstream.status_code,
                    headers=response_headers,
                )
                upstream = None  # The stream owns the successful response from here on.
                return response
            # A successful response is never replayed, even if reading its body fails.
            body = await upstream.aread()
            result = json.loads(body)
            observation.timing.result = result
            self._record(result, api, payload, prefix_context)
            outgoing = _with_client_model(result, client_model)
            if outgoing is not result:
                body = json.dumps(outgoing, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            observation.finish()
            return Response(body, status_code=upstream.status_code, headers=response_headers)
        except BaseException as exc:
            observation.finish(exc)
            raise
        finally:
            if upstream is not None:
                await upstream.aclose()

    async def _request_with_retries(
        self,
        request: Request,
        payload: dict[str, Any],
        headers: dict[str, str],
        observation: _CallObservation,
    ) -> httpx.Response:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _RETRY_WINDOW_SECONDS
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            observation.attempts = attempt
            observation.http_status = None
            observation.request_id = None
            upstream: httpx.Response | None = None
            retry_after: float | None = None
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("NorthGate retry budget exhausted")
                outbound = self._http_client.build_request(
                    "POST",
                    self._base_url + request.url.path,
                    json=payload,
                    headers=headers,
                    params=request.query_params,
                )
                timeouts = dict(outbound.extensions.get("timeout", {}))
                connect_timeout = timeouts.get("connect")
                timeouts["connect"] = min(
                    120.0,
                    remaining,
                    connect_timeout if connect_timeout is not None else 120.0,
                )
                outbound.extensions["timeout"] = timeouts
                # A non-streaming response may send headers only after generation.
                # Let successful requests use the existing request/agent limits.
                upstream = await self._http_client.send(outbound, stream=True)
                observation.http_status = upstream.status_code
                observation.request_id = _response_request_id(upstream.headers)
                if upstream.status_code >= 400:
                    retry_after = _retry_after_seconds(upstream.headers.get("retry-after"))
                    async with asyncio.timeout_at(deadline):
                        body = await upstream.aread()
                    try:
                        data = json.loads(body)
                    except ValueError:
                        data = body.decode("utf-8", errors="replace")
                    error = _UpstreamFailure(
                        upstream.status_code, data, request_id=observation.request_id
                    )
                    observation.remember_failure(error)
                    raise _context_error(data, payload) or error
                observation.retrying = None
                observation.upstream_activity()
                response, upstream = upstream, None
                return response
            except BaseException as exc:
                protocol_error = exc if isinstance(exc, _UpstreamFailure) else None
                retryable = (
                    protocol_error is not None
                    and protocol_error.source == "http"
                    and (
                        protocol_error.status == 429
                        or (
                            protocol_error.status is not None and 500 <= protocol_error.status < 600
                        )
                    )
                ) or (
                    (
                        upstream is None
                        or upstream.status_code == 429
                        or 500 <= upstream.status_code < 600
                    )
                    and isinstance(
                        exc, (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)
                    )
                )
                if not retryable or attempt >= _RETRY_MAX_ATTEMPTS:
                    raise
                if upstream is not None:
                    await upstream.aclose()
                    upstream = None
                remaining = deadline - loop.time()
                # With three attempts, the two backoff bases are one and two seconds.
                delay_base = float(attempt)
                delay = max(random.uniform(delay_base / 2, delay_base), retry_after or 0.0)
                if delay >= remaining:
                    raise TimeoutError(
                        "NorthGate retry delay exceeds remaining retry budget"
                    ) from exc
                try:
                    logger.warning(
                        "NorthGate request retry: %s",
                        json.dumps(
                            {
                                "path": observation.path,
                                "payload_sha256": observation.key[1].hex(),
                                "operation": observation.operation,
                                "attempt": attempt,
                                "max_attempts": _RETRY_MAX_ATTEMPTS,
                                "error_type": type(exc).__name__,
                                "http_status": observation.http_status,
                                "error_code": (
                                    str(protocol_error.code)[:256]
                                    if protocol_error is not None
                                    and protocol_error.code is not None
                                    else None
                                ),
                                "request_id": (
                                    str(observation.request_id)[:256]
                                    if observation.request_id
                                    else None
                                ),
                                "delay_seconds": round(delay, 3),
                                "remaining_seconds": round(remaining, 3),
                            },
                            sort_keys=True,
                        ),
                    )
                except Exception:
                    pass  # Diagnostics must not change retries or failure accounting.
                observation.retrying = (
                    f"HTTP {observation.http_status}"
                    if observation.http_status is not None
                    else type(exc).__name__
                )
                async with asyncio.timeout_at(deadline):
                    await asyncio.sleep(delay)
            finally:
                if upstream is not None:
                    await upstream.aclose()
        raise AssertionError("NorthGate retry loop exited without a response or failure")

    def _record(
        self,
        result: dict[str, Any],
        api: Literal["openai", "anthropic"],
        payload: dict[str, Any],
        prefix_context: str | None = None,
    ) -> None:
        if not isinstance(result, dict):
            raise ValueError("NorthGate response must be an object")
        if api == "openai":
            choices = result.get("choices")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0].get("message"), dict)
            ):
                raise ValueError("NorthGate response must contain one assistant message")
            assistant = choices[0]["message"]
            calls = assistant.get("tool_calls") or []
            prefix_complete = choices[0].get("finish_reason") in ("stop", "tool_calls")
            # NorthGate history_tokens must end with a model/template terminator
            # used to locate the history boundary. Request-defined stops do not
            # guarantee this, and OpenAI finish_reason does not identify which
            # stop was hit. Without tokenizer metadata to validate the final
            # token, conservatively skip cache writes when custom stops are
            # configured. Sampling and response recording continue normally.
            native_sampling = payload.get("sampling_params", {})
            if (
                payload.get("stop")
                or native_sampling.get("stop")
                or native_sampling.get("stop_token_ids")
            ):
                prefix_complete = False
        else:
            if not isinstance(result.get("content"), list):
                raise ValueError("NorthGate response must contain native content blocks")
            assistant = {"role": "assistant", "content": result["content"]}
            calls = [
                block
                for block in result["content"]
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            prefix_complete = result.get("stop_reason") in ("end_turn", "tool_use")
        self._metrics.record_model_result(result)
        with self._metrics.emitted_tool_calls(len(calls)):
            training = result.get("token_data")
            if not isinstance(training, dict):
                raise ValueError("NorthGate response is missing token_data")
            for name in ("prompt_tokens", "response_tokens", "response_logprobs"):
                if not isinstance(training.get(name), list):
                    raise ValueError(f"NorthGate token_data.{name} must be an array")
            for name in ("prompt_tokens", "response_tokens"):
                # bool is an int subclass; token IDs must be actual integers.
                # pylint: disable-next=unidiomatic-typecheck
                if any(type(token) is not int or token < 0 for token in training[name]):
                    raise ValueError(
                        f"NorthGate token_data.{name} must contain nonnegative integer token IDs"
                    )
            history_tokens = payload.get("history_tokens")
            if history_tokens is not None:
                if training["prompt_tokens"][: len(history_tokens)] != history_tokens:
                    raise ValueError("NorthGate did not preserve the requested history_tokens")
            old_logprobs = training.get("response_old_logprobs")
            if old_logprobs is None:
                if self._flags["return_old_logprob"]:
                    raise ValueError(
                        "NorthGate response is missing requested response_old_logprobs"
                    )
                old_logprobs = training["response_logprobs"]
            elif not isinstance(old_logprobs, list):
                raise ValueError("NorthGate response_old_logprobs must be an array")
            mask = training.get("response_sampling_masks")
            if training.get("response_sampling_mask_ref") is not None:
                mask = {"sampling_mask_ref": training["response_sampling_mask_ref"]}
                if training.get("response_sampling_mask_slice") is not None:
                    mask["sampling_mask_slice"] = training["response_sampling_mask_slice"]
            if self._flags["return_sampling_mask"] and mask is None:
                raise ValueError("NorthGate response is missing requested sampling masks")
            self._recorder.record_inference_result(
                {
                    **result,
                    "northrl_train": {**training, "response_old_logprobs": old_logprobs},
                    "sampling_mask": mask,
                },
                message_format=api,
                message_system=payload.get("system"),
                tools=payload.get("tools"),
                messages=[*payload["messages"], assistant],
                # OpenAI cannot distinguish a custom stop from the template's
                # normal terminator. Mark completion as uncertain for consumers.
                truncated=bool(
                    payload.get("stop")
                    or payload.get("stop_sequences")
                    or payload.get("sampling_params", {}).get("stop")
                    or payload.get("sampling_params", {}).get("stop_token_ids")
                    or result.get("stop_reason") == "stop_sequence"
                ),
            )
        if api == "anthropic":
            self._remember_tool_input_orders(calls)
        if self._token_prefix_cache is not None and prefix_context is not None and prefix_complete:
            self._token_prefix_cache.remember(
                prefix_cache_messages([*payload["messages"], assistant], api=api),
                training["prompt_tokens"] + training["response_tokens"],
                prefix_context,
            )

    async def _stream(
        self,
        upstream: httpx.Response,
        api: Literal["openai", "anthropic"],
        payload: dict[str, Any],
        observation: _CallObservation,
        client_model: str,
        prefix_context: str | None = None,
    ):
        observation.http_status = upstream.status_code
        observation.request_id = _response_request_id(upstream.headers)
        reader = _StreamObservation(
            api, http_status=upstream.status_code, request_id=observation.request_id
        )
        buffer = b""
        try:
            async for chunk in upstream.aiter_bytes():
                observation.upstream_activity()
                buffer += chunk
                while match := _FRAME_END.search(buffer):
                    frame, buffer = buffer[: match.end()], buffer[match.end() :]
                    try:
                        event = reader.observe(frame)
                    except _UpstreamFailure as exc:
                        observation.remember_failure(exc)
                        raise _context_error(exc.body, payload) or exc
                    frame = _client_sse_frame(frame, event, api, client_model)
                    if reader.terminal:
                        observation.timing.result = reader.result
                        self._record(
                            reader.result,
                            api,
                            payload,
                            prefix_context if reader.prefix_reusable else None,
                        )
                        observation.finish()
                        observation.response_emitted = True
                        yield frame
                        return
                    observation.response_emitted = True
                    yield frame
            raise ConnectionError("NorthGate stream ended before its terminal event")
        except BaseException as exc:
            observation.finish(exc)
            raise
        finally:
            if not observation.finished:
                observation.finish(ConnectionError("NorthGate response stream was closed"))
            await upstream.aclose()

    @staticmethod
    def _error(api: Literal["openai", "anthropic"], status: int, message: str) -> JSONResponse:
        kind = (
            "invalid_request_error"
            if status == 400
            else "api_error" if api == "anthropic" else "server_error"
        )
        error: dict[str, Any] = {"error": {"type": kind, "message": message}}
        if api == "anthropic":
            error["type"] = "error"
        return JSONResponse(error, status_code=status)
