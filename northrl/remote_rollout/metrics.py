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

"""Per-rollout model and tool observations, independent of training trajectories."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .tool_errors import (
    TOOL_ERROR_CATEGORIES,
    ToolCallContext,
    classify_tool_result,
    is_tool_timeout,
    tool_call_context,
)

# Shared cadence for sandbox heartbeats and driver monitoring logs.
LIVE_METRICS_INTERVAL_SECONDS = 30.0

# NorthGate reasons are deduplicated per response, not counts of invalid calls.
TOOL_FORMAT_REASONS = (
    "incomplete_tool_call",
    "invalid_arguments_json",
    "missing_tool_name",
    "invalid_tool_call_structure",
    "arguments_not_object",
    "duplicate_parameter",
    "undeclared_tool",
)

# Cumulative histogram buckets are mergeable across requests and sandboxes.
REQUEST_DURATION_BUCKETS_SECONDS = (1, 5, 10, 30, 60, 120, 300, 600, 1800)
REQUEST_DURATION_BUCKET_METRICS = {
    **{
        f"model_request_seconds_bucket_le_{bound}": float(bound)
        for bound in REQUEST_DURATION_BUCKETS_SECONDS
    },
    "model_request_seconds_bucket_le_inf": math.inf,
}
MODEL_REQUEST_GAUGES = frozenset(
    ("model_request_inflight", "model_request_oldest_inflight_seconds")
)
SLOW_MODEL_REQUEST_SECONDS = 60.0

logger = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    """A logical generate call; its result supplies optional token counts."""

    request_seq: int
    result: dict[str, Any] | None = None


class RolloutMetrics:
    """Accumulate observations once, even when requests repeat their full history.

    ``model_turns`` counts successful backend results, including results which
    cannot be converted for the harness. ``max_context_tokens`` is the largest
    prompt plus generated response. Tool results are observed receipts, not proof
    that every emitted call was executed. No JSON Schema validation is implied.
    """

    def __init__(
        self,
        *,
        rollout_id: str | None = None,
        attempt_id: str | None = None,
        observe_tools: bool = True,
    ) -> None:
        self._lock = RLock()
        self._observe_tools = observe_tools
        self._request_context = {
            key: value
            for key, value in (("rollout_id", rollout_id), ("attempt_id", attempt_id))
            if value is not None
        }
        self._values = dict.fromkeys(
            (
                "runtime_revision",
                "model_turns",
                "generated_tokens",
                "max_context_tokens",
                "trace_segments",
                "tool_calls_detected",
                "tool_calls_emitted",
                "tool_calls_invalid",
                "tool_results_observed",
                "tool_result_errors",
                "tool_timeouts",
                "tool_calls_unobserved",
                "tool_results_unassigned",
                "runtime_tool_observations",
                "runtime_ambiguous_tool_ids",
                "model_request_usage_count",
                *(
                    f"request_stat_{name}_{stat}"
                    for name in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "duration_seconds",
                    )
                    for stat in ("count", "sum", "min", "max")
                ),
                *(f"tool_error_{category}" for category in TOOL_ERROR_CATEGORIES),
                "tool_format_error_turns",
                "tool_format_unknown_turns",
                "tool_call_counts_unknown_turns",
                *(f"tool_format_reason_{reason}_turns" for reason in TOOL_FORMAT_REASONS),
                "tool_arguments_json_errors",
                "tool_arguments_type_errors",
                "model_requests_started",
                "model_requests_completed",
                "model_requests_success",
                "model_requests_error",
                "model_requests_timeout",
                "model_requests_cancelled",
                "model_request_seconds_sum",
                "model_request_seconds_max",
                *REQUEST_DURATION_BUCKET_METRICS,
            ),
            0.0,
        )
        self._tool_calls: dict[str, ToolCallContext] = {}
        self._seen_results: set[str] = set()
        self._seen_error_results: set[str] = set()
        self._seen_timeouts: set[str] = set()
        self._emitted_ids: set[str] = set()
        self._tool_segments: dict[str, int] = {}
        self._known_segments: set[int] = set()
        self._ambiguous_ids: set[str] = set()
        self._ambiguous_segments: set[int] = set()
        self._request_sequence = 0
        self._inflight_requests: dict[int, float] = {}
        self._request_observer: Callable[[dict[str, Any]], None] | None = None
        self._request_observer_error_reported = False

    def snapshot(self) -> dict[str, float]:
        """Return a consistent copy, including zero-valued counters."""
        with self._lock:
            snapshot = dict(self._values)
            snapshot["runtime_tool_observations"] = float(self._observe_tools)
            snapshot["runtime_ambiguous_tool_ids"] = float(len(self._ambiguous_ids))
            snapshot["tool_calls_unobserved"] = float(len(self._emitted_ids - self._seen_results))
            snapshot["tool_results_unassigned"] = float(
                len(self._seen_results - self._tool_segments.keys())
            )
            snapshot["model_request_inflight"] = float(len(self._inflight_requests))
            snapshot["model_request_oldest_inflight_seconds"] = (
                max(0.0, time.perf_counter() - min(self._inflight_requests.values()))
                if self._inflight_requests
                else 0.0
            )
            return snapshot

    def set_request_observer(self, observer: Callable[[dict[str, Any]], None] | None) -> None:
        """Set a best-effort sink for completed request records, without model content."""
        with self._lock:
            self._request_observer = observer
            self._request_observer_error_reported = False

    @contextmanager
    def track_request(self, api: str) -> Iterator[RequestTiming]:
        """Time generate including backend queueing and retries, even on cancellation.

        The caller assigns ``request.result`` only after generate returns. Response
        conversion and trajectory recording belong outside this context. This is
        client-observed latency, not GPU execution time or time to first token.
        """
        started_at = time.perf_counter()
        with self._lock:
            self._values["runtime_revision"] += 1
            self._request_sequence += 1
            request = RequestTiming(self._request_sequence)
            self._inflight_requests[request.request_seq] = started_at
            self._values["model_requests_started"] += 1
        status = "success"
        error_type = None
        try:
            yield request
        except BaseException as exc:
            error_type = type(exc).__name__
            if isinstance(exc, asyncio.CancelledError):
                status = "cancelled"
            elif _is_request_timeout(exc):
                status = "timeout"
            else:
                status = "error"
            raise
        finally:
            duration = max(0.0, time.perf_counter() - started_at)
            with self._lock:
                self._values["runtime_revision"] += 1
                self._inflight_requests.pop(request.request_seq, None)
                self._values["model_requests_completed"] += 1
                self._values[f"model_requests_{status}"] += 1
                self._values["model_request_seconds_sum"] += duration
                self._values["model_request_seconds_max"] = max(
                    self._values["model_request_seconds_max"], duration
                )
                for name, bound in REQUEST_DURATION_BUCKET_METRICS.items():
                    self._values[name] += int(duration <= bound)
                observer = self._request_observer
            prompt_tokens, completion_tokens = _request_token_counts(request.result)
            self._record_request_stats(duration, request.result)
            record = {
                **self._request_context,
                "api": api,
                "request_seq": request.request_seq,
                "status": status,
                "duration_seconds": duration,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            if error_type is not None:
                record["error_type"] = error_type
            if duration > SLOW_MODEL_REQUEST_SECONDS:
                logger.warning(
                    "Slow model request: rollout_id=%s attempt_id=%s "
                    "api=%s request_seq=%d status=%s duration_seconds=%.3f",
                    self._request_context.get("rollout_id", "-"),
                    self._request_context.get("attempt_id", "-"),
                    api,
                    request.request_seq,
                    status,
                    duration,
                )
            if observer is not None:
                try:
                    observer(record)
                except Exception:  # pylint: disable=broad-exception-caught
                    with self._lock:
                        report_error = not self._request_observer_error_reported
                        self._request_observer_error_reported = True
                    if report_error:
                        logger.warning("Could not record model request timing", exc_info=True)

    def _record_request_stats(self, duration: float, result: dict[str, Any] | None) -> None:
        values = {"duration_seconds": duration}
        prompt, output = _request_token_counts(result)
        if prompt is not None:
            values["input_tokens"] = prompt
        if output is not None:
            values["output_tokens"] = output
        if prompt is not None and output is not None:
            values["total_tokens"] = prompt + output
        with self._lock:
            self._values["runtime_revision"] += 1
            self._values["model_request_usage_count"] += int("total_tokens" in values)
            for name, value in values.items():
                prefix = f"request_stat_{name}_"
                count = self._values[prefix + "count"]
                self._values[prefix + "count"] = count + 1
                self._values[prefix + "sum"] += value
                self._values[prefix + "min"] = (
                    min(self._values[prefix + "min"], value) if count else value
                )
                self._values[prefix + "max"] = (
                    max(self._values[prefix + "max"], value) if count else value
                )

    def enable_tool_observations(self) -> None:
        self._observe_tools = True

    def assign_single_segment(self) -> None:
        """A direct, untruncated trajectory owns its execution's emitted calls."""
        with self._lock:
            self._known_segments.add(0)
            self._tool_segments.update({tool_id: 0 for tool_id in self._emitted_ids})

    def record_emitted_ids(self, result: dict[str, Any], segment_index: int | None = None) -> None:
        """Register successfully delivered call IDs, optionally tied to saved tokens."""
        calls = _observed_tool_calls(result)
        with self._lock:
            self._values["runtime_revision"] += 1
            if segment_index is not None:
                self._known_segments.add(segment_index)
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, dict) or not _non_empty_string(call.get("id")):
                    continue
                self._observe_tools = True
                tool_id = call["id"]
                if tool_id in self._emitted_ids:
                    self._ambiguous_ids.add(tool_id)
                    previous_segment = self._tool_segments.pop(tool_id, None)
                    if previous_segment is not None:
                        self._ambiguous_segments.add(previous_segment)
                    if segment_index is not None:
                        self._ambiguous_segments.add(segment_index)
                self._emitted_ids.add(tool_id)
                function = call.get("function") or {}
                if isinstance(function, dict):
                    self._tool_calls[tool_id] = tool_call_context(
                        function.get("name"), function.get("arguments")
                    )
                if segment_index is not None and tool_id not in self._ambiguous_ids:
                    self._tool_segments[tool_id] = segment_index

    def segment_snapshot(self, index: int) -> dict[str, int]:
        """Only emitted IDs establish segment ownership, never a parent's totals."""
        with self._lock:
            if (
                index not in self._known_segments
                or index in self._ambiguous_segments
                or not self._observe_tools
            ):
                return {}
            tool_ids = {
                tool_id for tool_id, segment in self._tool_segments.items() if segment == index
            }
            return {
                "tool_calls": len(tool_ids),
                "tool_timeouts": len(tool_ids & self._seen_timeouts),
                "tool_results": len(tool_ids & self._seen_results),
            }

    def runtime_snapshot(self) -> dict[str, Any]:
        return runtime_metrics_from_snapshot(self.snapshot())

    def set_trace_segments(self, count: int) -> None:
        """Set the recorded sequence count within this rollout (not model turns)."""
        with self._lock:
            self._values["trace_segments"] = float(count)

    @contextmanager
    def emitted_tool_calls(self, count: int) -> Iterator[None]:
        """Include new calls in a checkpoint and roll back if recording fails.

        The recorder may read snapshot() from this thread while it writes its
        checkpoint. Other threads see only the committed or rolled-back state.
        Keep the synchronous recorder call inside this context, with no await.
        """
        with self._lock:
            self._values["runtime_revision"] += 1
            self._values["tool_calls_emitted"] += count
            previous_ids = set(self._emitted_ids)
            previous_segments = dict(self._tool_segments)
            previous_known = set(self._known_segments)
            previous_ambiguous_ids = set(self._ambiguous_ids)
            previous_ambiguous_segments = set(self._ambiguous_segments)
            try:
                yield
            except BaseException:
                self._values["runtime_revision"] += 1
                self._values["tool_calls_emitted"] -= count
                self._emitted_ids = previous_ids
                self._tool_segments = previous_segments
                self._known_segments = previous_known
                self._ambiguous_ids = previous_ambiguous_ids
                self._ambiguous_segments = previous_ambiguous_segments
                raise

    def record_model_result(self, result: dict[str, Any]) -> None:
        """Observe generated output before provider conversion can reject it."""
        prompt_tokens, generated_tokens = _request_token_counts(result)
        prompt_tokens = prompt_tokens or 0.0
        generated_tokens = generated_tokens or 0.0
        diagnostics = result.get("tool_call_diagnostics")
        northgate_diagnostics = isinstance(diagnostics, dict)
        counts_unknown = False
        reasons: list[Any] = []
        if isinstance(diagnostics, dict):
            # These counts cover raw candidates, including repaired or discarded
            # calls. Native output cannot reconstruct unknown (null) counts.
            detected = _count(diagnostics.get("detected"))
            invalid = _count(diagnostics.get("invalid"))
            format_checked = diagnostics.get("checked") is True
            format_error = diagnostics.get("malformed") is True or invalid > 0
            counts_unknown = any(
                _optional_count(diagnostics.get(name)) is None for name in ("detected", "invalid")
            )
            if isinstance(diagnostics.get("reasons"), list):
                reasons = diagnostics["reasons"]
        else:
            # Older providers and saved responses can still use the original field name.
            diagnostics = result.get(
                "northrl_parse_diagnostics", result.get("nexrl_parse_diagnostics")
            )
            diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
            detected = _count(diagnostics.get("tool_calls_detected"))
            invalid = _count(diagnostics.get("tool_calls_invalid"))
            format_checked = diagnostics.get("tool_format_checked") == 1
            format_error = bool(diagnostics.get("tool_format_error")) or invalid > 0
        json_errors = 0
        type_errors = 0
        calls = _observed_tool_calls(result)
        if not isinstance(calls, list):
            if not northgate_diagnostics:
                detected = max(detected, 1)
                invalid += 1
            format_error = True
            calls = []
        elif not northgate_diagnostics:
            detected = max(detected, len(calls))
        for call in calls:
            malformed = not isinstance(call, dict)
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                malformed = True
            else:
                if not _non_empty_string(function.get("name")):
                    malformed = True
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, RecursionError):
                        json_errors += 1
                        malformed = True
                        # JSON syntax failure is distinct from a valid JSON scalar.
                        arguments = {}
                if not isinstance(arguments, dict):
                    type_errors += 1
                    malformed = True
            if isinstance(call, dict) and not _non_empty_string(call.get("id")):
                malformed = True
            if malformed:
                if not northgate_diagnostics:
                    invalid += 1
                format_error = True
        with self._lock:
            self._values["runtime_revision"] += 1
            self._values["model_turns"] += 1
            self._values["generated_tokens"] += generated_tokens
            self._values["max_context_tokens"] = max(
                self._values["max_context_tokens"], prompt_tokens + generated_tokens
            )
            self._values["tool_calls_detected"] += detected
            self._values["tool_calls_invalid"] += invalid
            self._values["tool_format_error_turns"] += int(format_error)
            # Structural checks can prove an error even when the parser cannot
            # fully check its raw syntax; otherwise an unchecked turn is unknown.
            self._values["tool_format_unknown_turns"] += int(
                not format_error and not format_checked
            )
            self._values["tool_call_counts_unknown_turns"] += int(counts_unknown)
            for reason in TOOL_FORMAT_REASONS:
                self._values[f"tool_format_reason_{reason}_turns"] += int(reason in reasons)
            self._values["tool_arguments_json_errors"] += json_errors
            self._values["tool_arguments_type_errors"] += type_errors

    def observe_messages(self, message_format: str, messages: Sequence[Any]) -> None:
        """Observe tool results in request history before inference may fail.

        Histories can repeat or compact. Retain seen IDs for the whole rollout.
        An error flag observed later is counted once without recounting the result.
        """
        calls: dict[str, ToolCallContext] = {}
        results: list[tuple[str, dict[str, Any]]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if message_format == "openai":
                if role == "assistant" and isinstance(message.get("tool_calls"), list):
                    for call in message["tool_calls"]:
                        if not isinstance(call, dict) or not _non_empty_string(call.get("id")):
                            continue
                        function = call.get("function")
                        if isinstance(function, dict):
                            calls[call["id"]] = tool_call_context(
                                function.get("name"), function.get("arguments")
                            )
                elif role == "tool" and _non_empty_string(message.get("tool_call_id")):
                    results.append((message["tool_call_id"], message))
            elif message_format == "anthropic" and role in ("assistant", "user"):
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if role == "assistant" and block.get("type") == "tool_use":
                        if _non_empty_string(block.get("id")):
                            calls[block["id"]] = tool_call_context(
                                block.get("name"), block.get("input")
                            )
                    elif role == "user" and block.get("type") == "tool_result":
                        if _non_empty_string(block.get("tool_use_id")):
                            results.append((block["tool_use_id"], block))
        with self._lock:
            self._values["runtime_revision"] += 1
            # Keep call metadata after history compaction, but never raw commands.
            self._tool_calls.update(calls)
            if calls or results:
                self._observe_tools = True
            for tool_id, result in results:
                if tool_id not in self._seen_results:
                    self._seen_results.add(tool_id)
                    self._values["tool_results_observed"] += 1
                if tool_id not in self._seen_timeouts and is_tool_timeout(result):
                    self._seen_timeouts.add(tool_id)
                    self._values["tool_timeouts"] += 1
                if tool_id in self._seen_error_results:
                    continue
                context = self._tool_calls.get(tool_id) or tool_call_context(
                    result.get("name"), None
                )
                category = classify_tool_result(result, context)
                if category is not None:
                    self._seen_error_results.add(tool_id)
                    self._values["tool_result_errors"] += 1
                    self._values[f"tool_error_{category}"] += 1


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _request_token_counts(result: dict[str, Any] | None) -> tuple[float | None, float | None]:
    result = result if isinstance(result, dict) else {}
    usage = result.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt = _optional_count(usage.get("prompt_tokens", usage.get("input_tokens")))
    output = _optional_count(usage.get("completion_tokens", usage.get("output_tokens")))
    training = result.get(
        "token_data", result.get("northrl_train", result.get("nexrl_train", result))
    )
    if isinstance(training, dict):
        if prompt is None and isinstance(training.get("prompt_tokens"), list):
            prompt = float(len(training["prompt_tokens"]))
        if output is None and isinstance(training.get("response_tokens"), list):
            output = float(len(training["response_tokens"]))
    return prompt, output


def _observed_tool_calls(result: dict[str, Any]) -> Any:
    """Read native tool receipts for metrics without changing provider messages."""
    if isinstance(result.get("content"), list):
        return [
            {
                "id": block.get("id"),
                "function": {"name": block.get("name"), "arguments": block.get("input")},
            }
            for block in result["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
    choices = result.get("choices") or []
    message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
    return (message.get("tool_calls") or []) if isinstance(message, dict) else []


def _count(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return 0.0


def _optional_count(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


def _is_request_timeout(exc: BaseException) -> bool:
    """Recognize transport and polling timeouts without requiring optional SDKs."""
    if (
        isinstance(exc, TimeoutError)
        or getattr(exc, "status", None) == 504
        or getattr(exc, "code", None) in {"timeout", "timeout_error"}
    ):
        return True
    try:
        from httpx import TimeoutException
    except ImportError:
        pass
    else:
        if isinstance(exc, TimeoutException):
            return True
    try:
        from weaver import WeaverAPIError
    except ImportError:
        return False
    return isinstance(exc, WeaverAPIError) and (exc.status_code == 504 or exc.code == "timeout")


def runtime_metrics_from_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    """Convert wire-compatible numeric counters into the trainer's metadata contract.

    Absent telemetry stays absent; old artifacts do not acquire invented zeros.
    A RolloutMetrics instance belongs to exactly one rollout execution attempt.
    """
    names = {
        "model_requests_started": "model_requests",
        "model_requests_completed": "model_requests_completed",
        "model_requests_timeout": "model_request_timeouts",
        "model_request_usage_count": "request_usage_count",
    }
    if values.get("runtime_tool_observations") == 1:
        names.update(
            {
                "tool_calls_emitted": "tool_calls",
                "tool_results_observed": "tool_results",
                "tool_result_errors": "tool_errors",
                "tool_timeouts": "tool_timeouts",
                "tool_calls_unobserved": "tool_unobserved",
                "tool_results_unassigned": "tool_unassigned",
            }
        )
    result = {target: values[source] for source, target in names.items() if source in values}
    if values.get("runtime_ambiguous_tool_ids", 0) > 0:
        result.pop("tool_unobserved", None)
    stats = {}
    for name in ("input_tokens", "output_tokens", "total_tokens", "duration_seconds"):
        fields = {
            stat: values[f"request_stat_{name}_{stat}"]
            for stat in ("count", "sum", "min", "max")
            if f"request_stat_{name}_{stat}" in values
        }
        if len(fields) == 4 and fields["count"] > 0:
            stats[name] = fields
    if stats:
        result["request_stats"] = stats
    if values.get("runtime_tool_observations") == 1:
        result["tool_error_counts"] = {
            category: values[f"tool_error_{category}"]
            for category in TOOL_ERROR_CATEGORIES
            if f"tool_error_{category}" in values
        }
        parse_names = {
            "calls_detected": "tool_calls_detected",
            "calls_invalid": "tool_calls_invalid",
            "format_error_turn": "tool_format_error_turns",
            "format_unknown_turn": "tool_format_unknown_turns",
            "call_counts_unknown_turn": "tool_call_counts_unknown_turns",
            **{
                f"format_reason_{reason}_turn": f"tool_format_reason_{reason}_turns"
                for reason in TOOL_FORMAT_REASONS
            },
            "arguments_json_error": "tool_arguments_json_errors",
            "arguments_type_error": "tool_arguments_type_errors",
        }
        result["tool_parse_counts"] = {
            key: values[source] for key, source in parse_names.items() if source in values
        }
    return result
