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

"""Compact rollout records, gzip storage and bounded downloads.

Only token IDs are inherited from an earlier segment. Response positions and
both probability vectors always belong to the current segment: an inherited
token can be a response in one segment and context in another.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import math
import os
import threading
import time
import zlib
from array import array
from collections import deque
from collections.abc import Awaitable, Callable, MutableSequence, Sequence
from contextvars import copy_context
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .contracts import (
    AgentResult,
    AgentStageResult,
    RemoteRolloutResult,
    RemoteRolloutTrace,
    RemoteSegment,
    validate_turn_annotations,
)

# Compact records and training-array conversion.


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ResponseLogProbsRef(_WireModel):
    """Reference only this segment's response probabilities, never another segment."""

    ref: Literal["response_log_probs"]


class CompactSegment(_WireModel):
    """A token prefix reference and independently owned response supervision.

    Spans are ordered, disjoint, half-open intervals in *token* coordinates.
    All positions outside them have loss_mask=0 and both log probabilities=0.
    Sampling masks retain their existing response-order representation.
    """

    base_segment: int | None = Field(ge=0)
    prefix_len: int = Field(ge=0)
    suffix_tokens: list[int]
    response_spans: list[list[int]] = Field(min_length=1)
    response_log_probs: list[float]
    old_response_log_probs: list[float] | ResponseLogProbsRef
    sampling_mask: list[list[int]] | dict[str, JsonValue] | None = None
    segment_metrics: dict[str, int] = Field(default_factory=dict)
    turn_annotations: list[dict[str, JsonValue]] = Field(default_factory=list)

    @property
    def token_count(self) -> int:
        return self.prefix_len + len(self.suffix_tokens)

    @model_validator(mode="after")
    def _validate_supervision(self) -> Self:
        if (self.base_segment is None) != (self.prefix_len == 0):
            raise ValueError("a nonempty prefix requires a base_segment; an empty prefix has none")
        if not self.token_count:
            raise ValueError("a segment must contain tokens")
        if any(token < 0 for token in self.suffix_tokens):
            raise ValueError("suffix_tokens must contain non-negative token IDs")

        previous_stop = 1
        response_count = 0
        for span in self.response_spans:
            if len(span) != 2:
                raise ValueError("response_spans entries must contain exactly [start, stop]")
            start, stop = span
            if not 1 <= start < stop <= self.token_count:
                raise ValueError("response_spans must be nonempty, in bounds, and exclude token 0")
            if start < previous_stop:
                raise ValueError("response_spans must be ordered and non-overlapping")
            previous_stop = stop
            response_count += stop - start
        if len(self.response_log_probs) != response_count:
            raise ValueError("response_log_probs must align with response_spans")
        if isinstance(self.old_response_log_probs, list) and (
            len(self.old_response_log_probs) != response_count
        ):
            raise ValueError("old_response_log_probs must align with response_spans")

        validate_turn_annotations(self.turn_annotations)
        for annotation in self.turn_annotations:
            start, stop = cast(list[int], annotation["response_span"])
            if not any(left <= start < stop <= right for left, right in self.response_spans):
                raise ValueError("turn_annotations spans must contain only response tokens")

        sampling_mask = self.sampling_mask
        if isinstance(sampling_mask, list):
            if len(sampling_mask) != response_count:
                raise ValueError("sampling_mask must have one entry for each response token")
            # Pylint does not narrow this Pydantic union inside a comprehension.
            # pylint: disable-next=not-an-iterable
            if any(not row or any(token < 0 for token in row) for row in sampling_mask):
                raise ValueError("sampling_mask rows must contain non-negative candidate tokens")
        elif isinstance(sampling_mask, dict):
            from ..utils.sampling_mask_refs import mask_parts

            try:
                count = sum(
                    part["sampling_mask_slice"]["stop"] - part["sampling_mask_slice"]["start"]
                    for part in mask_parts(sampling_mask)
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid sampling_mask reference") from exc
            if count != response_count:
                raise ValueError("sampling_mask references must align with response_spans")
        return self


def _resolve_token_at(
    segments: Sequence[CompactSegment],
    segment_index: int,
    position: int,
    cache: dict[tuple[int, int], int],
) -> int:
    """Resolve one queried position and compress its path through prefix refs.

    The cache is local to one graph validation. Only positions inspected for
    inline sampling masks are stored, never a segment's entire context.
    """

    path: list[tuple[int, int]] = []
    while True:
        key = (segment_index, position)
        if key in cache:
            token = cache[key]
            break
        path.append(key)
        source = segments[segment_index]
        if position >= source.prefix_len:
            token = source.suffix_tokens[position - source.prefix_len]
            break
        # References have already been validated in topological order.
        assert source.base_segment is not None
        segment_index = source.base_segment
    for key in path:
        cache[key] = token
    return token


def _validate_segment_graph(segments: Sequence[CompactSegment]) -> None:
    """Validate references and queried response tokens without dense expansion."""

    token_cache: dict[tuple[int, int], int] = {}
    for index, segment in enumerate(segments):
        base = segment.base_segment
        if base is not None:
            if base >= index:
                raise ValueError("base_segment must reference an earlier segment")
            if segment.prefix_len > segments[base].token_count:
                raise ValueError("prefix_len exceeds the referenced segment's token count")

        if isinstance(segment.sampling_mask, list):
            response_positions = (
                position
                for start, stop in segment.response_spans
                for position in range(start, stop)
            )
            for position, candidates in zip(response_positions, segment.sampling_mask):
                token = _resolve_token_at(segments, index, position, token_cache)
                if token not in candidates:
                    raise ValueError(f"sampling_mask must contain sampled token {token}")


class _Envelope(_WireModel):
    # No default: a legacy or unversioned record must never be accepted as v2.
    schema_version: Literal[2]
    rollout_id: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_version(cls, value: object) -> object:
        # bool and 2.0 are not wire version integers.
        if type(value) is not int:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("schema_version must be the integer 2")
        return value

    @field_validator("rollout_id")
    @classmethod
    def _nonempty_rollout_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rollout_id must be a non-empty string")
        return value


class _SegmentEnvelope(_Envelope):
    segments: list[CompactSegment]
    message_format: Literal["openai", "anthropic"] | None = None
    message_system: JsonValue | None = None
    tools: list[dict[str, JsonValue]] | None = None
    messages: list[dict[str, JsonValue]] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        _validate_segment_graph(self.segments)
        return self


class CompactRolloutTrace(_SegmentEnvelope):
    """A self-contained partial trace; no references to an older snapshot."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    segments: list[CompactSegment] = Field(min_length=1)


class CompactRolloutResult(_SegmentEnvelope):
    """Completed outcome with compact segments and unchanged task metadata."""

    status: Literal["success", "task_failed", "env_error"] = "success"
    error_type: str | None = None
    error_message: str | None = None
    termination_reason: Literal["context_length_exceeded"] | None = None
    length_details: dict[str, int] = Field(default_factory=dict)
    segments: list[CompactSegment] = Field(default_factory=list)
    reward: float

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if self.status != "success" and self.reward != 0.0:
            raise ValueError("a failed rollout must have reward 0.0")
        return self


class CompactAgentStageResult(_Envelope):
    """Agent/evaluator handoff with an optional nested v2 trace."""

    attempt_id: str
    phase: Literal["agent_completed"] = "agent_completed"
    workspace: str
    agent_result: AgentResult
    trace: CompactRolloutTrace | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    termination_reason: Literal["context_length_exceeded"] | None = None
    length_details: dict[str, int] = Field(default_factory=dict)

    @field_validator("attempt_id")
    @classmethod
    def _nonempty_attempt_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("attempt_id must be a non-empty string")
        return value

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if self.trace is not None and self.trace.rollout_id != self.rollout_id:
            raise ValueError("agent stage trace rollout_id must match rollout_id")
        if not Path(self.workspace).is_absolute():
            raise ValueError("agent stage workspace must be an absolute path")
        return self


def _same_probabilities(first: list[float], second: list[float]) -> bool:
    # Preserve signed zero as well as every finite float value when selecting a ref.
    return len(first) == len(second) and all(
        left == right and (left != 0.0 or math.copysign(1.0, left) == math.copysign(1.0, right))
        for left, right in zip(first, second)
    )


def encode_segment(
    segment: RemoteSegment,
    previous_segment: RemoteSegment | None = None,
    *,
    base_segment: int | None = None,
) -> CompactSegment:
    """Encode one segment, comparing tokens only with the preceding segment.

    Callers can cache finalized segments and re-encode only their mutable tail.
    ``base_segment`` is the preceding segment's index in the same snapshot.
    """

    if (previous_segment is None) != (base_segment is None):
        raise ValueError("previous_segment and its base_segment index must be supplied together")
    prefix_len = 0
    if previous_segment is not None:
        for current, previous in zip(segment.tokens, previous_segment.tokens):
            if current != previous:
                break
            prefix_len += 1

    spans: list[list[int]] = []
    log_probs: list[float] = []
    old_log_probs: list[float] = []
    for position, selected in enumerate(segment.loss_mask):
        if not selected:
            continue
        if spans and spans[-1][1] == position:
            spans[-1][1] = position + 1
        else:
            spans.append([position, position + 1])
        log_probs.append(segment.log_probs[position])
        old_log_probs.append(segment.old_log_probs[position])

    return CompactSegment(
        base_segment=base_segment if prefix_len else None,
        prefix_len=prefix_len,
        suffix_tokens=segment.tokens[prefix_len:],
        response_spans=spans,
        response_log_probs=log_probs,
        old_response_log_probs=(
            ResponseLogProbsRef(ref="response_log_probs")
            if _same_probabilities(log_probs, old_log_probs)
            else old_log_probs
        ),
        sampling_mask=deepcopy(segment.sampling_mask),
        segment_metrics=dict(segment.segment_metrics),
        turn_annotations=deepcopy(segment.turn_annotations),
    )


def encode_segments(segments: Sequence[RemoteSegment]) -> list[CompactSegment]:
    return [
        encode_segment(
            segment,
            segments[index - 1] if index else None,
            base_segment=index - 1 if index else None,
        )
        for index, segment in enumerate(segments)
    ]


def decode_segments(
    segments: Sequence[CompactSegment], *, packed: bool = False
) -> list[RemoteSegment]:
    """Expand validated wire segments into independently owned numeric sequences.

    The driver can retain packed buffers; public/runtime callers keep the dense
    list contract and its validators by default.
    """

    _validate_segment_graph(segments)
    decoded: list[RemoteSegment] = []
    for segment in segments:
        tokens: MutableSequence[int] = (
            decoded[segment.base_segment].tokens[: segment.prefix_len]
            if segment.base_segment is not None
            else (array("q") if packed else [])
        )
        tokens.extend(segment.suffix_tokens)
        loss_mask: MutableSequence[int] = (array("b", [0]) if packed else [0]) * len(tokens)
        log_probs: MutableSequence[float] = (array("d", [0.0]) if packed else [0.0]) * len(tokens)
        old_log_probs: MutableSequence[float] = (array("d", [0.0]) if packed else [0.0]) * len(
            tokens
        )
        old_responses = (
            segment.old_response_log_probs
            if isinstance(segment.old_response_log_probs, list)
            else segment.response_log_probs
        )
        offset = 0
        for start, stop in segment.response_spans:
            count = stop - start
            selected = segment.response_log_probs[offset : offset + count]
            old_selected = old_responses[offset : offset + count]
            loss_mask[start:stop] = (array("b", [1]) if packed else [1]) * count
            log_probs[start:stop] = array("d", selected) if packed else selected
            old_log_probs[start:stop] = array("d", old_selected) if packed else old_selected
            offset += count
        fields: dict[str, Any] = {
            "tokens": tokens,
            "loss_mask": loss_mask,
            "log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "sampling_mask": deepcopy(segment.sampling_mask),
            "segment_metrics": dict(segment.segment_metrics),
            "turn_annotations": deepcopy(segment.turn_annotations),
        }
        # The compact schema and graph already validate supervision, token
        # IDs and sampling masks. Driver-only packed buffers must not be
        # converted back into Python lists by Pydantic's dense validators.
        decoded.append(
            RemoteSegment.model_construct(**fields)
            if packed
            else RemoteSegment.model_validate(fields)
        )
    return decoded


def encode_trace(trace: RemoteRolloutTrace) -> CompactRolloutTrace:
    return CompactRolloutTrace(
        schema_version=2,
        **trace.model_dump(exclude={"segments"}),
        segments=encode_segments(trace.segments),
    )


def decode_trace(trace: CompactRolloutTrace, *, packed: bool = False) -> RemoteRolloutTrace:
    return (RemoteRolloutTrace.model_construct if packed else RemoteRolloutTrace)(
        **trace.model_dump(exclude={"schema_version", "segments"}),
        segments=decode_segments(trace.segments, packed=packed),
    )


def encode_result(result: RemoteRolloutResult) -> CompactRolloutResult:
    return CompactRolloutResult(
        schema_version=2,
        **result.model_dump(exclude={"segments"}),
        segments=encode_segments(result.segments),
    )


def decode_result(result: CompactRolloutResult, *, packed: bool = False) -> RemoteRolloutResult:
    return (RemoteRolloutResult.model_construct if packed else RemoteRolloutResult)(
        **result.model_dump(exclude={"schema_version", "segments"}),
        segments=decode_segments(result.segments, packed=packed),
    )


def encode_agent_stage(stage: AgentStageResult) -> CompactAgentStageResult:
    return CompactAgentStageResult(
        schema_version=2,
        **stage.model_dump(exclude={"trace"}),
        trace=encode_trace(stage.trace) if stage.trace is not None else None,
    )


def decode_agent_stage(stage: CompactAgentStageResult) -> AgentStageResult:
    return AgentStageResult(
        **stage.model_dump(exclude={"schema_version", "trace"}),
        trace=decode_trace(stage.trace) if stage.trace is not None else None,
    )


def load_trace(payload: str | bytes | bytearray) -> CompactRolloutTrace:
    return CompactRolloutTrace.model_validate_json(payload)


def load_result(payload: str | bytes | bytearray) -> CompactRolloutResult:
    return CompactRolloutResult.model_validate_json(payload)


def load_agent_stage(payload: str | bytes | bytearray) -> CompactAgentStageResult:
    return CompactAgentStageResult.model_validate_json(payload)


# Gzip artifacts and coalesced checkpoints.


def dump_artifact(model: BaseModel) -> bytes:
    """Serialize compact JSON and use reproducible, fast gzip compression."""
    return gzip.compress(model.model_dump_json().encode("utf-8"), compresslevel=1, mtime=0)


def _decompress(payload: bytes | bytearray) -> bytes:
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("rollout artifacts must be gzip bytes")
    try:
        return gzip.decompress(payload)
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("rollout artifact is not valid gzip") from exc


def load_trace_gzip(payload: bytes | bytearray) -> CompactRolloutTrace:
    return load_trace(_decompress(payload))


def load_result_gzip(payload: bytes | bytearray) -> CompactRolloutResult:
    return load_result(_decompress(payload))


def load_agent_stage_gzip(payload: bytes | bytearray) -> CompactAgentStageResult:
    return load_agent_stage(_decompress(payload))


def write_artifact(path: Path, model: BaseModel) -> None:
    """Publish one complete gzip artifact atomically with private permissions."""
    directory = path.parent
    if directory.is_symlink():
        raise RuntimeError(f"platform directory must not be a symlink: {directory}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.resolve() != directory.absolute() or directory.stat().st_uid != os.geteuid():
        raise RuntimeError(f"platform directory must be owned by the runtime: {directory}")
    directory.chmod(0o700)
    if path.is_symlink():
        raise RuntimeError(f"rollout output must not be a symlink: {path}")
    payload = dump_artifact(model)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=directory, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class CoalescingArtifactWriter:
    """One writer thread with at most one latest pending immutable snapshot.

    Submit only detached/read-only models. The worker never accesses a recorder,
    metrics collector, or mutable event-loop state. Closing drains the newest
    snapshot and joins the thread before returning or propagating cancellation.
    """

    def __init__(
        self,
        path: Path,
        *,
        write: Callable[[Path, BaseModel], None] = write_artifact,
    ) -> None:
        self._path = path
        self._write = write
        self._condition = threading.Condition()
        self._pending: BaseModel | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def submit(self, snapshot: BaseModel) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("artifact writer is closed")
            self._pending = snapshot
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="northrl-artifact-writer", daemon=False
                )
                self._thread.start()
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None:
                    return
                snapshot, self._pending = self._pending, None
            try:
                self._write(self._path, snapshot)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._error = exc
            else:
                self._error = None

    async def aclose(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._thread is not None:
            await run_artifact_io(self._thread.join)
        if self._error is not None:
            raise self._error


# Shared thread work and download concurrency.

Priority = Literal["critical", "control", "background"]
_T = TypeVar("_T")


async def _drain(future: asyncio.Future[Any]) -> None:
    """Wait for actual completion even if the waiting task is cancelled again."""

    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
        except BaseException:  # The owner retrieves the original result/exception.
            break
    if not future.cancelled():
        future.exception()  # Retrieve errors even when cancellation wins the race.


class DownloadQueue:
    """Unlimited waiting downloads with bounded concurrency on one event loop.

    ``deadline`` uses absolute ``time.monotonic()`` seconds and includes
    queueing and execution. Optional local postprocessing keeps the slot after
    the remote deadline ends, bounding downloaded buffers until they are consumed.

    Control reads use reserved slots; other downloads share the remaining
    slots, with critical jobs taking priority over background jobs.
    Cancelling active work drains its operation before releasing its slot.
    """

    def __init__(
        self,
        max_active: int = 64,
        control_slots: int = 1,
    ) -> None:
        if not 1 <= control_slots < max_active:
            raise ValueError("control_slots must leave at least one non-control active slot")
        self._loop = asyncio.get_running_loop()
        self._normal_capacity = max_active - control_slots
        self._control_capacity = control_slots
        self._pending: dict[Priority, deque[asyncio.Future[None]]] = {
            "critical": deque(),
            "control": deque(),
            "background": deque(),
        }
        self._active: dict[Priority, int] = {priority: 0 for priority in self._pending}

    @property
    def active_count(self) -> int:
        return sum(self._active.values())

    @property
    def pending_count(self) -> int:
        return sum(len(pending) for pending in self._pending.values())

    def _can_start(self, priority: Priority) -> bool:
        if priority == "control":
            return self._active["control"] < self._control_capacity
        normal_active = self._active["critical"] + self._active["background"]
        return normal_active < self._normal_capacity

    def _start_ready(self) -> None:
        for priority in ("control", "critical", "background"):
            pending = self._pending[priority]
            while pending and self._can_start(priority):
                admitted = pending.popleft()
                if admitted.cancelled():
                    continue
                self._active[priority] += 1
                admitted.set_result(None)

    async def _run(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        priority: Priority,
        deadline: float | None,
        postprocess: Callable[[_T], Awaitable[Any]] | None,
    ) -> _T:
        admitted: asyncio.Future[None] = self._loop.create_future()
        self._pending[priority].append(admitted)

        async def execute() -> _T:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("download deadline expired before execution")
            return await operation()

        try:
            remaining = None if deadline is None else deadline - time.monotonic()
            async with asyncio.timeout(remaining):
                self._start_ready()
                await admitted
                # The caller owns the slot before creating a worker. Even cancelling
                # this task before its first instruction cannot orphan that slot.
                task = self._loop.create_task(execute(), name="northrl-download")
                try:
                    result = await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.cancelling():
                        task.cancel()
                    await _drain(task)
                    raise
            # Local processing owns cancellation/draining and keeps this permit.
            # Its completed download must survive expiry of the remote deadline.
            return await postprocess(result) if postprocess is not None else result
        finally:
            # A grant can race with caller cancellation before await resumes.
            # A completed, non-cancelled Future still means this caller owns it.
            if admitted.done() and not admitted.cancelled():
                self._active[priority] -= 1
            else:
                admitted.cancel()
                if admitted in self._pending[priority]:
                    self._pending[priority].remove(admitted)
            self._start_ready()

    async def run(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        priority: Priority = "critical",
        deadline: float | None = None,
        postprocess: Callable[[_T], Awaitable[Any]] | None = None,
    ) -> _T:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("DownloadQueue must be used on its owning event loop")
        if priority not in self._pending:
            raise ValueError(f"unknown download priority: {priority}")
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("download deadline expired before admission")
        return await self._run(
            operation, priority=priority, deadline=deadline, postprocess=postprocess
        )


def get_download_queue() -> DownloadQueue:
    """Return the queue attached to this loop, without process-global loop references."""

    loop = asyncio.get_running_loop()
    queue = getattr(loop, "_northrl_download_queue", None)
    if queue is None:
        queue = DownloadQueue(max_active=128, control_slots=8)
        setattr(loop, "_northrl_download_queue", queue)
    return cast(DownloadQueue, queue)


async def run_blocking(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run at most two functions per loop; cancellation drains real thread work.

    Submission happens only after acquiring the permit, so the default
    executor cannot accumulate an unbounded queue of payload-processing jobs.
    A cancelled caller retains its permit until its function really returns.
    The loop still owns and joins its default executor during shutdown.
    """

    loop = asyncio.get_running_loop()
    semaphore = getattr(loop, "_northrl_download_blocking_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.BoundedSemaphore(2)
        setattr(loop, "_northrl_download_blocking_semaphore", semaphore)
    async with semaphore:
        return await run_artifact_io(function, *args, **kwargs)


async def run_background_io(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Bound diagnostic work separately so slow trace writes cannot occupy result slots."""
    loop = asyncio.get_running_loop()
    semaphore = getattr(loop, "_northrl_background_io_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.BoundedSemaphore(4)
        setattr(loop, "_northrl_background_io_semaphore", semaphore)
    async with semaphore:
        return await run_artifact_io(function, *args, **kwargs)


async def run_artifact_io(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run off-loop work and drain the actual thread before propagating cancellation.

    Shield the executor Future directly: a wrapper Task could itself be cancelled
    during loop-wide shutdown while its thread continues publishing an old file.
    A stalled filesystem can extend cleanup; this is not a wall-clock deadline.
    """

    future = asyncio.get_running_loop().run_in_executor(
        None, copy_context().run, functools.partial(function, *args, **kwargs)
    )
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await _drain(future)
        raise
