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

"""Assemble model turns into token-faithful remote segments."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from .contracts import RemoteSegment
from .wire import CompactSegment, encode_segment

if TYPE_CHECKING:
    from .metrics import RolloutMetrics


class TrajectoryRecorder:
    """Record a rollout's trajectory as segments of exact model tokens."""

    def __init__(
        self,
        *,
        on_update: Callable[["TrajectoryRecorder"], None] | None = None,
    ) -> None:
        self._metrics: RolloutMetrics | None = None
        self._segments: list[RemoteSegment] = []
        self._compact_segments: list[CompactSegment] = []
        self._message_format: Literal["openai", "anthropic"] | None = None
        self._message_system: Any = None
        self._tools: list[dict[str, Any]] | None = None
        self._messages: list[dict[str, Any]] = []
        self._on_update = on_update
        self._turn_count = 0

    def set_metrics(self, metrics: "RolloutMetrics") -> None:
        """Associate recorded calls with this execution's telemetry collector."""
        self._metrics = metrics

    def record_model_turn(
        self,
        *,
        prompt_tokens: Sequence[int],
        response_tokens: Sequence[int],
        response_log_probs: Sequence[float],
        response_old_log_probs: Sequence[float],
        sampling_mask: Sequence[Sequence[int]] | dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        truncated: bool = False,
        tools_declared: bool = False,
    ) -> None:
        """Append one Weaver request and response.

        Args:
            prompt_tokens: Exact token IDs sent to Weaver for this request.
            response_tokens: Token IDs sampled by Weaver.
            response_log_probs: Weaver response log probabilities aligned with
                ``response_tokens``.
            response_old_log_probs: Behavior-policy log probabilities aligned
                with ``response_tokens``.
            sampling_mask: Optional candidate token IDs for each response token.
            diagnostics: Unmodified NorthGate diagnostics; absent evidence stays None.
            finish_reason: The provider's original response termination reason.
            truncated: Whether a response was cut off or its completion is uncertain.
            tools_declared: Whether the request's tool declarations are known.

        When ``prompt_tokens`` continues the latest segment, only its new
        suffix is appended as context. Otherwise, this turn starts a new
        segment without interrupting the rollout.
        """

        prompt = list(prompt_tokens)
        response = list(response_tokens)
        response_log_probs = list(response_log_probs)
        response_old_log_probs = list(response_old_log_probs)
        if not prompt:
            raise ValueError("prompt_tokens must not be empty")
        if not response:
            raise ValueError("response_tokens must not be empty")

        current = self._segments[-1] if self._segments else None
        current_tokens = current.tokens if current is not None else []
        context_suffix = self._prompt_suffix(prompt, current_tokens)
        if context_suffix is None:
            current = None
            context_suffix = prompt

        turn_sampling_mask = self._copy_sampling_mask(sampling_mask)
        current_sampling_mask = current.sampling_mask if current is not None else None
        if current is not None and ((current_sampling_mask is None) != (sampling_mask is None)):
            raise ValueError("sampling_mask must be provided for every model turn or none")

        combined_sampling_mask = None
        if turn_sampling_mask is not None:
            from northrl.utils.sampling_mask_refs import join_sampling_masks

            combined_sampling_mask = join_sampling_masks(current_sampling_mask, turn_sampling_mask)

        candidate = RemoteSegment(
            tokens=[*(current.tokens if current is not None else []), *context_suffix, *response],
            loss_mask=[
                *(current.loss_mask if current is not None else []),
                *([0] * len(context_suffix)),
                *([1] * len(response)),
            ],
            log_probs=[
                *(current.log_probs if current is not None else []),
                *([0.0] * len(context_suffix)),
                *response_log_probs,
            ],
            old_log_probs=[
                *(current.old_log_probs if current is not None else []),
                *([0.0] * len(context_suffix)),
                *response_old_log_probs,
            ],
            sampling_mask=combined_sampling_mask,
            turn_annotations=[
                *(current.turn_annotations if current is not None else []),
                {
                    "version": 1,
                    "turn_id": f"turn-{self._turn_count}",
                    "response_span": [len(prompt), len(prompt) + len(response)],
                    "diagnostics": deepcopy(diagnostics),
                    "finish_reason": finish_reason,
                    "truncated": truncated,
                    "tools_declared": tools_declared,
                    "source": "northgate",
                },
            ],
        )
        if current is None:
            self._segments.append(candidate)
        else:
            self._segments[-1] = candidate
            if len(self._compact_segments) == len(self._segments):
                self._compact_segments.pop()
        self._turn_count += 1

    def record_inference_result(
        self,
        result: dict[str, Any],
        *,
        message_format: Literal["openai", "anthropic"] | None = None,
        message_system: Any = None,
        tools: Sequence[dict[str, Any]] | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        truncated: bool = False,
    ) -> None:
        """Record token data and a provider-native message transcript."""

        transcript: list[dict[str, Any]] | None = None
        tool_definitions: list[dict[str, Any]] | None = None
        if messages is not None:
            if message_format not in {"openai", "anthropic"}:
                raise ValueError("message_format must be openai or anthropic when messages are set")
            transcript = [deepcopy(message) for message in messages]
            if tools is not None:
                tool_definitions = [deepcopy(tool) for tool in tools]

        training_data = result["northrl_train"]
        response_log_probs = training_data["response_logprobs"]
        choices = result.get("choices") or []
        finish_reason = (
            result.get("stop_reason")
            if message_format == "anthropic"
            else choices[0].get("finish_reason") if choices else None
        )
        if not isinstance(finish_reason, str):
            finish_reason = None
        diagnostics = result.get("tool_call_diagnostics")
        self.record_model_turn(
            prompt_tokens=training_data["prompt_tokens"],
            response_tokens=training_data["response_tokens"],
            response_log_probs=response_log_probs,
            response_old_log_probs=training_data.get(
                "response_old_logprobs",
                response_log_probs,
            ),
            sampling_mask=result.get("sampling_mask"),
            diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
            finish_reason=finish_reason,
            truncated=truncated or finish_reason in {"length", "max_tokens"},
            tools_declared=tools is not None,
        )
        if self._metrics is not None:
            self._metrics.record_emitted_ids(result, len(self._segments) - 1)
        if transcript is not None:
            self._message_format = message_format
            self._message_system = deepcopy(message_system)
            self._tools = tool_definitions
            self._messages = transcript
        if self._on_update is not None:
            self._on_update(self)

    @property
    def segment_count(self) -> int:
        """Return the segment count without copying token arrays for telemetry."""

        return len(self._segments)

    @property
    def message_format(self) -> Literal["openai", "anthropic"] | None:
        """Return the provider format used by the latest transcript."""

        return self._message_format

    @property
    def message_system(self) -> Any:
        """Return the provider-native top-level system value, if any."""

        return deepcopy(self._message_system)

    def to_messages(self) -> list[dict[str, Any]]:
        """Return the latest successful provider-native message transcript."""

        return deepcopy(self._messages)

    @property
    def tools(self) -> list[dict[str, Any]] | None:
        """Return native tool definitions from the latest successful transcript."""

        return deepcopy(self._tools)

    def to_remote_segments(self) -> list[RemoteSegment]:
        """Return copies of the recorded segments using the Driver/E2B wire contract."""

        if not self._segments:
            raise ValueError("cannot create segments before recording a model turn")
        segments = [segment.model_copy(deep=True) for segment in self._segments]
        if self._metrics is not None:
            for index, segment in enumerate(segments):
                segment.segment_metrics = self._metrics.segment_snapshot(index)
        return segments

    def to_compact_segments(self) -> list[CompactSegment]:
        """Borrow read-only wire segments, encoding only new or changed token data.

        Finalized segments retain their compact token/logprob arrays across trace
        checkpoints. Tool observations may arrive later, so refresh their small
        metrics dictionaries separately without copying or encoding those arrays.
        """

        if not self._segments:
            raise ValueError("cannot create segments before recording a model turn")
        while len(self._compact_segments) < len(self._segments):
            index = len(self._compact_segments)
            self._compact_segments.append(
                encode_segment(
                    self._segments[index],
                    self._segments[index - 1] if index else None,
                    base_segment=index - 1 if index else None,
                )
            )
        if self._metrics is not None:
            for index, segment in enumerate(self._compact_segments):
                metrics = self._metrics.segment_snapshot(index)
                if metrics != segment.segment_metrics:
                    self._compact_segments[index] = segment.model_copy(
                        update={"segment_metrics": metrics}
                    )
        return list(self._compact_segments)

    def to_remote_trajectories(self) -> list[RemoteSegment]:
        """Compatibility wrapper for :meth:`to_remote_segments`."""

        return self.to_remote_segments()

    @staticmethod
    def _prompt_suffix(prompt: list[int], recorded_tokens: list[int]) -> list[int] | None:
        if not recorded_tokens:
            return prompt
        if len(prompt) < len(recorded_tokens):
            return None
        if prompt[: len(recorded_tokens)] != recorded_tokens:
            return None
        return prompt[len(recorded_tokens) :]

    @staticmethod
    def _copy_sampling_mask(
        sampling_mask: Sequence[Sequence[int]] | dict[str, Any] | None,
    ) -> list[list[int]] | dict[str, Any] | None:
        if sampling_mask is None:
            return None
        if isinstance(sampling_mask, dict):
            import copy

            return copy.deepcopy(sampling_mask)
        return [list(candidates) for candidates in sampling_mask]
