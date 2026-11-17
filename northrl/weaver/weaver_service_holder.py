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
Weaver Service Holder - centralized async Weaver client manager.

This holder owns tokenizer and all Weaver clients (service, training, sampling).
All Weaver SDK interactions use nex-weaver's async clients.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import time
from array import array
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Coroutine, Literal, Mapping, TypeVar, cast
from uuid import uuid4

import httpx

try:
    from weaver import AsyncServiceClient, MetricsStoreConfig, WeaverAPIError, types
except ImportError:
    MetricsStoreConfig = None
    AsyncServiceClient = None
    WeaverAPIError = None
    types = None

try:
    # Router Replay is an internal cross-component protocol and is intentionally
    # not part of weaver.types.__all__.
    from weaver.types.router_replay import RouterReplayMetadata
except ImportError:
    RouterReplayMetadata = None

from northrl.utils.service_metrics import aggregate_service_metrics, reference_kl_from_outputs

from ..exceptions import ContextLengthExceededError, WeightVersionMismatchError
from ..utils.profiling import profile_block, profile_event
from ..utils.url_utils import ensure_url_scheme
from ..weight_sync.versioning import DEFAULT_WEIGHT_VERSION
from .custom_losses import parse_custom_loss_spec, resolve_custom_loss
from .datum_preparation import prepare_dense_loss_inputs
from .sampling_preparation import SamplingPreparationWorker, settle_task

logger = logging.getLogger(__name__)

_DatumPreparationResult = TypeVar("_DatumPreparationResult")


def _prepare_model_input(tokens: Any, *, tensor_transport: str = "default") -> Any:
    """Use the SDK's owned-buffer path only for binary requests; old SDKs fall back."""
    factory = types.ModelInput.from_ints
    if tensor_transport == "http-binary":
        factory = getattr(types.ModelInput, "_from_int_buffer", factory)
    return factory(tokens)


class _UnsetSentinel:
    """Sentinel for arguments that were not provided by the caller."""


_UNSET = _UnsetSentinel()


@dataclass(frozen=True, slots=True)
class ForwardLogprobsResult:
    """Result of one batched trainer ``forward_logprob`` operation.

    ``logprobs`` are target-token aligned. R2 RECORD additionally returns the
    trainer-produced opaque replay-set handle from that same forward operation.
    Keeping both values in one result prevents replay refs from leaking across
    calls through holder or trainer instance state.
    """

    logprobs: list[Sequence[float]]
    router_replay_record_ref: dict[str, Any] | None = None


# Default TTL (seconds) applied to sampler-only weight exports. Sampler
# weights are short-lived RL weight-sync artifacts, so we cap retention at
# 1 day to prevent the checkpoint store from accumulating per-step exports
# indefinitely. Pass ``sampler_ttl_seconds=None`` to keep them.
DEFAULT_SAMPLER_TTL_SECONDS = 86400

SAMPLE_MAX_ATTEMPTS = 3
SAMPLE_RETRY_BACKOFF_SECONDS = 0.5
SAMPLING_CONTROL_TIMEOUT_SECONDS = 30.0
SAMPLING_RESUME_MAX_ATTEMPTS = 3
SAMPLING_CONTROL_RETRY_BACKOFF_SECONDS = 1.0
GATEWAY_PUBLICATION_TIMEOUT_SECONDS = 300.0
GATEWAY_PUBLICATION_POLL_SECONDS = 5.0
SAMPLING_SESSION_CLEANUP_TIMEOUT_SECONDS = 5.0


class _MissingSampleSequencesError(ValueError):
    """Raised when a sampler response does not contain exactly one sequence."""


def _is_sample_rate_limit_error(exc: Exception) -> bool:
    """Return whether Weaver rejected a sample before creating the operation."""

    return WeaverAPIError is not None and isinstance(exc, WeaverAPIError) and exc.status_code == 429


def _sample_retry_delay(exc: Exception, attempt: int) -> float:
    """Respect Weaver's Retry-After hint while retaining the local backoff floor."""

    delay = SAMPLE_RETRY_BACKOFF_SECONDS * attempt
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        return delay
    try:
        return max(delay, float(retry_after))
    except (TypeError, ValueError):
        return delay


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _operation_result(operation_or_result: Any) -> Any:
    if hasattr(operation_or_result, "result"):
        return await _maybe_await(operation_or_result.result())
    return await _maybe_await(operation_or_result)


def _extract_metrics_from_result(
    result_obj: Any, *, accumulate_loss: bool = False
) -> dict[str, Any]:
    """Normalize metrics across Weaver response shapes."""
    if not isinstance(result_obj, dict):
        return {}

    top = result_obj
    inner = result_obj.get("result")
    if isinstance(inner, dict):
        top = inner

    metrics = top.get("metrics")
    if isinstance(metrics, dict):
        return metrics

    fb_task_results = top.get("forward_backward_task_results")
    if not isinstance(fb_task_results, dict) or not fb_task_results:
        return {}

    records = [
        task["metrics"]
        for task in fb_task_results.values()
        if isinstance(task, dict) and isinstance(task.get("metrics"), dict)
    ]
    return aggregate_service_metrics(records, accumulate=accumulate_loss)


def _forward_result_payload(result_obj: Any) -> dict[str, Any]:
    """Return the canonical Weaver forward result payload.

    AsyncTrainingClient returns ``{"result": {...}}``. Treating any other
    nesting as a protocol error keeps the Weaver/NorthRL boundary explicit and
    prevents a partially compatible response from being silently accepted.
    """
    if not isinstance(result_obj, dict):
        raise ValueError("Weaver forward response must be a dict.")
    payload = result_obj.get("result")
    if not isinstance(payload, dict):
        raise ValueError("Weaver forward response is missing the canonical result object.")
    return payload


def _extract_forward_logprobs_batch_from_result(result_obj: Any) -> list[Sequence[float]]:
    """Extract canonical batched ``forward_logprob`` outputs."""
    payload = _forward_result_payload(result_obj)
    outputs = payload.get("loss_fn_outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Weaver forward result is missing non-empty loss_fn_outputs.")

    # Import lazily: the holder also supports deployments without the training extra.
    import numpy as np
    import torch

    batch_logprobs: list[Sequence[float]] = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ValueError(f"loss_fn_outputs[{output_index}] must be a dict.")
        raw_logprobs = output.get("logprobs")
        if raw_logprobs is None:
            raw_logprobs = output.get("Logprobs")
        if isinstance(raw_logprobs, dict):
            raw_logprobs = raw_logprobs.get("data")
        if isinstance(raw_logprobs, torch.Tensor):
            if raw_logprobs.device.type != "cpu" or raw_logprobs.ndim != 1:
                raise ValueError(f"loss_fn_outputs[{output_index}].logprobs must be a CPU vector.")
            if raw_logprobs.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64):
                raise ValueError(
                    f"loss_fn_outputs[{output_index}].logprobs must be a supported numeric tensor."
                )
            # Match float()/array('d') precision, without per-token Python objects.
            values = np.asarray(raw_logprobs.detach().numpy(), dtype=np.float64, order="C")
            compact = array("d")
            compact.frombytes(memoryview(values).cast("B"))
            batch_logprobs.append(compact)
            continue
        if not isinstance(raw_logprobs, list):
            raise ValueError(f"loss_fn_outputs[{output_index}] is missing a logprobs list.")
        try:
            batch_logprobs.append([float(value) for value in raw_logprobs])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"loss_fn_outputs[{output_index}].logprobs must contain numbers."
            ) from exc
    return batch_logprobs


def _extract_router_replay_record_ref(result_obj: Any) -> dict[str, Any]:
    """Extract the canonical R2 recorded-index envelope without reshaping it."""
    payload = _forward_result_payload(result_obj)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Weaver forward result is missing canonical metrics.")
    envelope = metrics.get("router_replay_recorded_indices")
    if not isinstance(envelope, dict):
        raise ValueError("Weaver forward metrics are missing router_replay_recorded_indices.")
    index_set_uri = envelope.get("uri")
    if not isinstance(index_set_uri, str) or not index_set_uri:
        raise ValueError("Router replay recorded-index envelope is missing uri.")
    return {**envelope, "index_set_uri": index_set_uri}


class WeaverServiceHolder:  # pylint: disable=too-many-public-methods
    """Centralized manager for async Weaver services."""

    def __init__(
        self,
        base_model: str = "",
        lora_rank: int = 32,
        base_url: str | None = None,
        tokenizer_path: str | None = None,
        training_mode: str = "lora",
        need_sampling: bool = True,
        sampler_ttl_seconds: int | None = DEFAULT_SAMPLER_TTL_SECONDS,
        max_concurrent_requests: int | None = None,
        model_path: str | None = None,
        sampling_only: bool = False,
        allow_weight_version_mismatch: bool = False,
        max_sequence_length: int | None = None,
        performance_tier: str | None = None,
        chat_template: str | None = None,
        training_max_sequence_length: int | None = None,
        scheduling_policy: Mapping[str, Any] | None = None,
        metrics_store: Mapping[str, Any] | None = None,
        metric_observer: Callable[[str, Any], None] | None = None,
        gateway_publication_timeout_seconds: float | None = None,
    ):
        assert AsyncServiceClient is not None, "weaver package not installed"
        assert types is not None, "weaver package not installed"

        if model_path == "":
            raise ValueError("model_path must not be empty")
        if model_path is not None and not tokenizer_path:
            raise ValueError("tokenizer_path is required when loading a sampling model path")
        if sampling_only and model_path is None and not base_model:
            raise ValueError("base_model is required for sampling_only without a model_path")

        logger.info(
            "Initializing WeaverServiceHolder with base_model: %s, base_url: %s",
            base_model,
            base_url,
        )

        self._base_model = base_model
        self._lora_rank = lora_rank
        self._base_url = ensure_url_scheme(base_url, default_scheme="https") if base_url else None
        self._scheduling_policy = (
            deepcopy(dict(scheduling_policy)) if scheduling_policy is not None else None
        )
        self._metrics_store = dict(metrics_store) if metrics_store is not None else {}
        self._metric_observer = metric_observer
        self._tokenizer_path = tokenizer_path
        self._chat_template = chat_template
        self._training_mode = training_mode
        self._performance_tier = performance_tier
        self._need_sampling = need_sampling
        self._sampler_ttl_seconds = sampler_ttl_seconds
        self._model_path = model_path
        self._sampling_only = sampling_only or model_path is not None
        self._allow_weight_version_mismatch = allow_weight_version_mismatch
        if max_sequence_length is not None and (
            not isinstance(max_sequence_length, int)
            or isinstance(max_sequence_length, bool)
            or max_sequence_length < 1
        ):
            raise ValueError("max_sequence_length must be a positive integer")
        self._max_sequence_length = max_sequence_length
        if self._sampling_only and self._tokenizer_path is None:
            self._tokenizer_path = base_model
        self._training_max_sequence_length = training_max_sequence_length
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        self._max_concurrent_requests = max_concurrent_requests
        if gateway_publication_timeout_seconds is not None:
            self._validate_gateway_publication_timeout(gateway_publication_timeout_seconds)
        self._gateway_publication_timeout_seconds = gateway_publication_timeout_seconds

        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._provision_lock: asyncio.Lock | None = None
        self._datum_preparation_lock: asyncio.Lock | None = None
        self._sampling_preparation_worker: SamplingPreparationWorker | None = None
        self._sampling_semaphore: asyncio.Semaphore | None = None
        self._sampling_semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._service_client: Any | None = None
        self._training_client: Any | None = None
        self._sampling_client: Any | None = None
        self._tokenizer: Any | None = None

        self._active_sampling_snapshot: tuple[str, str] | None = None
        self._gateway_publication_tasks: dict[str, asyncio.Task[dict[str, str]]] = {}
        self._gateway_publication_cache: dict[str, dict[str, str]] = {}
        self._return_sampling_mask = False
        self._return_old_logprob = False
        self._return_moe_topk_indices = False

        if self._max_concurrent_requests is not None:
            logger.info(
                "Weaver sampler request concurrency limit: %s",
                self._max_concurrent_requests,
            )

    async def _ensure_provisioned(self, *, require_sampling: bool = False) -> None:
        if self._closed:
            raise RuntimeError("WeaverServiceHolder is closed")
        model_path = getattr(self, "_model_path", None)
        inference_only = getattr(self, "_sampling_only", False) or model_path is not None
        if inference_only:
            if self._sampling_client is not None and self._tokenizer is not None:
                return
        elif (
            self._training_client is not None
            and self._tokenizer is not None
            and (not require_sampling or self._sampling_client is not None)
        ):
            return

        if self._provision_lock is None:
            self._provision_lock = asyncio.Lock()

        async with self._provision_lock:
            if self._closed:
                raise RuntimeError("WeaverServiceHolder is closed")
            if self._service_client is None:
                with profile_block("weaver.service-connect"):
                    scheduling_kwargs = (
                        {"scheduling_policy": self._scheduling_policy}
                        if self._scheduling_policy is not None
                        else {}
                    )
                    self._service_client = AsyncServiceClient(
                        base_url=self._base_url,
                        api_key=os.getenv("WEAVER_API_KEY"),
                        tensor_transport="http-binary",
                        tensor_compression="zstd",
                        metrics_store=MetricsStoreConfig(**self._metrics_store),
                        wandb_link=None,  # NorthRL alone owns W&B init/log/finish.
                        metric_observer=self._metric_observer,
                        **scheduling_kwargs,
                    )
                    await self._service_client.connect()

            if inference_only:
                if self._tokenizer is None:
                    with profile_block("weaver.tokenizer-load"):
                        self._tokenizer = await self._get_sampling_preparation_worker().run(
                            self._load_tokenizer
                        )
                if self._sampling_client is None:
                    sampling_target = {
                        "base_model": self._base_model or None,
                        "tokenizer_path": self._tokenizer_path,
                    }
                    if model_path is not None:
                        sampling_target["model_path"] = model_path
                    logger.info(
                        "Creating Weaver sampling client from: %s", model_path or self._base_model
                    )
                    with profile_block("weaver.sampling-client-create"):
                        self._sampling_client = await self._service_client.create_sampling_client(
                            **sampling_target,
                        )
                logger.info("Initialized WeaverServiceHolder for inference")
                return

            if self._training_client is None:
                logger.info("Creating Weaver async training client")
                model_kwargs: dict[str, Any] = {}
                if self._performance_tier is not None:
                    model_kwargs["performance_tier"] = self._performance_tier
                if self._training_max_sequence_length is not None:
                    model_kwargs["training_max_sequence_length"] = (
                        self._training_max_sequence_length
                    )
                self._training_client = await self._service_client.create_model(
                    base_model=self._base_model,
                    training_mode=self._training_mode,
                    lora_config={"rank": self._lora_rank},
                    **model_kwargs,
                )
                logger.info(
                    "Training client model id: %s",
                    getattr(self._training_client, "model_id", None),
                )

            # Loading can fail or be cancelled after the model was created.
            # Reuse that client and retry the still-missing tokenizer next time.
            if self._tokenizer is None:
                self._tokenizer = await self._get_sampling_preparation_worker().run(
                    self._load_tokenizer
                )

            if (self._need_sampling or require_sampling) and self._sampling_client is None:
                logger.info("Saving initial weights for Weaver async sampler")
                sampling_path = await self.save_weights_for_sampler(name=DEFAULT_WEIGHT_VERSION)
                await self.update_sampling_client(sampling_path, DEFAULT_WEIGHT_VERSION)

            logger.info("Initialized WeaverServiceHolder async clients")

    def _get_sampling_preparation_worker(self) -> SamplingPreparationWorker:
        if getattr(self, "_closed", False):
            raise RuntimeError("WeaverServiceHolder is closed")
        worker = getattr(self, "_sampling_preparation_worker", None)
        if worker is None:
            worker = self._sampling_preparation_worker = SamplingPreparationWorker()
        return worker

    def _load_tokenizer(self) -> Any:
        """Load and configure the tokenizer in the holder's preparation worker."""
        if self._tokenizer_path is None:
            assert self._training_client is not None
            tokenizer = self._training_client.get_tokenizer()
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)
        if self._chat_template is not None:
            tokenizer.chat_template = self._chat_template
        return tokenizer

    async def _get_sampling_client_snapshot(self):
        await self._ensure_provisioned(require_sampling=True)
        return self._sampling_client, await self.get_current_sampling_version()

    def _get_sampling_semaphore(self) -> asyncio.Semaphore | None:
        if self._max_concurrent_requests is None:
            return None

        loop = asyncio.get_running_loop()
        if self._sampling_semaphore is None or self._sampling_semaphore_loop is not loop:
            self._sampling_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
            self._sampling_semaphore_loop = loop
        return self._sampling_semaphore

    async def _sample_with_limit(self, sampling_client: Any, **kwargs: Any) -> Any:
        # Keep one slot for the complete operation, including result polling.
        semaphore = self._get_sampling_semaphore()
        if semaphore is None:
            return await self._sample_operation_with_retries(sampling_client, **kwargs)

        async with semaphore:
            return await self._sample_operation_with_retries(sampling_client, **kwargs)

    @staticmethod
    def _require_single_sample(num_samples: int) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples != 1:
            raise ValueError("Weaver sampling only supports num_samples=1")

    @staticmethod
    def _extract_sample_sequence(sample_result: Any) -> dict[str, Any]:
        if not isinstance(sample_result, dict):
            raise _MissingSampleSequencesError("sample result must contain exactly one sequence")
        # wait=False returns the operation's raw result, whose sequences may be
        # nested under result. Preserve all sequence fields (including opaque
        # mask/routing references) instead of rebuilding a reduced SDK shape.
        if "sequences" not in sample_result and isinstance(sample_result.get("result"), dict):
            sample_result = sample_result["result"]
        sequences = sample_result.get("sequences")
        if not isinstance(sequences, list) or len(sequences) != 1:
            raise _MissingSampleSequencesError("sample result must contain exactly one sequence")
        sequence = sequences[0]
        if not isinstance(sequence, dict):
            raise _MissingSampleSequencesError("sample result first sequence must be an object")
        return sequence

    @staticmethod
    def _sample_idempotency_parameter(sampling_client: Any) -> str | None:
        # Current nex-weaver has no submission idempotency parameter. Do not
        # invent headers or assume **kwargs implies server-side deduplication.
        try:
            parameters = inspect.signature(sampling_client.sample).parameters
        except (TypeError, ValueError):
            return None
        if "idempotency_key" in parameters:
            return "idempotency_key"
        return None

    @staticmethod
    def _sample_query_retryable(exc: Exception) -> bool:
        if isinstance(exc, (OSError, httpx.TransportError, _MissingSampleSequencesError)):
            return True
        if WeaverAPIError is None or not isinstance(exc, WeaverAPIError):
            return False
        status_code = exc.status_code
        return bool(exc.retryable) or (
            isinstance(status_code, int) and (status_code == 429 or status_code >= 500)
        )

    @staticmethod
    async def _sample_retry_backoff(
        exc: Exception, attempt: int, *, stage: str, operation_id: str | None
    ) -> None:
        if attempt == SAMPLE_MAX_ATTEMPTS:
            logger.error(
                "Sample failed: stage=%s operation_id=%s attempts=%d error_type=%s",
                stage,
                operation_id,
                attempt,
                type(exc).__name__,
            )
            raise exc
        delay = _sample_retry_delay(exc, attempt)
        logger.warning(
            "Sample retry: stage=%s operation_id=%s attempt=%d/%d "
            "retry_delay_seconds=%.3f error_type=%s",
            stage,
            operation_id,
            attempt,
            SAMPLE_MAX_ATTEMPTS,
            delay,
            type(exc).__name__,
        )
        await asyncio.sleep(delay)

    async def _sample_operation_with_retries(self, sampling_client: Any, **kwargs: Any) -> Any:
        """Retry submission and queries without abandoning an accepted operation.

        The installed SDK also has transport retries of its own. This holder
        avoids additional ambiguous submissions; end-to-end deduplication still
        requires an SDK/server idempotency contract.
        """
        idempotency_parameter = self._sample_idempotency_parameter(sampling_client)
        if idempotency_parameter is not None:
            kwargs[idempotency_parameter] = str(uuid4())

        for attempt in range(1, SAMPLE_MAX_ATTEMPTS + 1):
            try:
                operation = await sampling_client.sample(wait=False, **kwargs)
                break
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A read/write failure may arrive after the server accepted the
                # POST. Without an SDK idempotency contract it must never cause
                # another sample. Only explicit rejection or a failure before
                # sending any request is safe to submit again.
                retryable = _is_sample_rate_limit_error(exc) or isinstance(
                    exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
                )
                if idempotency_parameter is not None:
                    retryable = retryable or self._sample_query_retryable(exc)
                if not retryable:
                    logger.error(
                        "Sample failed: stage=submit operation_id=None error_type=%s",
                        type(exc).__name__,
                    )
                    raise
                await self._sample_retry_backoff(exc, attempt, stage="submit", operation_id=None)
        else:
            raise AssertionError("unreachable")

        if not callable(getattr(operation, "result", None)):
            # Some adapters already return a completed response. Validate once;
            # without an operation handle there is no safe query to retry.
            self._extract_sample_sequence(operation)
            return operation

        operation_id = getattr(operation, "operation_id", None)
        logger.debug("Sample submitted: stage=submit operation_id=%s", operation_id)
        for attempt in range(1, SAMPLE_MAX_ATTEMPTS + 1):
            try:
                if attempt > 1 and callable(getattr(operation, "refresh", None)):
                    # Refresh the same operation after failures, including a
                    # malformed cached done response. Never submit a new one.
                    await operation.refresh()
                result = await _operation_result(operation)
                self._extract_sample_sequence(result)
                return result
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if not self._sample_query_retryable(exc):
                    logger.error(
                        "Sample failed: stage=query operation_id=%s error_type=%s",
                        operation_id,
                        type(exc).__name__,
                    )
                    raise
                await self._sample_retry_backoff(
                    exc, attempt, stage="query", operation_id=operation_id
                )
        raise AssertionError("unreachable")

    async def _sample_with_retries(self, sampling_client: Any, **kwargs: Any) -> dict[str, Any]:
        result = await self._sample_with_limit(sampling_client, **kwargs)
        sequence = self._extract_sample_sequence(result)
        return await self._get_sampling_preparation_worker().run(
            self._decode_sample_sequence, sequence
        )

    def _decode_sample_sequence(self, sequence: dict[str, Any]) -> dict[str, Any]:
        tokens = sequence.get("tokens")
        if (
            sequence.get("text") is None
            and isinstance(tokens, list)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
        ):
            # wait=True previously decoded raw operation results inside the SDK.
            # Preserve that behavior with wait=False without dropping references
            # or mutating the operation handle's cached sequence.
            assert self._tokenizer is not None
            sequence = dict(sequence)
            sequence["text"] = self._tokenizer.decode(tokens, skip_special_tokens=False)
        return sequence

    def _bounded_max_tokens(self, prompt_tokens: int, max_tokens: int) -> int:
        limit = getattr(self, "_max_sequence_length", None)
        if limit is None:
            return max_tokens
        remaining = limit - prompt_tokens
        if remaining <= 0:
            raise ContextLengthExceededError(prompt_tokens, max_tokens, limit)
        return min(max_tokens, remaining)

    async def _sample_with_context_limit(
        self,
        sampling_client: Any,
        *,
        prompt_tokens: int,
        requested_max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return await self._sample_with_retries(sampling_client, **kwargs)
        except Exception as exc:
            if not (
                WeaverAPIError is not None
                and isinstance(exc, WeaverAPIError)
                and getattr(exc, "status_code", None) == 400
                and getattr(exc, "code", None) == "sequence_too_long"
            ):
                raise
            # Prefer the server's actual window when it reports a different limit.
            details = getattr(exc, "details", None)
            limit = details.get("max_seq_len") if isinstance(details, Mapping) else None
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                match = re.search(r">\s*(\d+)\s*$", str(getattr(exc, "message", "")))
                limit = (
                    int(match.group(1)) if match else getattr(self, "_max_sequence_length", None)
                )
            if limit is None:
                # No configured window or server limit: retain the original error
                # instead of inventing context details for an unconfigured caller.
                raise
            raise ContextLengthExceededError(prompt_tokens, requested_max_tokens, limit) from exc

    async def close(self) -> None:
        """Close the Weaver service client and cleanup resources."""
        task = getattr(self, "_close_task", None)
        if task is not None and not task.done():
            await settle_task(task)
            return
        if self._closed:
            logger.info("WeaverServiceHolder close skipped: holder is already closed")
            profile_event(
                "weaver.service-holder-close",
                "skipped",
                reason="already_closed",
            )
            return

        self._closed = True
        # Share cleanup across close callers. Cancellation must not strand the
        # service client after preparation has drained and the holder is closed.
        task = self._close_task = asyncio.create_task(self._close_resources())
        await settle_task(task)

    async def _close_resources(self) -> None:
        worker = getattr(self, "_sampling_preparation_worker", None)
        if worker is not None:
            await worker.close()
        publication_tasks = list(getattr(self, "_gateway_publication_tasks", {}).values())
        for task in publication_tasks:
            task.cancel()
        if publication_tasks:
            await asyncio.gather(*publication_tasks, return_exceptions=True)
        service_client = self._service_client
        close_fields = {
            "client_type": type(service_client).__name__ if service_client is not None else None,
            "training_client": self._training_client is not None,
            "sampling_client": self._sampling_client is not None,
        }
        if service_client is None:
            logger.info(
                "WeaverServiceHolder close skipped: no async service client was provisioned"
            )
            profile_event(
                "weaver.service-holder-close",
                "skipped",
                reason="no_service_client",
                **close_fields,
            )
            return

        logger.info(
            "Closing Weaver async service client "
            "(client_type=%s, training_client=%s, sampling_client=%s)",
            close_fields["client_type"],
            close_fields["training_client"],
            close_fields["sampling_client"],
        )
        try:
            with profile_block("weaver.service-holder-close", **close_fields):
                await service_client.aclose()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Weaver async service client close failed; holder remains marked closed",
                exc_info=True,
            )
        else:
            logger.info("Weaver async service client close completed")

    def __del__(self):
        if getattr(self, "_closed", True) or getattr(self, "_service_client", None) is None:
            return
        logger.warning(
            "WeaverServiceHolder finalized without an explicit close; attempting best-effort cleanup"
        )
        try:
            from ..utils.async_utils import run_coro

            run_coro(self.close())
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug("Best-effort WeaverServiceHolder cleanup failed", exc_info=True)

    # --------------------------------------------------------------------- #
    # Public API used by NorthRL components
    # --------------------------------------------------------------------- #
    async def get_managed_dataset_info(self, *, name: str, version: str) -> Any:
        """Return authorized safe catalog metadata for one explicit version."""

        await self._ensure_provisioned()
        assert self._service_client is not None
        datasets_client = getattr(self._service_client, "datasets", None)
        if datasets_client is None:
            raise RuntimeError(
                "The installed Weaver SDK does not support managed dataset catalog APIs."
            )
        return await _maybe_await(datasets_client.get(name=name, version=version))

    async def resolve_managed_sample_ref_lengths(
        self, refs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Resolve model-bound input lengths without exposing managed tokens."""

        await self._ensure_provisioned()
        assert self._training_client is not None
        sample_ref_type = getattr(types, "SampleRef", None)
        resolve = getattr(self._training_client, "resolve_sample_ref_lengths", None)
        if sample_ref_type is None or not callable(resolve):
            raise RuntimeError(
                "The installed Weaver SDK does not support managed sample length resolution."
            )

        sdk_refs = [
            sample_ref_type(
                dataset=ref["dataset"],
                version=ref["version"],
                sample_idx=ref["sample_idx"],
            )
            for ref in refs
        ]
        resolve_lengths = cast(Callable[[list[Any]], Any], resolve)
        resolved = await _maybe_await(resolve_lengths(sdk_refs))  # pylint: disable=not-callable
        if not isinstance(resolved, list) or len(resolved) != len(refs):
            actual_size = len(resolved) if isinstance(resolved, list) else type(resolved).__name__
            raise ValueError(
                "Weaver managed length response size does not match the request: "
                f"got={actual_size!r} expected={len(refs)}"
            )

        normalized: list[dict[str, Any]] = []
        for index, (expected, item) in enumerate(zip(refs, resolved)):
            returned_ref = getattr(item, "sample_ref", None)
            if returned_ref is None and isinstance(item, Mapping):
                returned_ref = item.get("sample_ref")

            def _ref_field(value: Any, field_name: str) -> Any:
                if isinstance(value, Mapping):
                    return value.get(field_name)
                return getattr(value, field_name, None)

            actual = {
                "dataset": _ref_field(returned_ref, "dataset"),
                "version": _ref_field(returned_ref, "version"),
                "sample_idx": _ref_field(returned_ref, "sample_idx"),
            }
            if actual != expected:
                raise ValueError(
                    "Weaver managed length response order/reference mismatch at "
                    f"index={index}: got={actual!r} expected={expected!r}"
                )
            count = (
                item.get("input_token_count")
                if isinstance(item, Mapping)
                else getattr(item, "input_token_count", None)
            )
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(
                    "Weaver managed length response has invalid input_token_count "
                    f"at index={index}: {count!r}"
                )
            normalized.append(
                {
                    "sample_ref": actual,
                    "input_token_count": count,
                }
            )
        return normalized

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        await self._ensure_provisioned()
        assert self._tokenizer is not None
        return await self._get_sampling_preparation_worker().run(
            self._apply_chat_template,
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        assert self._tokenizer is not None
        # Deep-copy so we do not mutate the caller's data.
        messages_copy = deepcopy(messages)

        # Ensure tool_call function.arguments are dicts, not JSON strings.
        # Chat templates use Jinja2 .items() on arguments, which requires a mapping.
        for message in messages_copy:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue
            for tool_call in tool_calls:
                function = tool_call.get("function")
                if not function:
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    if arguments != "":
                        try:
                            function["arguments"] = json.loads(arguments, strict=False)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Failed to parse tool call arguments as JSON: %s, "
                                "keeping as string: %s",
                                e,
                                arguments[:100],
                            )
                    else:
                        function["arguments"] = {}
                elif arguments is None:
                    function["arguments"] = {}

        template_output = self._tokenizer.apply_chat_template(
            messages_copy,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            tools=tools,
        )
        # Newer Transformers versions return BatchEncoding by default. Keep
        # the service interface returning token IDs for all callers.
        if tokenize and isinstance(template_output, Mapping):
            return template_output["input_ids"]
        return template_output

    async def configure_sampling_flags(
        self,
        return_sampling_mask: bool = False,
        return_old_logprob: bool = False,
        return_moe_topk_indices: bool = False,
        topk_output_logprobs: int = 0,
    ) -> None:
        """Configure optional sampler outputs used by the training pipeline."""
        self._return_sampling_mask = return_sampling_mask
        self._return_old_logprob = return_old_logprob
        self._return_moe_topk_indices = return_moe_topk_indices
        self._topk_output_logprobs = topk_output_logprobs

    def _sampling_client_supports(self, sampling_client: Any, parameter_name: str) -> bool:
        return parameter_name in inspect.signature(sampling_client.sample).parameters

    def _build_sample_kwargs(
        self,
        *,
        sampling_client: Any,
        model_input: Any,
        num_samples: int,
        sampling_params: Any,
    ) -> dict[str, Any]:
        sample_kwargs: dict[str, Any] = {
            "prompt": model_input,
            "num_samples": num_samples,
            "sampling_params": sampling_params,
            "include_prompt_logprobs": False,
            "return_sampling_mask": getattr(self, "_return_sampling_mask", False),
            "return_old_logprob": getattr(self, "_return_old_logprob", False),
        }
        if (
            sample_kwargs["return_sampling_mask"]
            and getattr(self, "_training_mode", "lora") == "full_ft"
        ):
            if not self._sampling_client_supports(sampling_client, "sampling_mask_transport"):
                raise RuntimeError(
                    "Sampling masks require a Weaver SDK with sampling_mask_transport"
                )
            sample_kwargs["sampling_mask_transport"] = "ref"
        return_moe_topk_indices = getattr(self, "_return_moe_topk_indices", False)
        if return_moe_topk_indices:
            if not self._sampling_client_supports(sampling_client, "return_moe_topk_indices"):
                raise RuntimeError(
                    "R3 router replay requires a Weaver SDK whose "
                    "SamplingClient.sample accepts return_moe_topk_indices."
                )
            sample_kwargs["return_moe_topk_indices"] = True
        elif self._sampling_client_supports(sampling_client, "return_moe_topk_indices"):
            sample_kwargs["return_moe_topk_indices"] = False
        k = getattr(self, "_topk_output_logprobs", 0)
        # Training config enforces unfiltered sampling. Greedy evaluation and
        # warmup share this holder but do not consume SC distributions.
        if k and (sampling_params.temperature, sampling_params.top_p, sampling_params.top_k) == (
            1,
            1,
            -1,
        ):
            if not hasattr(sampling_params, "score_centering"):
                raise ValueError(
                    "score_centering requires a Weaver SDK with SamplingParams.score_centering"
                )
            sample_kwargs["sampling_params"] = replace(
                sampling_params, score_centering={"head_size": k, "transport": "ref"}
            )
        return sample_kwargs

    def _extract_sampling_outputs(
        self,
        sequence: dict[str, Any],
        response_logprobs: list[float],
    ) -> tuple[list[float], list[list[int]] | dict[str, Any] | None, dict[str, Any] | None]:
        """Extract requested optional sampler outputs without reshaping refs."""
        response_old_logprobs = response_logprobs
        raw_old_logprobs = sequence.get("old_logprobs")
        if raw_old_logprobs is not None:
            if not isinstance(raw_old_logprobs, list):
                raise TypeError(
                    f"Weaver sequence old_logprobs must be a list, got {raw_old_logprobs!r}"
                )
            if any(not isinstance(logprob, float) for logprob in raw_old_logprobs):
                raise TypeError(
                    f"Weaver sequence old_logprobs must contain only floats, got {raw_old_logprobs!r}"
                )
            response_old_logprobs = raw_old_logprobs
        elif getattr(self, "_return_old_logprob", False):
            raise TypeError(
                f"Weaver sequence old_logprobs must be a list when requested, got {raw_old_logprobs!r}"
            )

        response_sampling_masks: list[list[int]] | dict[str, Any] | None = None
        raw_sampling_masks = sequence.get("sampling_masks")
        if sequence.get("sampling_mask_ref") is not None:
            from weaver.sampling_masks import validate_mask_sequence

            validate_mask_sequence(sequence)
            response_sampling_masks = {"sampling_mask_ref": dict(sequence["sampling_mask_ref"])}
        elif raw_sampling_masks is not None:
            if not isinstance(raw_sampling_masks, list):
                raise TypeError(
                    f"Weaver sequence sampling_masks must be a list, got {raw_sampling_masks!r}"
                )
            if any(
                not isinstance(mask, list)
                or any(not isinstance(token, int) or isinstance(token, bool) for token in mask)
                for mask in raw_sampling_masks
            ):
                raise TypeError(
                    f"Weaver sequence sampling_masks must contain only token-ID lists, got {raw_sampling_masks!r}"
                )
            response_sampling_masks = raw_sampling_masks
        elif getattr(self, "_return_sampling_mask", False):
            raise TypeError(
                f"Weaver sequence sampling_masks must be a list when requested, got {raw_sampling_masks!r}"
            )

        routing_ref: dict[str, Any] | None = None
        raw_routing_ref = sequence.get("moe_topk_indices_ref")
        if raw_routing_ref is not None:
            if not isinstance(raw_routing_ref, dict):
                raise TypeError(
                    f"Weaver sequence moe_topk_indices_ref must be a dict, got {raw_routing_ref!r}"
                )
            routing_ref = raw_routing_ref
        elif getattr(self, "_return_moe_topk_indices", False):
            raise TypeError(
                f"Weaver sequence moe_topk_indices_ref must be present when requested, got {raw_routing_ref!r}"
            )

        return response_old_logprobs, response_sampling_masks, routing_ref

    def _weaver_sequence_to_sampling_result(
        self,
        sequence: dict[str, Any],
        prompt_tokens: list[int],
        expected_weight_version: str | None,
        expected_topk: int = 0,
    ) -> dict[str, Any]:
        """Build NorthRL's sampling result from one normalized Weaver sequence."""
        response_tokens = sequence.get("tokens")
        if not isinstance(response_tokens, list):
            raise TypeError("Weaver sequence tokens must be a list")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in response_tokens):
            raise TypeError("Weaver sequence tokens must contain only integers")

        response_logprobs = sequence.get("logprobs")
        if not isinstance(response_logprobs, list):
            raise TypeError("Weaver sequence logprobs must be a list")
        invalid_logprobs = [
            logprob for logprob in response_logprobs if not isinstance(logprob, float)
        ]
        if invalid_logprobs:
            raise TypeError(
                "Weaver sequence logprobs must contain only floats; "
                f"invalid_count={len(invalid_logprobs)}"
            )

        (
            response_old_logprobs,
            response_sampling_masks,
            response_moe_topk_indices_ref,
        ) = self._extract_sampling_outputs(sequence, response_logprobs)

        if len(response_tokens) != len(response_logprobs):
            raise ValueError(
                "Weaver returned different token/logprob lengths: "
                f"tokens={len(response_tokens)}, logprobs={len(response_logprobs)}"
            )
        if len(response_tokens) != len(response_old_logprobs):
            raise ValueError(
                "Weaver returned different token/old_logprob lengths: "
                f"tokens={len(response_tokens)}, "
                f"old_logprobs={len(response_old_logprobs)}"
            )
        if isinstance(response_sampling_masks, list) and len(response_tokens) != len(
            response_sampling_masks
        ):
            raise ValueError(
                "Weaver returned different token/sampling-mask lengths: "
                f"tokens={len(response_tokens)}, "
                f"sampling_masks={len(response_sampling_masks)}"
            )

        raw_weight_version = sequence.get("weight_version")
        if raw_weight_version is None:
            raise ValueError("Weaver sequence is missing weight_version")
        weight_version = str(raw_weight_version)
        if not weight_version:
            raise ValueError("Weaver sequence weight_version must be non-empty")
        if expected_weight_version is not None and weight_version != expected_weight_version:
            message = (
                "Weaver returned a different weight version than the sampling client: "
                f"expected={expected_weight_version!r}, observed={weight_version!r}"
            )
            if getattr(self, "_allow_weight_version_mismatch", False):
                logger.warning(
                    "%s; continuing with the returned version in fully-async mode", message
                )
            else:
                raise WeightVersionMismatchError(message)

        response = sequence.get("text")
        if not isinstance(response, str):
            raise TypeError("Weaver sequence text must be a string")
        finish_reason = sequence.get("stop_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise TypeError("Weaver sequence stop_reason must be a non-empty string")

        sampling_result: dict[str, Any] = {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "response_logprobs": response_logprobs,
            "response_old_logprobs": response_old_logprobs,
            "response_sampling_masks": response_sampling_masks,
            "finish_reason": finish_reason,
            "weight_version": weight_version,
        }
        if response_moe_topk_indices_ref is not None:
            sampling_result["response_moe_topk_indices_ref"] = response_moe_topk_indices_ref
        k = expected_topk
        if k:
            from weaver.score_centering import sampler_distribution_data, validate_sampler_sequence

            validate_sampler_sequence(sequence, k, transport="ref")
            sampling_result["sampler_distribution_data"] = sampler_distribution_data(sequence)
        return sampling_result

    async def warmup_sampling(self, *, refresh_gateway: bool = False) -> None:
        """Exercise the shared sampler and prepare its gateway before E2B fan-out."""

        logger.info("Starting Weaver rollout sampler warmup")
        with profile_block("weaver.rollout-warmup"):
            await self.sample_from_token_ids(
                input_ids=[0], max_tokens=10, temperature=1.0, top_p=1.0, top_k=1
            )
            publication_timeout = getattr(self, "_gateway_publication_timeout_seconds", None)
            if publication_timeout is not None:
                # Startup can reuse publication already checked during resume.
                # Recovery explicitly refreshes after the probe wakes the sampler.
                await self.get_current_gateway_snapshot(
                    timeout_seconds=publication_timeout, refresh=refresh_gateway
                )
        logger.info("Weaver rollout sampler warmup completed")

    async def sample_from_messages(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        num_samples: int = 1,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        import traceback

        self._require_single_sample(num_samples)
        await self._ensure_provisioned(require_sampling=True)
        assert self._tokenizer is not None

        try:
            template_output: object = await self.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=True
            )
            if isinstance(template_output, Mapping):
                template_output = template_output.get("input_ids")
            if not isinstance(template_output, list):
                raise TypeError(
                    "tokenized chat template must return a list of token IDs; "
                    f"actual_type={type(template_output).__name__}, "
                    f"actual_value={template_output!r}"
                )
            if not template_output:
                raise ValueError("tokenized chat template must return a non-empty list")

            invalid_tokens = [
                token
                for token in template_output
                if not isinstance(token, int) or isinstance(token, bool)
            ]
            if invalid_tokens:
                raise TypeError(
                    "tokenized chat template returned invalid token IDs; "
                    f"invalid_count={len(invalid_tokens)}"
                )
            prompt_tokens = cast(list[int], template_output)
        except Exception as e:
            logger.error("Error in apply_chat_template: %s", e)
            logger.error(traceback.format_exc())
            raise

        try:
            model_input = await self._get_sampling_preparation_worker().run(
                types.ModelInput.from_ints, tokens=prompt_tokens
            )
        except Exception as e:
            logger.error("Error in ModelInput.from_ints: %s", e)
            logger.error(traceback.format_exc())
            raise

        sampling_params = types.SamplingParams(
            max_tokens=self._bounded_max_tokens(len(prompt_tokens), max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        sampling_client, expected_weight_version = await self._get_sampling_client_snapshot()
        assert sampling_client is not None, "Sampling client not initialized (need_sampling=False?)"

        sample_kwargs = self._build_sample_kwargs(
            sampling_client=sampling_client,
            model_input=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
        sequence = await self._sample_with_context_limit(
            sampling_client,
            prompt_tokens=len(prompt_tokens),
            requested_max_tokens=max_tokens,
            **sample_kwargs,
        )

        try:
            return await self._get_sampling_preparation_worker().run(
                self._weaver_sequence_to_sampling_result,
                sequence,
                prompt_tokens,
                expected_weight_version,
                expected_topk=(
                    getattr(sample_kwargs["sampling_params"], "score_centering", None) or {}
                ).get("head_size", 0),
            )
        except Exception as e:
            logger.error("Error in sequence: %s", e)
            logger.error(traceback.format_exc())
            raise

    async def sample_from_prompt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        num_samples: int = 1,
        stop: list[str] | None = None,
    ) -> dict:
        self._require_single_sample(num_samples)
        await self._ensure_provisioned(require_sampling=True)
        assert self._tokenizer is not None

        prompt_tokens, model_input = await self._get_sampling_preparation_worker().run(
            self._encode_sampling_prompt, prompt
        )
        sampling_params = types.SamplingParams(
            max_tokens=self._bounded_max_tokens(len(prompt_tokens), max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        sampling_client, expected_weight_version = await self._get_sampling_client_snapshot()
        assert sampling_client is not None, "Sampling client not initialized (need_sampling=False?)"
        sample_kwargs = self._build_sample_kwargs(
            sampling_client=sampling_client,
            model_input=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
        sequence = await self._sample_with_context_limit(
            sampling_client,
            prompt_tokens=len(prompt_tokens),
            requested_max_tokens=max_tokens,
            **sample_kwargs,
        )
        return await self._get_sampling_preparation_worker().run(
            self._weaver_sequence_to_sampling_result,
            sequence,
            prompt_tokens,
            expected_weight_version,
            expected_topk=(
                getattr(sample_kwargs["sampling_params"], "score_centering", None) or {}
            ).get("head_size", 0),
        )

    def _encode_sampling_prompt(self, prompt: str) -> tuple[list[int], Any]:
        assert self._tokenizer is not None
        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
        return prompt_tokens, types.ModelInput.from_ints(tokens=prompt_tokens)

    async def sample_from_token_ids(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        num_samples: int = 1,
        stop: list[str] | None = None,
    ) -> dict:
        self._require_single_sample(num_samples)
        await self._ensure_provisioned(require_sampling=True)
        assert self._tokenizer is not None

        model_input = await self._get_sampling_preparation_worker().run(
            types.ModelInput.from_ints, tokens=input_ids
        )
        sampling_params = types.SamplingParams(
            max_tokens=self._bounded_max_tokens(len(input_ids), max_tokens),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        sampling_client, expected_weight_version = await self._get_sampling_client_snapshot()
        assert sampling_client is not None
        sample_kwargs = self._build_sample_kwargs(
            sampling_client=sampling_client,
            model_input=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
        sequence = await self._sample_with_context_limit(
            sampling_client,
            prompt_tokens=len(input_ids),
            requested_max_tokens=max_tokens,
            **sample_kwargs,
        )
        return await self._get_sampling_preparation_worker().run(
            self._weaver_sequence_to_sampling_result,
            sequence,
            list(input_ids),
            expected_weight_version,
            expected_topk=(
                getattr(sample_kwargs["sampling_params"], "score_centering", None) or {}
            ).get("head_size", 0),
        )

    async def compute_target_logprobs_batch(
        self,
        tokens_batch: list[list[int]],
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None = None,
        datum_metadata_batch: list[dict[str, Any]] | None = None,
        *,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> list[Sequence[float]]:
        """Return trainer-forward logprobs for a batch, aligned with tokens[1:].

        ``datum_metadata_batch`` carries per-sequence protocol metadata such as
        an R3 rollout-routing REPLAY envelope. It is kept aligned with the input
        batch and copied onto the corresponding Weaver Datum.
        ``loss_fn_config`` is forwarded unchanged (e.g. actor temperature).
        Binary results use owned float64 arrays; legacy JSON results remain lists.
        """
        result = await self._compute_forward_logprobs_result(
            tokens_batch,
            sampling_masks=sampling_masks,
            router_replay_record=None,
            datum_metadata_batch=datum_metadata_batch,
            loss_fn_config=loss_fn_config,
        )
        return result.logprobs

    async def compute_target_logprobs_batch_with_record(
        self,
        tokens_batch: list[list[int]],
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None = None,
        *,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardLogprobsResult:
        """Run R2 RECORD and return logprobs plus its replay-set handle.

        Both values come from the same Weaver forward operation. The explicit
        result object is intentionally per-call: neither the holder nor the
        trainer keeps a pending response that a later batch could consume.
        """
        if RouterReplayMetadata is None:
            raise RuntimeError("RouterReplayMetadata is unavailable in the installed Weaver SDK.")
        result = await self._compute_forward_logprobs_result(
            tokens_batch,
            sampling_masks=sampling_masks,
            router_replay_record=RouterReplayMetadata.r2_record(fail_fast=True),
            datum_metadata_batch=None,
            loss_fn_config=loss_fn_config,
        )
        assert result.router_replay_record_ref is not None
        logger.info(
            "[router_replay] R2 RECORD: recorded %d sequences; manifest_uri=%s",
            len(tokens_batch),
            result.router_replay_record_ref["manifest_uri"],
        )
        return result

    async def _prepare_datums(
        self,
        builder: Callable[..., _DatumPreparationResult],
        *args: Any,
        sequence_count: int,
        token_count: int,
        **kwargs: Any,
    ) -> _DatumPreparationResult:
        """Build request data without occupying the shared SDK event loop.

        Builders only read caller-owned inputs and create local SDK objects;
        they must not submit operations or modify holder/client state. Inputs
        must stay read-only while preparation is running. Serialize builds per
        holder, and drain the worker even on repeated cancellation so cancelled
        batches cannot leave unbounded background work or submit an operation.
        """
        lock = getattr(self, "_datum_preparation_lock", None)
        if lock is None:
            lock = self._datum_preparation_lock = asyncio.Lock()

        lock_requested_at = time.perf_counter()
        async with lock:
            started_at = time.perf_counter()
            lock_wait_seconds = started_at - lock_requested_at
            worker_timings: dict[str, float] = {}

            def build_in_worker() -> _DatumPreparationResult:
                worker_started_at = time.perf_counter()
                worker_cpu_started_at = time.thread_time()
                # Includes dispatch to asyncio.to_thread and its executor queue;
                # neither lock waiting nor SDK submission is included.
                worker_timings["queue"] = worker_started_at - started_at
                try:
                    return builder(*args, **kwargs)
                finally:
                    worker_timings["wall"] = time.perf_counter() - worker_started_at
                    worker_timings["thread_cpu"] = time.thread_time() - worker_cpu_started_at

            task = asyncio.create_task(asyncio.to_thread(build_in_worker))
            cancelled = False
            try:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        cancelled = True
                    except BaseException:
                        break
                try:
                    result = task.result()
                except BaseException as exc:
                    if cancelled:
                        raise asyncio.CancelledError() from exc
                    raise
                if cancelled:
                    raise asyncio.CancelledError()
                if getattr(self, "_closed", False):
                    raise RuntimeError("WeaverServiceHolder is closed")
                return result
            finally:
                logger.info(
                    "[datum_preparation] sequences=%d tokens=%d duration_seconds=%.3f cancelled=%s "
                    "lock_wait_seconds=%.6f worker_queue_seconds=%s "
                    "builder_wall_seconds=%s builder_thread_cpu_seconds=%s",
                    sequence_count,
                    token_count,
                    time.perf_counter() - started_at,
                    cancelled,
                    lock_wait_seconds,
                    *(
                        round(worker_timings[key], 6) if key in worker_timings else None
                        for key in ("queue", "wall", "thread_cpu")
                    ),
                )

    async def _prepare_training_datums(self, datums_data: list[dict]) -> list[Any]:
        """Offload all training Datum factories through the same bounded path."""
        return await self._prepare_datums(
            self._build_weaver_datums,
            datums_data,
            sequence_count=len(datums_data),
            token_count=sum(len(datum.get("input_tokens", [])) for datum in datums_data),
            tensor_transport=self._datum_transport(),
        )

    def _datum_transport(self) -> str:
        return getattr(getattr(self, "_service_client", None), "tensor_transport", "default")

    @staticmethod
    def _build_forward_datums(
        tokens_batch: list[list[int]],
        *,
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None,
        router_replay_record: Any | None,
        datum_metadata_batch: list[dict[str, Any]] | None,
        tensor_transport: str = "default",
    ) -> tuple[list[Any], list[int]]:
        """Create forward Datums and their input indices using only local data."""
        if sampling_masks is not None and len(sampling_masks) != len(tokens_batch):
            raise ValueError(
                "sampling_masks length must match tokens_batch length. "
                f"Got sampling_masks={len(sampling_masks)}, tokens_batch={len(tokens_batch)}."
            )
        if datum_metadata_batch is not None and len(datum_metadata_batch) != len(tokens_batch):
            raise ValueError(
                "datum_metadata_batch length must match tokens_batch length. "
                f"Got datum_metadata_batch={len(datum_metadata_batch)}, "
                f"tokens_batch={len(tokens_batch)}."
            )
        if router_replay_record is not None and datum_metadata_batch is not None:
            raise ValueError("R2 RECORD metadata and per-datum metadata are mutually exclusive.")

        datums = []
        active_indices: list[int] = []
        active_masks: list[list[list[int]] | dict[str, Any] | None] = []
        for idx, tokens in enumerate(tokens_batch):
            if len(tokens) <= 1:
                continue

            sampling_mask = sampling_masks[idx] if sampling_masks is not None else None
            datum_metadata = datum_metadata_batch[idx] if datum_metadata_batch is not None else {}
            if not isinstance(datum_metadata, dict):
                raise TypeError(
                    f"datum_metadata_batch[{idx}] must be a dict, got {datum_metadata!r}"
                )
            loss_fn_inputs: dict[str, Any] = {"target_tokens": tokens[1:]}
            datum_metadata = dict(datum_metadata)
            if isinstance(sampling_mask, dict):
                datum_metadata.update(sampling_mask)
            elif sampling_mask is not None:
                loss_fn_inputs["sampling_mask"] = sampling_mask

            datums.append(
                types.Datum.from_raw(
                    model_input=_prepare_model_input(
                        tokens[:-1], tensor_transport=tensor_transport
                    ),
                    loss_fn_inputs=prepare_dense_loss_inputs(loss_fn_inputs),
                    metadata=dict(datum_metadata),
                )
            )
            active_indices.append(idx)
            active_masks.append(sampling_mask)

        if not datums:
            if router_replay_record is not None:
                raise ValueError("R2 RECORD requires at least one non-degenerate sequence.")
            return [], []

        if any(mask is not None for mask in active_masks) and any(
            mask is None for mask in active_masks
        ):
            raise ValueError("sampling_masks must be provided for every item in a batch or none.")

        if router_replay_record is not None:
            replay_payload = router_replay_record.to_payload()
            for datum in datums:
                datum.metadata["router_replay"] = dict(replay_payload)

        return datums, active_indices

    async def _compute_forward_logprobs_result(
        self,
        tokens_batch: list[list[int]],
        *,
        sampling_masks: list[list[list[int]] | dict[str, Any] | None] | None,
        router_replay_record: Any | None,
        datum_metadata_batch: list[dict[str, Any]] | None,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> ForwardLogprobsResult:
        """Execute one canonical Weaver forward_logprob request."""
        await self._ensure_provisioned()
        assert self._training_client is not None

        datums, active_indices = await self._prepare_datums(
            self._build_forward_datums,
            tokens_batch,
            sampling_masks=sampling_masks,
            router_replay_record=router_replay_record,
            datum_metadata_batch=datum_metadata_batch,
            sequence_count=len(tokens_batch),
            token_count=sum(map(len, tokens_batch)),
            tensor_transport=self._datum_transport(),
        )
        if not datums:
            return ForwardLogprobsResult(logprobs=[[] for _ in tokens_batch])
        results: list[Sequence[float] | None] = [None for _ in tokens_batch]

        forward_kwargs: dict[str, Any] = (
            {"loss_fn_config": dict(loss_fn_config)} if loss_fn_config is not None else {}
        )
        # Older SDKs retain the legacy result path without rejecting a new keyword.
        if "tensor_output_format" in inspect.signature(self._training_client.forward).parameters:
            forward_kwargs["tensor_output_format"] = "torch"
        handle = await self._training_client.forward(
            datums, "forward_logprob", wait=False, **forward_kwargs
        )
        result = await _operation_result(handle)
        batch_logprobs = await asyncio.to_thread(
            _extract_forward_logprobs_batch_from_result, result
        )
        if len(batch_logprobs) != len(datums):
            raise ValueError(
                "training_client.forward forward_logprob returned an unexpected batch size. "
                f"Got outputs={len(batch_logprobs)}, expected={len(datums)}."
            )
        for idx, logprobs in zip(active_indices, batch_logprobs):
            expected_len = len(tokens_batch[idx]) - 1
            if len(logprobs) != expected_len:
                raise ValueError(
                    "training_client.forward forward_logprob returned an unexpected length. "
                    f"Got logprobs={len(logprobs)}, expected={expected_len}."
                )
            results[idx] = logprobs

        record_ref = (
            _extract_router_replay_record_ref(result) if router_replay_record is not None else None
        )
        return ForwardLogprobsResult(
            logprobs=[item or [] for item in results],
            router_replay_record_ref=record_ref,
        )

    def _build_weaver_datums(
        self, datums_data: list[dict], *, tensor_transport: str = "default"
    ) -> list[Any]:
        datums = []
        for datum_data in datums_data:
            if datum_data.get("kind") == "sample_ref":
                datums.append(
                    types.Datum.from_sample_ref(
                        dataset=datum_data["dataset"],
                        version=datum_data["version"],
                        sample_idx=datum_data["sample_idx"],
                        loss_fn_inputs=datum_data.get("loss_fn_inputs") or {},
                        metadata=datum_data.get("metadata") or {},
                        datum_id=datum_data["datum_id"],
                    )
                )
                continue
            raw_kwargs = {
                "model_input": _prepare_model_input(
                    datum_data["input_tokens"], tensor_transport=tensor_transport
                ),
                "loss_fn_inputs": prepare_dense_loss_inputs(datum_data["loss_fn_inputs"]),
                "metadata": datum_data.get("metadata") or {},
            }
            if "datum_id" in datum_data:
                raw_kwargs["datum_id"] = datum_data["datum_id"]
            datums.append(types.Datum.from_raw(**raw_kwargs))
        return datums

    @staticmethod
    def _extract_loss_value(result: Any, metrics: dict[str, Any]) -> float | None:
        loss_value = metrics.get("loss")
        if loss_value is None and isinstance(result, dict) and "loss" in result:
            raw_loss = result["loss"]
            if hasattr(raw_loss, "detach"):
                raw_loss = raw_loss.detach().cpu()
            loss_value = float(raw_loss)
        return float(loss_value) if loss_value is not None else None

    async def forward_backward(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
    ) -> dict:
        handle = await self.submit_forward_backward(
            datums_data=datums_data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        return await self.collect_forward_backward_result(
            handle, datums_data=datums_data, loss_fn_config=loss_fn_config
        )

    async def submit_forward_backward(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
    ) -> Any:
        """Submit one backward operation without waiting for its result.

        Callers that accumulate several microbatches can enqueue their stable
        seq_id-ordered operations back-to-back, then collect every result before
        submitting the optimizer step. This preserves the no-partial-update
        failure boundary while avoiding a scheduler round trip between
        microbatches.
        """
        await self._ensure_provisioned()
        assert self._training_client is not None

        datums = await self._prepare_training_datums(datums_data)
        return await self._training_client.forward_backward(
            datums,
            loss_fn,
            loss_fn_config=loss_fn_config,
            wait=False,
        )

    @staticmethod
    async def collect_forward_backward_result(
        handle: Any,
        *,
        datums_data: list[dict],
        loss_fn_config: dict | None = None,
    ) -> dict:
        """Collect one previously submitted backward operation.

        ``datums_data`` and ``loss_fn_config`` must be the values the operation
        was submitted with; they select loss accumulation and reference-KL
        diagnostics exactly as the synchronous ``forward_backward`` path does.
        """

        result = await _operation_result(handle)
        metrics = dict(
            _extract_metrics_from_result(
                result,
                accumulate_loss=bool(
                    loss_fn_config and loss_fn_config.get("loss_agg_mode") == "sum"
                ),
            )
        )
        metrics.update(reference_kl_from_outputs(result, datums_data, loss_fn_config))
        return {
            "loss": WeaverServiceHolder._extract_loss_value(result, metrics),
            "metrics": metrics,
        }

    async def forward_backward_custom(
        self,
        datums_data: list[dict],
        custom_loss_spec: dict[str, Any],
        *,
        mini_batch_sizes: list[int] | None = None,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> dict:
        """Evaluate a custom loss over one optimizer batch, optionally in chunks.

        A callback can couple arbitrary samples, so it always sees the complete
        optimizer batch. All forward-only requests finish before any surrogate
        backward request; a forward-only call may clear accumulated gradients
        on the training backend.
        Probability settings must be identical for the forward and surrogate backward.
        """
        if mini_batch_sizes is not None and (
            not mini_batch_sizes
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in mini_batch_sizes
            )
            or sum(mini_batch_sizes) != len(datums_data)
        ):
            raise ValueError(
                "mini_batch_sizes must contain positive integers summing to len(datums_data)"
            )

        await self._ensure_provisioned()
        assert self._training_client is not None

        normalized_spec = parse_custom_loss_spec(custom_loss_spec)
        if normalized_spec is None:
            raise ValueError("custom_loss_spec is required for forward_backward_custom")

        datums = await self._prepare_training_datums(datums_data)
        callback = resolve_custom_loss(normalized_spec)
        probability_kwargs = (
            {"loss_fn_config": dict(loss_fn_config)} if loss_fn_config is not None else {}
        )
        if mini_batch_sizes is None or len(mini_batch_sizes) == 1:
            result = await self._training_client.forward_backward_custom(
                datums, callback, **probability_kwargs
            )
        else:
            # Use the SDK's result validation and surrogate construction.
            from weaver._payloads import build_surrogate_data, parse_logprob_tensors

            offset = 0
            logprob_tensors = []
            for size in mini_batch_sizes:
                chunk = datums[offset : offset + size]
                offset += size
                handle = await self._training_client.forward(
                    chunk, "forward_logprob", wait=False, **probability_kwargs
                )
                forward_result = await _operation_result(handle)
                logprob_tensors.extend(
                    await asyncio.to_thread(parse_logprob_tensors, forward_result, chunk)
                )

            try:
                loss, callback_metrics = callback(datums, logprob_tensors)
            except Exception as exc:
                raise RuntimeError(f"User loss_fn failed: {exc}") from exc
            if loss.dim() != 0:
                raise ValueError(f"loss_fn must return a scalar loss, got shape {loss.shape}")
            loss.backward()
            surrogate_data = build_surrogate_data(datums, logprob_tensors)
            # Older SDK helpers omit metadata; retain routing and sampling-mask refs.
            surrogate_data = [
                replace(surrogate, metadata=dict(datum.metadata))
                for datum, surrogate in zip(datums, surrogate_data, strict=True)
            ]

            offset = 0
            for size in mini_batch_sizes:
                handle = await self._training_client.forward_backward(
                    surrogate_data[offset : offset + size],
                    "surrogate",
                    wait=False,
                    **probability_kwargs,
                )
                await _operation_result(handle)
                offset += size
            result = {"loss": loss.detach(), "metrics": callback_metrics}

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        loss_value = self._extract_loss_value(result, metrics)
        if "loss" not in metrics:
            metrics = {**metrics, "loss": loss_value}
        return {"loss": loss_value, "metrics": metrics}

    async def optim_step(
        self,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
    ) -> dict:
        await self._ensure_provisioned()
        assert self._training_client is not None

        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            grad_clip_norm=grad_clip_norm,
        )

        handle = await self._training_client.optim_step(adam_params, wait=False)
        result = await _operation_result(handle)
        optim_metrics = {}
        if isinstance(result, dict) and "metrics" in result:
            optim_metrics = result["metrics"]

        return {"status": "success", "metrics": optim_metrics}

    async def forward_backward_and_optim_step(
        self,
        datums_data: list[dict],
        loss_fn: str = "importance_sampling",
        loss_fn_config: dict | None = None,
        learning_rate: float = 2e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 1e-2,
        eps: float = 1e-8,
        grad_clip_norm: float = 1.0,
    ) -> tuple[Any, Any]:
        """Submit forward_backward + optim_step without blocking."""
        await self._ensure_provisioned()
        assert self._training_client is not None

        datums = await self._prepare_training_datums(datums_data)
        fb_handle = await self._training_client.forward_backward(
            datums,
            loss_fn,
            loss_fn_config=loss_fn_config,
            wait=False,
        )
        adam_params = types.AdamParams(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            grad_clip_norm=grad_clip_norm,
        )
        try:
            optim_handle = await self._training_client.optim_step(adam_params, wait=False)
        except Exception:
            # The forward/backward operation was already submitted. Consume it
            # before propagating the optimizer-submission error so the remote
            # queue is not left with an unobserved operation.
            try:
                await _operation_result(fb_handle)
            except Exception:
                pass
            raise
        return fb_handle, optim_handle

    @staticmethod
    async def collect_training_result(handles: tuple[Any, Any]) -> dict:
        """Wait for a (fb_handle, optim_handle) pair and return combined metrics."""
        fb_handle, optim_handle = handles
        results: tuple[Any, Any] = await asyncio.gather(
            _operation_result(fb_handle),
            _operation_result(optim_handle),
            return_exceptions=True,
        )
        fb_result, optim_result = results
        if isinstance(fb_result, BaseException):
            raise fb_result
        if isinstance(optim_result, BaseException):
            raise optim_result

        fb_metrics = _extract_metrics_from_result(fb_result)
        combined_metrics = fb_metrics.copy()
        if isinstance(optim_result, dict) and "metrics" in optim_result:
            combined_metrics.update(optim_result["metrics"])
        return combined_metrics

    async def save_weights_for_sampler(
        self, name: str, ttl_seconds: int | None | _UnsetSentinel = _UNSET
    ) -> str:
        await self._ensure_provisioned()
        assert self._training_client is not None

        ttl = self._sampler_ttl_seconds if isinstance(ttl_seconds, _UnsetSentinel) else ttl_seconds
        started = time.monotonic()
        logger.info("Sampler weights save started: version=%s", name)
        save_path = await self._training_client.save_weights_for_sampler(
            name=name,
            ttl_seconds=ttl,
        )
        sampling_path = str(save_path)
        logger.info(
            "Sampler weights save completed: version=%s path=%s elapsed_seconds=%.3f",
            name,
            sampling_path,
            time.monotonic() - started,
        )
        return sampling_path

    async def save_state(self, name: str) -> str:
        """Save current model weights as a checkpoint and return the path."""
        await self._ensure_provisioned()
        assert self._training_client is not None

        checkpoint = await self._training_client.save_state(name=name, checkpoint_type="weight")
        if hasattr(checkpoint, "path"):
            return str(checkpoint.path)
        if isinstance(checkpoint, dict) and "path" in checkpoint:
            return str(checkpoint["path"])
        return str(checkpoint)

    async def save_weight_and_optimizer(self, name: str) -> str:
        """Save model weights and optimizer state for training resume."""
        await self._ensure_provisioned()
        assert self._training_client is not None

        checkpoint = await self._training_client.save_state(
            name=name,
            checkpoint_type="weight_and_optimizer",
        )
        if hasattr(checkpoint, "path"):
            return str(checkpoint.path)
        if isinstance(checkpoint, dict) and "path" in checkpoint:
            return str(checkpoint["path"])
        return str(checkpoint)

    async def load_state(self, path: str) -> None:
        """Load model weights from a checkpoint path."""
        await self._ensure_provisioned()
        assert self._training_client is not None
        await self._training_client.load_state(path)

    async def load_state_with_optimizer(self, path: str) -> None:
        """Load model weights and optimizer state from a checkpoint path."""
        await self._ensure_provisioned()
        assert self._training_client is not None
        await self._training_client.load_state_with_optimizer(path)

    async def get_current_sampling_version(self) -> str | None:
        if self._active_sampling_snapshot is not None:
            return self._active_sampling_snapshot[1]
        # A directly loaded sampler reports its own version in each response;
        # there is no locally exported version to assert against.
        if getattr(self, "_sampling_only", False) or self._model_path is not None:
            return None
        return DEFAULT_WEIGHT_VERSION

    async def get_current_sampling_snapshot(self) -> tuple[str, str]:
        """Return the model path and version used by the active sampler."""

        await self._ensure_provisioned(require_sampling=True)
        if self._active_sampling_snapshot is None:
            raise RuntimeError("The active Weaver sampler has no model path and version")
        return self._active_sampling_snapshot

    async def get_current_gateway_snapshot(
        self,
        timeout_seconds: float = GATEWAY_PUBLICATION_TIMEOUT_SECONDS,
        *,
        refresh: bool = False,
    ) -> dict[str, str]:
        """Resolve the current shared sampler's NorthGate entry on the SDK loop.

        Capture the client and version together before polling so a concurrent
        weight update cannot pair one sampler with another version. Rollout
        fan-out reuses the published gateway for the sampler's lifetime. Explicit
        refresh invalidates that result after a lifecycle event such as warmup.
        """
        self._validate_gateway_publication_timeout(timeout_seconds)
        await self._ensure_provisioned(require_sampling=True)
        sampling_client = self._sampling_client
        snapshot = self._active_sampling_snapshot
        if snapshot is None:
            raise RuntimeError("The active Weaver sampler has no model path and version")
        sampling_session_id = getattr(sampling_client, "sampling_session_id", None)
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            raise RuntimeError("The active Weaver sampler has no sampling session ID")
        if refresh:
            self._gateway_publication_cache.pop(sampling_session_id, None)
        gateway = await self._get_gateway_publication(sampling_session_id, timeout_seconds)
        return {
            **gateway,
            "sampling_session_id": sampling_session_id,
            "weight_version": snapshot[1],
        }

    async def refresh_gateway_snapshot(
        self, timeout_seconds: float = GATEWAY_PUBLICATION_TIMEOUT_SECONDS
    ) -> dict[str, str]:
        """Centrally recheck a gateway after waking or recovering its sampler."""
        return await self.get_current_gateway_snapshot(timeout_seconds, refresh=True)

    @staticmethod
    def _validate_gateway_publication_timeout(timeout_seconds: float) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("gateway publication timeout must be positive")
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("gateway publication timeout must be positive and finite")

    async def _get_gateway_publication(
        self, sampling_session_id: str, timeout_seconds: float
    ) -> dict[str, str]:
        cached = self._gateway_publication_cache.get(sampling_session_id)
        if cached is not None:
            gateway = cached
        else:
            task = self._gateway_publication_tasks.get(sampling_session_id)
            if task is None:
                task = asyncio.create_task(
                    self._wait_for_gateway_publication(sampling_session_id, timeout_seconds)
                )
                self._gateway_publication_tasks[sampling_session_id] = task

                def settled(completed: asyncio.Task[dict[str, str]]) -> None:
                    if self._gateway_publication_tasks.get(sampling_session_id) is completed:
                        del self._gateway_publication_tasks[sampling_session_id]
                    if not completed.cancelled():
                        completed.exception()

                task.add_done_callback(settled)
            try:
                # A cancelled rollout must not cancel other publication waiters.
                gateway = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"NorthGate publication timed out for sampler {sampling_session_id} "
                    f"after {timeout_seconds:g}s; "
                    "check gateway publication, owner binding and codec profile"
                ) from exc
            finally:
                if task.done() and self._gateway_publication_tasks.get(sampling_session_id) is task:
                    del self._gateway_publication_tasks[sampling_session_id]
        return gateway

    async def _wait_for_gateway_publication(
        self, sampling_session_id: str, timeout_seconds: float
    ) -> dict[str, str]:
        assert self._service_client is not None
        started = time.monotonic()
        deadline = started + timeout_seconds
        last_status = None
        logger.info(
            "Waiting for NorthGate publication: sampler=%s timeout_seconds=%s",
            sampling_session_id,
            timeout_seconds,
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"NorthGate publication timed out for sampler {sampling_session_id}; "
                    "check gateway publication, owner binding and codec profile"
                )
            state = await asyncio.wait_for(
                self._service_client.http.get(f"/api/v1/sampling-sessions/{sampling_session_id}"),
                timeout=remaining,
            )
            if not isinstance(state, dict):
                raise ValueError("Weaver sampler query must return an object")
            if state.get("Status") in {"deleted", "expired"}:
                raise RuntimeError(f"Weaver sampler {sampling_session_id} is unavailable")
            gateway = state.get("gateway") or {}
            if not isinstance(gateway, dict):
                raise ValueError("Weaver sampler gateway must be an object")
            status = (state.get("Status"), gateway.get("state"))
            if status != last_status:
                logger.info(
                    "NorthGate publication state: sampler=%s sampler_status=%s gateway_state=%s "
                    "elapsed_seconds=%.3f",
                    sampling_session_id,
                    *status,
                    time.monotonic() - started,
                )
                last_status = status
            if gateway.get("state") == "published":
                base_url, model = gateway.get("base_url"), gateway.get("model")
                if not isinstance(base_url, str) or not base_url.strip():
                    raise ValueError("Published NorthGate sampler has no base_url")
                if not isinstance(model, str) or not model.strip():
                    raise ValueError("Published NorthGate sampler has no model")
                result = {"base_url": base_url.strip().rstrip("/"), "model": model.strip()}
                self._gateway_publication_cache[sampling_session_id] = result
                logger.info(
                    "NorthGate publication ready: sampler=%s model=%s elapsed_seconds=%.3f",
                    sampling_session_id,
                    result["model"],
                    time.monotonic() - started,
                )
                return result
            if gateway.get("state") in {"retiring", "retired"}:
                raise RuntimeError(
                    f"NorthGate sampler {sampling_session_id} is retired; "
                    "call sample() or compute_logprobs() on the host Weaver sampling client, "
                    "then wait for gateway publication before retrying"
                )
            await asyncio.sleep(
                min(GATEWAY_PUBLICATION_POLL_SECONDS, max(0, deadline - time.monotonic()))
            )

    async def update_sampling_client(
        self,
        sampling_path: str,
        weight_version: str,
        *,
        pause_mode: Literal["retract", "in_place"] | None = None,
    ) -> None:
        """Publish weights, optionally pausing the engine until the swap completes.

        Weaver owns request scheduling during pause. Sampling retains its normal
        SDK path and can overlap these control operations.
        """
        if pause_mode not in (None, "retract", "in_place"):
            raise ValueError(f"Unsupported sampling pause mode: {pause_mode!r}")
        if pause_mode is not None and self._training_mode != "full_ft":
            raise ValueError("Sampling pause requires a full_ft Weaver model")
        await self._ensure_provisioned()
        if pause_mode is None or self._sampling_client is None:
            await self._replace_sampling_client(sampling_path)
            self._active_sampling_snapshot = (sampling_path, weight_version)
            return

        pause_generation = self._sampling_client.pause_generation
        stage_replacement = getattr(self, "_gateway_publication_timeout_seconds", None) is not None
        candidate = None
        failure: BaseException | None = None
        try:
            pause_started = time.monotonic()
            pause_session = getattr(self._sampling_client, "sampling_session_id", None)
            logger.info(
                "Sampling pause started: mode=%s target_version=%s session=%s",
                pause_mode,
                weight_version,
                pause_session,
            )
            await self._run_sampling_control(lambda: pause_generation(mode=pause_mode))
            logger.info(
                "Sampling pause completed: mode=%s target_version=%s session=%s elapsed_seconds=%.3f",
                pause_mode,
                weight_version,
                pause_session,
                time.monotonic() - pause_started,
            )
            if stage_replacement:
                candidate = await self._replace_sampling_client(sampling_path, activate=False)
            else:
                await self._replace_sampling_client(sampling_path)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # A failed pause RPC may have reached the engine. Attempt recovery
            # even without its acknowledgement. Replacement assigns only on
            # success, so the current client is usable after either outcome.
            try:
                resume_started = time.monotonic()
                resume_client = candidate if candidate is not None else self._sampling_client
                resume_session = getattr(resume_client, "sampling_session_id", None)
                logger.info(
                    "Sampling resume started: target_version=%s session=%s recovery=%s",
                    weight_version,
                    resume_session,
                    failure is not None,
                )
                await self._run_sampling_control(
                    resume_client.continue_generation,
                    max_attempts=SAMPLING_RESUME_MAX_ATTEMPTS,
                )
                logger.info(
                    "Sampling resume completed: target_version=%s session=%s recovery=%s elapsed_seconds=%.3f",
                    weight_version,
                    resume_session,
                    failure is not None,
                    time.monotonic() - resume_started,
                )
            except asyncio.CancelledError as exc:
                if candidate is not None:
                    await self._recover_unactivated_sampling_client(candidate)
                raise exc from failure
            except BaseException as resume_exc:
                if candidate is not None:
                    try:
                        await self._recover_unactivated_sampling_client(candidate)
                    except asyncio.CancelledError as exc:
                        raise exc from resume_exc
                if failure is None:
                    raise
                logger.error(
                    "Failed to continue sampling after synchronization failure",
                    exc_info=True,
                )
        if candidate is not None:
            try:
                self._activate_sampling_client(candidate)
            except BaseException:
                await self._discard_sampling_client(candidate)
                raise
        self._active_sampling_snapshot = (sampling_path, weight_version)

    async def _replace_sampling_client(self, sampling_path: str, *, activate: bool = True) -> Any:
        assert self._service_client is not None
        assert self._training_client is not None
        started = time.monotonic()
        old_session = getattr(self._sampling_client, "sampling_session_id", None)
        logger.info(
            "Sampling client replacement started: path=%s old_session=%s",
            sampling_path,
            old_session,
        )
        candidate = await self._service_client.create_sampling_client(
            model_path=sampling_path,
            base_model=self._base_model,
            model_id=getattr(self._training_client, "model_id", None),
        )
        candidate_session = getattr(candidate, "sampling_session_id", None)
        publication_timeout = getattr(self, "_gateway_publication_timeout_seconds", None)
        try:
            if publication_timeout is not None and self._sampling_client is not None:
                if not isinstance(candidate_session, str) or not candidate_session:
                    raise RuntimeError("The candidate Weaver sampler has no sampling session ID")
                # Keep the current sampler/version usable until its replacement is
                # published. Initial provisioning is checked after the warmup probe.
                await self._get_gateway_publication(candidate_session, publication_timeout)
            if activate:
                self._activate_sampling_client(candidate)
        except BaseException:
            await self._discard_sampling_client(candidate)
            raise
        logger.info(
            "Sampling client replacement %s: path=%s old_session=%s new_session=%s elapsed_seconds=%.3f",
            "completed" if activate else "prepared",
            sampling_path,
            old_session,
            candidate_session,
            time.monotonic() - started,
        )
        return candidate

    def _activate_sampling_client(self, sampling_client: Any) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError("WeaverServiceHolder is closed")
        self._sampling_client = sampling_client
        sampling_session_id = getattr(sampling_client, "sampling_session_id", None)
        self._gateway_publication_cache = {
            session: gateway
            for session, gateway in getattr(self, "_gateway_publication_cache", {}).items()
            if session == sampling_session_id
        }

    async def _recover_unactivated_sampling_client(self, candidate: Any) -> None:
        """Unpause the shared engine before discarding a failed E2B replacement."""
        assert self._sampling_client is not None
        try:
            await self._run_sampling_control(
                self._sampling_client.continue_generation,
                max_attempts=SAMPLING_RESUME_MAX_ATTEMPTS,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            logger.warning(
                "Failed to restore sampling after candidate resume failure", exc_info=True
            )
        finally:
            await self._discard_sampling_client(candidate)

    async def _discard_sampling_client(self, candidate: Any) -> None:
        """Settle cleanup of an abandoned session without deleting the active one."""
        session = getattr(candidate, "sampling_session_id", None)
        active_session = getattr(self._sampling_client, "sampling_session_id", None)
        if not isinstance(session, str) or not session or session == active_session:
            return

        async def cleanup() -> None:
            task = self._gateway_publication_tasks.pop(session, None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._gateway_publication_cache.pop(session, None)
            try:
                assert self._service_client is not None
                await asyncio.wait_for(
                    self._service_client.http.delete(f"/api/v1/sampling-sessions/{session}"),
                    timeout=SAMPLING_SESSION_CLEANUP_TIMEOUT_SECONDS,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to discard candidate sampler %s", session, exc_info=True)

        await settle_task(asyncio.create_task(cleanup()))

    @staticmethod
    async def _run_sampling_control(
        operation: Callable[[], Coroutine[Any, Any, Any]], *, max_attempts: int = 1
    ) -> None:
        """Settle bounded control RPCs and timeout retries before propagating cancellation."""

        async def run_with_retries() -> None:
            for attempt in range(1, max_attempts + 1):
                try:
                    await asyncio.wait_for(operation(), timeout=SAMPLING_CONTROL_TIMEOUT_SECONDS)
                    return
                except TimeoutError:
                    if attempt == max_attempts:
                        raise
                    delay = SAMPLING_CONTROL_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
                    logger.warning(
                        "Sampling control timed out: attempt=%s/%s timeout_seconds=%.3f "
                        "retry_delay_seconds=%.3f",
                        attempt,
                        max_attempts,
                        SAMPLING_CONTROL_TIMEOUT_SECONDS,
                        delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(delay)

        task = asyncio.create_task(run_with_retries())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        try:
            task.result()
        except BaseException as exc:
            if cancelled:
                raise asyncio.CancelledError() from exc
            raise
        if cancelled:
            raise asyncio.CancelledError()

    async def create_training_client(self, **kwargs: Any) -> Any:
        """Create an additional async Weaver training client from the shared service."""
        await self._ensure_provisioned()
        assert self._service_client is not None
        return await self._service_client.create_model(**kwargs)  # pylint: disable=missing-kwoa

    async def create_sampling_client(self, **kwargs: Any) -> Any:
        """Create an additional async Weaver sampling client from the shared service."""
        await self._ensure_provisioned()
        assert self._service_client is not None
        return await self._service_client.create_sampling_client(
            **kwargs
        )  # pylint: disable=missing-kwoa

    async def get_service_client(self):
        await self._ensure_provisioned()
        return self._service_client

    async def get_tokenizer(self):
        await self._ensure_provisioned()
        return self._tokenizer
