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
Remote API Inference Service Client - Base class for remote API backends (Tinker/Weaver)
"""

import inspect
import time
import uuid
from abc import abstractmethod
from typing import Any

from omegaconf import DictConfig

from ..utils.async_utils import await_coro
from ..utils.config_utils import validate_mis_sampling
from ..utils.response_parser import create_response_parser
from .base_inference_service_client import InferenceServiceClient


class RemoteApiInferenceServiceClient(InferenceServiceClient):
    """
    Base inference service client for remote API backends.

    This client is a thin wrapper that passes serializable data (strings, dicts)
    to service holders (Tinker/Weaver), which own the tokenizer and renderer.

    Derived classes should:
    - Set _service_holder in their __init__ or setter method
    - Optionally override methods for custom behavior
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the remote API inference service client.

        Args:
            config: Root NorthRLConfig (the rollout worker hands it through; see
                Phase 2.A). Client reads ``training.weight_sync.freeze_inflight``,
                ``rollout.parsers``, ``rollout.sampling.*``,
                and ``models.actor.name`` for the OpenAI-style response.model
                field.
        """
        super().__init__()
        self._config = config
        self._service_holder = None  # Set by derived class

        # ModelTag is hardcoded to "actor" (single trainable model in new schema).
        self._identifier = "actor"

        # freeze_inflight: when true, block in-flight inference during weight-sync
        # hot-swap (old name: inference_service.freeze_for_weight_sync).
        self._freeze_for_weight_sync = config.training.weight_sync.freeze_inflight

        parsers = config.rollout.parsers
        self._response_parser = create_response_parser(
            parsers,
            config_file_path=getattr(config, "_config_file_path", None),
        )

    def _sampling_defaults(self) -> dict[str, Any]:
        sampling = self._active_sampling()
        return {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
        }

    def _effective_max_tokens(self) -> int:
        """Use the active sampling.max_tokens or
        (data.max_sequence_length - data.max_prompt_length).
        Schema enforces the result is always ≥ 1.
        """
        explicit = self._active_sampling().max_tokens
        if explicit is not None:
            return int(explicit)
        return int(self._config.data.max_sequence_length - self._config.data.max_prompt_length)

    def _active_sampling(self):
        """Return rollout or validation sampling according to worker mode."""
        if self._is_validation:
            return self._config.validate.sampling
        return self._config.rollout.sampling

    def _parse_assistant_response(
        self,
        response: str,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Parse a raw backend response into the final assistant message."""
        diagnostics: dict[str, int] = {}
        message = self._response_parser.parse(response, tools, diagnostics=diagnostics)
        # generate() moves diagnostics out of the provider-native message.
        return {**message, "northrl_parse_diagnostics": diagnostics}

    @abstractmethod
    def set_service_holder(self, service_holder) -> None:
        """
        Set the service holder (must be implemented by derived classes).

        Args:
            service_holder: Service holder instance (TinkerServiceHolder, WeaverServiceHolder, etc.)
        """

    def _make_id(self, prefix: str) -> str:
        """Generate a unique identifier with prefix (mimics OpenAI ids)."""
        return f"{prefix}-{uuid.uuid4().hex}"

    async def _call_service_holder(self, method_name: str, **kwargs) -> Any:
        assert self._service_holder is not None, "ServiceHolder not set"
        method = getattr(self._service_holder, method_name)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            return await await_coro(result)
        return result

    @staticmethod
    def _attach_weight_version(
        payload: dict[str, Any], observed_version: str | None
    ) -> dict[str, Any]:
        if observed_version is not None:
            payload["weight_version"] = observed_version
        return payload

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages using service holder.

        Args:
            messages: List of message dictionaries (chat format)
            tools: List of tool dictionaries (None or empty list)
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        # Normalize tools to empty list if None
        if tools is None:
            tools = []

        return await self._call_service_holder(
            "apply_chat_template",
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            tools=tools,
        )

    async def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Async prompt completion using the service holder's sampling API."""
        assert self._service_holder is not None, "ServiceHolder not set"

        if self._freeze_for_weight_sync:
            await self._wait_for_weight_sync()

        max_tokens = self._effective_max_tokens()
        sampling_defaults = self._sampling_defaults()

        result = await self._call_service_holder(
            "sample_from_prompt",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=sampling_defaults["temperature"],
            top_p=sampling_defaults["top_p"],
            top_k=sampling_defaults["top_k"],
            num_samples=1,
        )
        response_weight_version = result.get("weight_version")

        prompt_tokens: list[int] = result["prompt_tokens"]
        response_tokens: list[int] = result["response_tokens"]
        response_logprobs: list[float] = result["response_logprobs"]
        response_old_logprobs: list[float] = result.get("response_old_logprobs", response_logprobs)
        response_sampling_masks = result.get("response_sampling_masks")
        response_moe_topk_indices_ref = result.get("response_moe_topk_indices_ref")

        openai_like = {
            "id": self._make_id("cmpl"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": self._config.models.actor.name,
            "choices": [
                {
                    "index": 0,
                    "text": result["response"],
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(prompt_tokens) + len(response_tokens),
            },
            "northrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
                "response_old_logprobs": response_old_logprobs,
            },
            "sampling_mask": response_sampling_masks,
            "moe_topk_indices_ref": response_moe_topk_indices_ref,
        }

        if "sampler_distribution_data" in result:
            openai_like["sampler_distribution_data"] = result["sampler_distribution_data"]
        self._attach_weight_version(openai_like, response_weight_version)
        return {**openai_like, **kwargs}

    async def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Call LLM inference for chat completion using service's sample API.

        Args:
            messages: List of message dictionaries (chat format, serializable)
            **kwargs: Additional parameters (passed through to result)

        Returns:
            dict[str, Any]: LLM response in NorthRL format
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        # Check weight sync status and block if necessary
        if self._freeze_for_weight_sync:
            await self._wait_for_weight_sync()

        max_tokens = self._effective_max_tokens()
        sampling_defaults = self._sampling_defaults()

        # Extract tools from kwargs if present
        tools = kwargs.pop("tools", [])

        result = await self._call_service_holder(
            "sample_from_messages",
            messages=messages,
            max_tokens=max_tokens,
            temperature=sampling_defaults["temperature"],
            top_p=sampling_defaults["top_p"],
            top_k=sampling_defaults["top_k"],
            num_samples=1,
            tools=tools,
        )
        response_weight_version = result.get("weight_version")

        prompt_tokens: list[int] = result["prompt_tokens"]
        response_tokens: list[int] = result["response_tokens"]
        response_logprobs: list[float] = result["response_logprobs"]
        response_old_logprobs: list[float] = result.get("response_old_logprobs", response_logprobs)
        response_sampling_masks = result.get("response_sampling_masks")
        response_moe_topk_indices_ref = result.get("response_moe_topk_indices_ref")
        message = self._parse_assistant_response(result["response"], tools)
        parse_diagnostics = message.pop("northrl_parse_diagnostics", {})

        openai_like = {
            "id": self._make_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._config.models.actor.name,
            "northrl_parse_diagnostics": parse_diagnostics,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(prompt_tokens) + len(response_tokens),
            },
            "northrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
                "response_old_logprobs": response_old_logprobs,
            },
            "sampling_mask": response_sampling_masks,
            "moe_topk_indices_ref": response_moe_topk_indices_ref,
        }

        if "sampler_distribution_data" in result:
            openai_like["sampler_distribution_data"] = result["sampler_distribution_data"]
        self._attach_weight_version(openai_like, response_weight_version)
        return {**openai_like, **kwargs}

    async def generate_with_token(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Call LLM inference using token IDs via the service holder.

        Returns the same OpenAI-style format as generate(), including
        reasoning/tool parsing.

        Args:
            input_ids: Input token IDs
            sampling_params: SGLang-style sampling parameters dict
            **kwargs: Top-level SGLang parameters (largely ignored for Weaver/Tinker)

        Returns:
            dict[str, Any]: OpenAI-style response with northrl_train attached
        """
        assert self._service_holder is not None, "ServiceHolder not set"

        if self._freeze_for_weight_sync:
            await self._wait_for_weight_sync()

        if sampling_params is None:
            sampling_params = {}

        if self._is_validation:
            # Agent-supplied training defaults must not override evaluation settings.
            sampling_params = {
                key: value
                for key, value in sampling_params.items()
                if key not in {"max_new_tokens", "temperature", "top_p", "top_k", "min_p"}
            }

        max_tokens = sampling_params.get("max_new_tokens", self._effective_max_tokens())
        sampling_defaults = self._sampling_defaults()
        temperature = sampling_params.get("temperature", sampling_defaults["temperature"])
        top_p = sampling_params.get("top_p", sampling_defaults["top_p"])
        top_k = sampling_params.get("top_k", sampling_defaults["top_k"])
        stop = sampling_params.get("stop", None)

        algorithm = getattr(self._config, "algorithm", None)
        mis = getattr(algorithm, "mis", None)
        if (
            not self._is_validation
            and mis is not None
            and (getattr(mis, "enabled", False) or getattr(mis, "get_mismatch_metrics", False))
        ):
            # Token agents may override the root sampling config per request.
            validate_mis_sampling(
                top_p=top_p,
                top_k=top_k,
                min_p=sampling_params.get("min_p", self._active_sampling().get("min_p", 0.0)),
                temperature=temperature,
                repetition_penalty=sampling_params.get(
                    "repetition_penalty",
                    self._active_sampling().get("repetition_penalty", 1.0),
                ),
            )

        result = await self._call_service_holder(
            "sample_from_token_ids",
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_samples=1,
            stop=stop,
        )
        response_weight_version = result.get("weight_version")

        prompt_tokens: list[int] = result["prompt_tokens"]
        response_tokens: list[int] = result["response_tokens"]
        response_logprobs: list[float] = result["response_logprobs"]
        tools = kwargs.get("tools", [])
        message = self._parse_assistant_response(result["response"], tools)
        parse_diagnostics = message.pop("northrl_parse_diagnostics", {})

        openai_like = {
            "id": self._make_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._config.models.actor.name,
            "northrl_parse_diagnostics": parse_diagnostics,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(response_tokens),
                "total_tokens": len(prompt_tokens) + len(response_tokens),
            },
            "northrl_train": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_logprobs": response_logprobs,
            },
        }

        if "sampler_distribution_data" in result:
            openai_like["sampler_distribution_data"] = result["sampler_distribution_data"]
        self._attach_weight_version(openai_like, response_weight_version)
        return openai_like
