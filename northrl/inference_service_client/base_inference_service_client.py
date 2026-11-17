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
Base Inference Service Client classes and utilities for NorthRL framework
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

from openai.types import Completion
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs (HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}")


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer.

    Args:
        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.
        **kwargs: The keyword arguments for the tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107."
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    logger.debug(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    return tokenizer


class InferenceServiceClient(ABC):
    """
    Abstract base class for inference service clients.
    Defines the interface that all inference service clients must implement.
    """

    def __init__(self):
        """Initialize common attributes for weight synchronization."""
        self._weight_sync_controller = None
        # identifier serves as model_tag for weight sync coordination
        self._identifier: str = "default"
        self._freeze_for_weight_sync: bool = True
        self._is_validation = False

    def set_validation_mode(self, enabled: bool) -> None:
        """Select validation sampling parameters for subsequent generations."""
        self._is_validation = bool(enabled)

    @abstractmethod
    async def completion(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Call LLM inference for prompt completion.

        Args:
            prompt: Input prompt string
            **kwargs: Additional parameters for completion

        Returns:
            dict[str, Any]: LLM response dict containing all completion fields plus:
                - northrl_train: dict with prompt_tokens, response_tokens,
                  response_logprobs, response_old_logprobs
        """

    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        """
        Call LLM inference for chat completion.

        Args:
            messages: List of message dictionaries (chat format)
            **kwargs: Additional parameters for generation

        Returns:
            dict[str, Any]: LLM response dict containing all completion fields plus:
                - northrl_train: dict with prompt_tokens, response_tokens,
                  response_logprobs, response_old_logprobs
        """

    async def generate_with_token(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Call LLM inference using token IDs.

        Implementations may call backend-native token generation APIs, but should
        return a NorthRL-compatible response with `northrl_train` attached. Backends
        may additionally normalize the payload to the same OpenAI-style structure
        used by `generate()`.

        Args:
            input_ids: Input token IDs
            sampling_params: SGLang-style sampling parameters dict (max_new_tokens,
                temperature, top_p, top_k, min_p, stop, stop_token_ids, etc.)
            **kwargs: Top-level SGLang parameters (model, return_logprob,
                logprob_start_len, stream, rid, lora_path, etc.)

        Returns:
            dict[str, Any]: Response dict containing at minimum:
                - northrl_train: Dict with prompt_tokens, response_tokens, response_logprobs
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support generate_with_token(). "
            "Use a backend that supports token-level generation (e.g., sglang, weaver, tinker)."
        )

    @abstractmethod
    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages to get a prompt string or tokens.

        Args:
            messages: List of message dictionaries (chat format)
            tools: List of tool dictionaries
            add_generation_prompt: Whether to add the generation prompt
            tokenize: If True, return token IDs; if False, return string

        Returns:
            str | list[int]: Formatted prompt string or token IDs
        """

    def set_weight_sync_controller(self, weight_sync_controller) -> None:
        """
        Set reference to weight manager for checking sync status.

        Args:
            weight_sync_controller: WeightSyncController instance
        """
        self._weight_sync_controller = weight_sync_controller

    async def _wait_for_weight_sync(self) -> None:
        """Wait asynchronously until rollout weights are ready for inference."""
        import asyncio

        assert self._weight_sync_controller is not None, "Weight sync controller not set"

        max_wait_seconds = 3600
        wait_interval = 1
        total_waited = 0.0

        while total_waited < max_wait_seconds:
            status = self._weight_sync_controller.check_rollout_service_status(self._identifier)

            if status == "continue":
                return
            if status == "block":
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval
                continue
            raise ValueError(f"Unknown status from weight manager: {status}")

        raise RuntimeError(
            f"Inference service timed out after waiting {max_wait_seconds}s for weight sync"
        )

    @property
    def completions(self):
        """
        Provide access to OpenAI completions API, aligned with OpenAI client interface.
        Allows async calls like: await client.completions.create(...)
        The create method calls our completion method internally.

        Returns:
            CompletionsWrapper: Wrapper object with create method
        """
        return CompletionsWrapper(self)

    @property
    def chat(self):
        """
        Provide access to OpenAI chat API, aligned with OpenAI client interface.
        Allows async calls like: await client.chat.completions.create(...)
        The create method calls our generate method internally.

        Returns:
            ChatWrapper: Wrapper object with completions attribute
        """
        return ChatWrapper(self)


class CompletionsWrapper:
    """
    Wrapper class for OpenAI completions API that calls our async completion method.
    Provides an OpenAI-compatible interface: await client.completions.create(...)
    """

    def __init__(self, client: "InferenceServiceClient"):
        """
        Initialize the completions wrapper.

        Args:
            client: The InferenceServiceClient instance
        """
        self._client = client

    async def create(self, prompt: str, **kwargs) -> Completion:
        """
        Create a completion by calling our completion method.
        This method aligns with OpenAI's completions.create interface, but is async.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters for completion

        Returns:
            Completion: OpenAI Completion object with northrl_train attribute attached
        """
        result_dict = dict(await self._client.completion(prompt=prompt, **kwargs))
        northrl_train = result_dict.pop("northrl_train")

        # Reconstruct Completion object from dict
        completion = Completion.model_validate(result_dict)

        # Attach northrl_train as an attribute
        completion.northrl_train = northrl_train  # type: ignore

        return completion


class ChatCompletionsWrapper:
    """
    Wrapper class for OpenAI chat completions API that calls our async generate method.
    Provides an OpenAI-compatible interface: await client.chat.completions.create(...)
    """

    def __init__(self, client: "InferenceServiceClient"):
        """
        Initialize the chat completions wrapper.

        Args:
            client: The InferenceServiceClient instance
        """
        self._client = client

    async def create(self, messages: list[dict[str, Any]], **kwargs) -> ChatCompletion:
        """
        Create a chat completion by calling our generate method.
        This method aligns with OpenAI's chat.completions.create interface, but is async.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for generation

        Returns:
            ChatCompletion: OpenAI ChatCompletion object with northrl_train attribute attached
        """
        result_dict = dict(await self._client.generate(messages=messages, **kwargs))
        northrl_train = result_dict.pop("northrl_train")

        # Reconstruct ChatCompletion object from dict
        completion = ChatCompletion.model_validate(result_dict)

        # Attach northrl_train as an attribute
        completion.northrl_train = northrl_train  # type: ignore

        return completion


class ChatWrapper:
    """
    Wrapper class for OpenAI chat API that provides completions access.
    Provides OpenAI-compatible interface: client.chat.completions.create(...)
    """

    def __init__(self, client: "InferenceServiceClient"):
        """
        Initialize the chat wrapper.

        Args:
            client: The InferenceServiceClient instance
        """
        self._client = client
        self.completions = ChatCompletionsWrapper(client)
