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

"""Shared NorthRL exceptions."""


class WeightVersionMismatchError(RuntimeError):
    """Raised when a rollout request used an unexpected remote weight version."""


class ContextLengthExceededError(RuntimeError):
    """Sampling cannot fit into the model's context window."""

    def __init__(
        self,
        prompt_tokens: int | None = None,
        max_tokens: int | None = None,
        max_sequence_length: int | None = None,
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_sequence_length = max_sequence_length
        super().__init__(
            f"Context length exceeded: prompt_tokens={prompt_tokens}, "
            f"requested_max_tokens={max_tokens}, max_sequence_length={max_sequence_length}"
        )

    @property
    def length_details(self) -> dict[str, int]:
        """Serializable context budget for rollout completion and diagnostics."""
        details = {
            "prompt_tokens": self.prompt_tokens,
            "requested_max_tokens": self.max_tokens,
            "max_sequence_length": self.max_sequence_length,
        }
        # Gateway errors may identify the condition without reporting counts.
        return {key: value for key, value in details.items() if value is not None}
