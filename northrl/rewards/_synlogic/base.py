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

"""Small data and verifier contracts adapted from the SynLogic runtime."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


class SynLogicDependencyError(RuntimeError):
    """A required local verifier dependency is missing or incompatible."""


@dataclass(slots=True)
class SynLogicData:
    """The serialized game state consumed by every SynLogic verifier."""

    question: str
    answer: Any
    difficulty: int | float = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    gpt_response: str = ""

    @classmethod
    def from_json_str(cls, value: str) -> "SynLogicData":
        """Decode and validate one ``metadata.game_data_str`` value."""
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("SynLogic game_data_str must decode to an object")
        missing = {"question", "answer", "metadata"} - decoded.keys()
        if missing:
            raise ValueError(f"SynLogic game_data_str is missing fields: {sorted(missing)}")
        if not isinstance(decoded["question"], str):
            raise TypeError("SynLogic game question must be a string")
        game_metadata = decoded["metadata"]
        if game_metadata is None:
            game_metadata = {}
        if not isinstance(game_metadata, dict):
            raise TypeError("SynLogic game metadata must be an object or null")
        return cls(
            question=decoded["question"],
            answer=decoded["answer"],
            difficulty=decoded.get("difficulty", 1),
            metadata=game_metadata,
            gpt_response=decoded.get("gpt_response", ""),
        )


class Verifier(ABC):
    """Interface implemented by one task-specific SynLogic verifier."""

    @abstractmethod
    def verify(self, data: SynLogicData, test_answer: str) -> Any:
        """Return whether ``test_answer`` solves ``data``."""

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Any:
        """Extract the answer representation expected by this verifier."""
