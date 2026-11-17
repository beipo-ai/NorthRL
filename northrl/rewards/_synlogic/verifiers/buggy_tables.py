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

import re

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class BuggyTableVerifier(Verifier):
    """
    Verifier for the BuggyTable game.
    Checks if the submitted answer matches the expected answer.
    """

    def extract_answer(self, test_solution: str) -> str:
        """
        Public method to extract and normalize an answer string from LLM output.
        Delegates to the private _extract_answer method.

        @param answer: The answer string to normalize
        @return: The normalized answer string
        """
        answer = test_solution
        return self._extract_answer(answer)

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Verify whether the test answer is consistent with the expected answer
        for the buggy table query.

        @param data: Data object containing the expected answer
        @param test_answer: The answer provided by the LLM to verify
        @return: bool indicating whether the answer is correct
        """
        expected_answer = data.answer if data and hasattr(data, "answer") else ""
        if not expected_answer and (not test_answer):
            return True
        normalized_expected = self._extract_answer(expected_answer)
        normalized_test = self._extract_answer(test_answer)
        return normalized_expected == normalized_test

    def _is_raw_numeric_answer(self, value: str) -> bool:
        """
        Check if a string represents a plain numeric answer without additional context.
        This is used to validate raw input format.

        @param value: The string to check
        @return: True if the string is a simple numeric value
        """
        value = value.strip()
        return bool(re.match("^-?\\d+(\\.\\d+)?$", value))

    def _raw_has_exactly_two_decimals(self, value: str) -> bool:
        """
        Check if a raw numeric string has exactly 2 decimal places.
        This is used to validate the format of the raw answer.

        @param value: The string to check
        @return: True if the string has exactly 2 decimal places
        """
        value = value.strip()
        parts = value.replace("-", "", 1).split(".")
        return len(parts) == 2 and len(parts[1]) == 2

    def _is_numeric(self, value: str) -> bool:
        """
        Check if a string represents a valid number (including negative numbers and decimals).

        @param value: The string to check
        @return: True if the string represents a valid number
        """
        value = value.strip()
        if value.startswith("-"):
            value = value[1:]
        return value.replace(".", "", 1).isdigit()

    def _has_exactly_two_decimals(self, value: str) -> bool:
        """
        Check if a number string has exactly 2 decimal places.

        @param value: The number string to check
        @return: True if the number has exactly 2 decimal places
        """
        value = value.strip()
        if value.startswith("-"):
            value = value[1:]
        parts = value.split(".")
        if len(parts) != 2:
            return False
        return len(parts[1]) == 2

    def _extract_answer(self, answer: str) -> str:
        """
        Extract and normalize an answer string from LLM output.
        Only finds values with exactly two decimal places.

        @param answer: The answer string to normalize
        @return: The normalized answer string
        """
        normalized = str(answer).strip() if answer is not None else ""
        exact_matches = re.findall("-?\\d+\\.\\d{2}\\b", normalized)
        if exact_matches:
            return exact_matches[-1]
        return normalized
