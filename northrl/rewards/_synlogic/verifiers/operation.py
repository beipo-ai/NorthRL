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

from ..base import SynLogicData as Data
from ..base import (
    SynLogicDependencyError,
    Verifier,
)

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic

_MATH_VERIFY_TIMEOUT_S = 10


class OperationVerifier(Verifier):
    """
    验证器用于物品计数游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        try:
            import math_verify  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise SynLogicDependencyError(
                "SynLogic operation requires math-verify==0.9.0"
            ) from error

        try:
            ground_truth = math_verify.parse(
                data.answer,
                parsing_timeout=_MATH_VERIFY_TIMEOUT_S,
            )
            parsed_answer = math_verify.parse(
                test_answer,
                parsing_timeout=_MATH_VERIFY_TIMEOUT_S,
            )
            if parsed_answer is None:
                return False
            return math_verify.verify(
                ground_truth,
                parsed_answer,
                timeout_seconds=_MATH_VERIFY_TIMEOUT_S,
            )
        except Exception:  # math-verify treats malformed model answers as parse failures
            return False

    def extract_answer(self, test_solution):
        answer_str = test_solution
        last_box_index = answer_str.rfind("\\boxed{")
        if last_box_index == -1:
            return None
        start_index = last_box_index + len("\\boxed{")
        bracket_stack = 1
        end_index = start_index
        while end_index < len(answer_str) and bracket_stack > 0:
            if answer_str[end_index] == "{":
                bracket_stack += 1
            elif answer_str[end_index] == "}":
                bracket_stack -= 1
            end_index += 1
        if bracket_stack != 0:
            return None
        latex_content = answer_str[start_index : end_index - 1].strip()
        return latex_content
