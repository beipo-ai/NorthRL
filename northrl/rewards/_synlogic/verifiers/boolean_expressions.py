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


class BooleanExpressionsVerifier(Verifier):
    """
    验证器用于布尔表达式游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        try:
            test_answer = self.extract_answer(test_answer)
            if test_answer is None:
                return False
            test_answer_letters = re.findall("[a-zA-Z]", test_answer)
            ground_truth_letters = re.findall("[a-zA-Z]", data.answer)
            test_answer_letters = self.lower(test_answer_letters)
            ground_truth_letters = self.lower(ground_truth_letters)
            test_set = set(test_answer_letters)
            ground_truth_set = set(ground_truth_letters)
            return test_set == ground_truth_set
        except Exception:
            return False

    def lower(self, answer_list):
        return [answer.lower() for answer in answer_list]

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
