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
from ..safe_eval import arithmetic

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class GameOf24Verifier(Verifier):
    """
    Verifier for Game of 24
    """

    def verify(self, data: Data, test_answer: str):
        test_solution = test_answer
        try:
            test_answer = self.extract_answer(test_solution)
            numbers = data.metadata["numbers"]
            operators = data.metadata["operators"]
            target_result = float(data.metadata["result"])
            input_numbers = [str(num) for num in numbers]
            answer_numbers_str = re.sub("[^0-9]", " ", test_answer)
            answer_numbers = [num for num in answer_numbers_str.split() if num]
            answer_wo_numbers_str = re.sub("[0-9\\s]", "", test_answer)
            unknown_chars = []
            for c in answer_wo_numbers_str:
                if c in operators:
                    continue
                if c in ["(", ")"]:
                    continue
                unknown_chars.append(c)
            if len(unknown_chars) > 0:
                return False
            for num in answer_numbers:
                if num not in input_numbers:
                    return False
                if answer_numbers.count(num) > input_numbers.count(num):
                    return False
            if len(answer_numbers) != len(input_numbers):
                return False
            if not re.fullmatch(r"[\d\s+\-*/%().]*", test_answer or ""):  # [0717] 表达式白名单守卫
                return False
            computed_result = arithmetic(test_answer)
            return (
                isinstance(computed_result, (int, float))
                and abs(float(computed_result) - target_result) < 1e-10
            )
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        regex_pattern = "```python.*?```"
        matches = re.findall(regex_pattern, test_solution, re.DOTALL)
        if matches:
            return matches[-1].replace("```python", "").replace("```", "").strip()
        else:
            return ""
