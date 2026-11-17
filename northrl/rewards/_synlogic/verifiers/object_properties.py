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


class ObjectPropertiesVerifier(Verifier):
    """
    验证器用于物品拥有游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = int(data.answer)
            parsed_answer = int(self.extract_answer(test_answer))
            if parsed_answer is None:
                return False
            return int(parsed_answer) == ground_truth
        except Exception:
            return False

    def extract_answer(self, test_solution):
        answer_str = test_solution
        last_box_index = answer_str.rfind("\\boxed{")
        if last_box_index == -1:
            return None
        last_box_substring = answer_str[last_box_index:]
        box_pattern = "\\\\boxed\\{([^}]*)\\}"
        match = re.search(box_pattern, last_box_substring)
        if match:
            return match.group(1).strip()
        return None
