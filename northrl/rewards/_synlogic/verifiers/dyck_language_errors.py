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


class DyckLanguageErrorsVerifier(Verifier):
    """
    验证器用于检查括号闭合错误识别游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
            if data.metadata["is_valid"]:
                correct_answer = "-1"
            else:
                correct_answer = str(data.metadata["first_error_pos"])
            test_answer = test_answer.strip()
            if correct_answer == "-1":
                is_correct = test_answer == "-1"
            else:
                try:
                    is_correct = int(test_answer) == int(correct_answer)
                except (ValueError, TypeError):
                    is_correct = False
            if is_correct:
                pass
            else:
                pass
            return is_correct
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案

        @param test_solution: 模型的完整回答
        @return: 提取的答案
        """
        answer_str = test_solution
        if answer_str is None:
            solution = test_solution.strip() if test_solution else ""
            numbers = re.findall("-?\\d+", solution)
            if numbers:
                if "-1" in numbers:
                    return "-1"
                for num in numbers:
                    if num.isdigit() and int(num) >= 0:
                        return num
                return numbers[0]
            return ""
        elif any((keyword in answer_str.lower() for keyword in ["合法", "valid", "correct"])):
            return "-1"
        else:
            return answer_str
