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

import json
import re

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class TimeSequenceVerifier(Verifier):
    """
    验证器用于验证 time sequence 的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案，格式为数字列表
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            test_answer = self.extract_answer(test_solution)
            metadata = data.metadata
            true_answers = metadata["records"]["answers"]
            try:
                test_list = json.loads(test_answer.replace("，", ","))
            except Exception:
                return False
            try:
                if test_list[0] != true_answers["answer_maxLen"]:
                    return False
                if test_list[1] != true_answers["answer_nums"]:
                    return False
            except Exception:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案（矩阵）

        @param test_solution: 模型的完整回答
        @return: 提取答案列表
        """
        if not test_solution:
            return ""
        matrix_pattern = "\\[.*?\\]"
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)
        if matrix_matches:
            return matrix_matches[-1].strip()
        return ""
