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

import numpy as np

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class SurvoVerifier(Verifier):
    """
    验证器用于检查Survo矩阵填充游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的矩阵答案字符串，格式为二维列表
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            test_answer = self.extract_answer(test_solution)
            metadata = data.metadata
            original_matrix = np.array(metadata["original_matrix"])
            candidate_numbers = metadata["candidate_numbers"]
            n = metadata["n"]
            try:
                test_matrix = json.loads(test_answer.replace("'", '"'))
                test_matrix = np.array(test_matrix)
            except Exception:
                test_answer = test_answer.strip()
                if test_answer.startswith("[") and test_answer.endswith("]"):
                    cleaned_answer = test_answer.replace("'", '"')
                    try:
                        test_matrix = json.loads(cleaned_answer)
                        test_matrix = np.array(test_matrix)
                    except Exception:
                        return False
                else:
                    return False
            if test_matrix.shape != original_matrix.shape:
                return False
            for i in range(n):
                for j in range(n):
                    if original_matrix[i, j] != 0 and original_matrix[i, j] != test_matrix[i, j]:
                        return False
            filled_numbers = []
            for i in range(n):
                for j in range(n):
                    if original_matrix[i, j] == 0:
                        filled_numbers.append(test_matrix[i, j])
            sorted_filled = sorted(filled_numbers)
            sorted_candidates = sorted(candidate_numbers)
            if sorted_filled != sorted_candidates:
                return False
            for i in range(n - 1):
                row_sum = sum(test_matrix[i, 0 : n - 1])
                expected_sum = test_matrix[i, n - 1]
                if row_sum != expected_sum:
                    return False
            for j in range(n - 1):
                col_sum = sum(test_matrix[0 : n - 1, j])
                expected_sum = test_matrix[n - 1, j]
                if col_sum != expected_sum:
                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案（矩阵）

        @param test_solution: 模型的完整回答
        @return: 提取的矩阵答案字符串
        """
        if not test_solution:
            return ""
        code_block_pattern = "```python\\s*([\\s\\S]*?)\\s*```"
        code_matches = re.findall(code_block_pattern, test_solution)
        if code_matches:
            matrix_str = code_matches[-1].strip()
            return matrix_str
        general_code_block = "```([\\s\\S]*?)```"
        general_matches = re.findall(general_code_block, test_solution)
        if general_matches:
            matrix_str = general_matches[-1].strip()
            return matrix_str
        matrix_pattern = "\\[\\s*\\[.*?\\]\\s*\\]"
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)
        if matrix_matches:
            return matrix_matches[-1].strip()
        return ""
