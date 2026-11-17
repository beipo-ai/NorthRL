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
from typing import List, Optional, Tuple

import numpy as np

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class FutoshikiVerifier(Verifier):
    """Verifier for Futoshiki"""

    def __init__(self):
        """Initialize the verifier."""

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型提供的回答是否正确：
        1. 数独规则：每行每列1-N各出现一次
        2. 预填充数字：模型保留了题目中的预填充数字
        3. 不等式约束：所有不等式约束都被满足

        注意：不需要和生成数据中的答案完全一致，只要满足上述三个条件即为正确。

        Args:
            data: 包含题目和元数据的Data对象
            test_solution: 模型给出的回答字符串

        Returns:
            bool: 回答是否正确
        """
        test_solution = test_answer
        answer = self.extract_answer(test_solution)
        if answer is None:
            return False
        grid_size = data.metadata["grid_size"]
        if not self.check_answer_format(answer, grid_size):
            return False
        if not self.check_rows_and_columns(answer, grid_size):
            return False
        prefilled_coords = data.metadata["prefilled_coords"]
        question = data.question
        prefilled_values: list[int | None] = []
        grid_lines = []
        in_grid = False
        for line in question.split("\n"):
            if "Current puzzle:" in line:
                in_grid = True
                continue
            if in_grid and line.strip() and ("X" in line):
                grid_lines.append(line.strip())
            elif in_grid and grid_lines and line.strip() and ("X" not in line):
                in_grid = False
        for row, col in prefilled_coords:
            if row < len(grid_lines):
                grid_line = grid_lines[row].split()
                if col < len(grid_line) and grid_line[col] != "X":
                    try:
                        prefilled_values.append(int(grid_line[col]))
                    except ValueError:
                        prefilled_values.append(None)
                else:
                    prefilled_values.append(None)
            else:
                prefilled_values.append(None)
        if not self.check_prefilled_numbers(answer, prefilled_coords, prefilled_values):
            return False
        constraints = data.metadata["constraints"]
        if not self.check_inequality_constraints(answer, constraints):
            return False
        return True

    def extract_answer(self, test_solution: str) -> Optional[np.ndarray]:
        """Extract the answer from the test solution string."""
        try:
            test_solution = test_solution.strip()
            answer_matches = re.findall("\\[\\[([^\\]]+)\\]\\]", test_solution)
            if answer_matches:
                answer_str = answer_matches[-1]
                rows = answer_str.split(",")
                grid = []
                for row in rows:
                    numbers = re.findall("\\d+", row)
                    row_numbers = [int(num) for num in numbers]
                    grid.append(row_numbers)
                if len(grid) > 0 and all((len(row) == len(grid) for row in grid)):
                    return np.array(grid)
            if test_solution.startswith("[[") and test_solution.endswith("]]"):
                test_solution = test_solution[2:-2]
                rows = test_solution.split(",")
                grid = []
                for row in rows:
                    numbers = row.strip().split()
                    row_numbers = [int(num) for num in numbers]
                    grid.append(row_numbers)
                return np.array(grid)
            return None
        except Exception:
            return None

    def check_answer_format(self, answer: np.ndarray, grid_size: int) -> bool:
        """Check if the answer has the correct format."""
        if answer.shape != (grid_size, grid_size):
            return False
        if answer.dtype != np.int64:
            return False
        return True

    def check_rows_and_columns(self, answer: np.ndarray, grid_size: int) -> bool:
        """Check if each row and column contains all numbers from 1 to n."""
        for row in answer:
            if not self.check_sequence(row, grid_size):
                return False
        for col in answer.T:
            if not self.check_sequence(col, grid_size):
                return False
        return True

    def check_sequence(self, sequence: np.ndarray, grid_size: int) -> bool:
        """Check if a sequence contains all numbers from 1 to n."""
        required = set(range(1, grid_size + 1))
        actual = set(sequence)
        return required == actual

    def check_prefilled_numbers(
        self,
        answer: np.ndarray,
        prefilled_coords: List[Tuple[int, int]],
        prefilled_values: list[int | None],
    ) -> bool:
        """Check if the prefilled numbers are preserved in the answer."""
        for (row, col), value in zip(prefilled_coords, prefilled_values):
            if value is not None and answer[row, col] != value:
                return False
        return True

    def check_inequality_constraints(
        self, answer: np.ndarray, constraints: List[Tuple[Tuple[int, int], Tuple[int, int], str]]
    ) -> bool:
        """Check if all inequality constraints are satisfied."""
        for coord1, coord2, sign in constraints:
            row1, col1 = coord1
            row2, col2 = coord2
            num1 = answer[row1, col1]
            num2 = answer[row2, col2]
            if sign == ">":
                if num1 <= num2:
                    return False
            elif sign == "<":
                if num1 >= num2:
                    return False
        return True
