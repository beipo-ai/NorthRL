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

import numpy as np

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class CalcudokoVerifier(Verifier):
    """
    Calcudoko 游戏的验证器
    """

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取数独解答

        @param test_solution: 模型的完整回答
        @return: 提取的答案 (元组形式的字符串)
        """
        if not test_solution:
            return ""
        code_block_pattern = "```python\\s*([\\s\\S]*?)\\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        if matches:
            python_code = matches[-1].strip()
            return python_code
        tuple_pattern = "\\(\\s*\\(\\s*\\d+\\s*,.*?\\)\\s*\\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return test_solution

    def parse_grid_from_answer(self, test_solution: str, data: Data) -> np.ndarray:
        """
        从模型的回答中解析出网格数据

        @param test_solution: 模型的完整回答
        @param data: 包含问题、元数据等信息的Data对象
        @return: numpy数组形式的网格
        """
        try:
            grid_size = data.metadata.get("grid_size", 4)
            answer = self.extract_answer(test_solution)
            bracket_pattern = "\\[\\[(.*?)\\]\\]"
            bracket_match = re.search(bracket_pattern, test_solution)
            if bracket_match:
                answer = bracket_match.group(1)
            answer = "".join((c for c in answer if c.isdigit() or c in "[]," or c.isspace()))
            answer = answer.replace("，", ",")
            answer = answer.strip("[]")
            rows = [row.strip() for row in answer.split(",")]
            grid: list[list[int]] = []
            for row in rows:
                numbers = [int(n) for n in row.split() if n.strip()]
                if len(numbers) != grid_size:
                    if len(numbers) < grid_size:
                        row_patterns = [
                            "第\\s*\\d+\\s*行\\s*[:：]\\s*([\\d\\s]+)",
                            "行\\s*\\d+\\s*[:：]\\s*([\\d\\s]+)",
                            "Row\\s*\\d+\\s*[:：]\\s*([\\d\\s]+)",
                            "- 第\\d+行[:：]?\\s*([\\d\\s]+)",
                            "- 行\\d+[:：]?\\s*([\\d\\s]+)",
                            "- Row\\d+[:：]?\\s*([\\d\\s]+)",
                        ]
                        found_rows = []
                        for pattern in row_patterns:
                            matches = re.findall(pattern, test_solution)
                            if matches:
                                found_rows.extend(matches)
                        if len(found_rows) == grid_size:
                            grid = []
                            for found_row in found_rows:
                                numbers = [int(n) for n in found_row.split() if n.strip()]
                                if len(numbers) != grid_size:
                                    if len(numbers) < grid_size:
                                        numbers.extend([1] * (grid_size - len(numbers)))
                                    else:
                                        numbers = numbers[:grid_size]
                                grid.append(numbers)
                            break
                    if len(numbers) < grid_size:
                        numbers.extend([1] * (grid_size - len(numbers)))
                    else:
                        numbers = numbers[:grid_size]
                grid.append(numbers)
            if len(grid) != grid_size:
                if len(grid) < grid_size:
                    for _ in range(grid_size - len(grid)):
                        grid.append([1] * grid_size)
                else:
                    grid = grid[:grid_size]
            return np.array(grid)
        except Exception:
            import traceback

            traceback.print_exc()
            try:
                bracket_pattern = "\\[\\[(.*?)\\]\\]"
                bracket_match = re.search(bracket_pattern, test_solution)
                if bracket_match:
                    answer_str = bracket_match.group(1)
                    rows = answer_str.split(",")
                    grid = []
                    for row in rows:
                        numbers = [int(n) for n in row.split() if n.strip()]
                        if len(numbers) != grid_size:
                            if len(numbers) < grid_size:
                                numbers.extend([1] * (grid_size - len(numbers)))
                            else:
                                numbers = numbers[:grid_size]
                        grid.append(numbers)
                    if len(grid) != grid_size:
                        if len(grid) < grid_size:
                            for _ in range(grid_size - len(grid)):
                                grid.append([1] * grid_size)
                        else:
                            grid = grid[:grid_size]
                    return np.array(grid)
            except Exception:
                pass
            return np.array(
                [[i % grid_size + 1 for i in range(grid_size)] for _ in range(grid_size)]
            )

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案字符串
        @return: 回答是否正确的布尔值
        """
        try:
            grid_size = data.metadata.get("grid_size", 4)
            regions = data.metadata.get("regions", [])
            try:
                grid = self.parse_grid_from_answer(test_answer, data)
            except Exception:
                return False
            for row in grid:
                if len(set(row)) != grid_size:
                    return False
                if not all((1 <= n <= grid_size for n in row)):
                    return False
            for col in grid.T:
                if len(set(col)) != grid_size:
                    return False
                if not all((1 <= n <= grid_size for n in col)):
                    return False
            for region in regions:
                cells = region["cells"]
                numbers = []
                for cell in cells:
                    try:
                        if isinstance(cell, str):
                            match = re.search("\\((\\d+)\\s*,\\s*(\\d+)\\)", cell)
                            if match:
                                r, c = (int(match.group(1)), int(match.group(2)))
                        r, c = (r - 1, c - 1)
                        if r < 0 or r >= grid_size or c < 0 or (c >= grid_size):
                            return False
                        numbers.append(grid[r][c])
                    except Exception:
                        return False
                if not all((1 <= n <= grid_size for n in numbers)):
                    return False
                if len(set(numbers)) != len(numbers):
                    return False
                target = region["target"]
                operator = region["operator"]
                try:
                    if operator == "+":
                        sum_result = sum(numbers)
                        valid = sum_result == target
                        if not valid:
                            return False
                    elif operator == "-":
                        if len(numbers) != 2:
                            return False
                        result1 = numbers[0] - numbers[1]
                        result2 = numbers[1] - numbers[0]
                        valid = target in (result1, result2)
                        if not valid:
                            return False
                    elif operator == "*":
                        product_result = np.prod(numbers)
                        valid = product_result == target
                        if not valid:
                            return False
                    else:
                        if len(numbers) != 2:
                            return False
                        if numbers[0] == 0 or numbers[1] == 0:
                            return False
                        result1 = numbers[0] / numbers[1]
                        result2 = numbers[1] / numbers[0]
                        valid = abs(result1 - target) < 0.001 or abs(result2 - target) < 0.001
                        if not valid:
                            return False
                except Exception:
                    return False
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            return False
