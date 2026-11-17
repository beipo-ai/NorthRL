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

import ast
import re

from ..base import SynLogicData as Data
from ..base import Verifier

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class NumbrixVerifier(Verifier):
    """
    Numbrix 游戏的验证器
    验证提交的解答是否符合 Numbrix 游戏规则
    """

    def verify(self, data: Data, test_answer: str):
        test_solution = test_answer
        try:
            test_grid = self.extract_answer(test_solution)
            if not test_grid:
                return False
            original_grid = data.metadata["grid"]
            n = len(original_grid)
            n_squared = n * n
            if len(test_grid) != n or any((len(row) != n for row in test_grid)):
                return False
            flattened_grid = [cell for row in test_grid for cell in row]
            if sorted(flattened_grid) != list(range(1, n_squared + 1)):
                return False
            for i in range(n):
                for j in range(n):
                    if original_grid[i][j] != "X" and test_grid[i][j] != original_grid[i][j]:
                        return False
            for num in range(1, n_squared):
                current_pos = None
                next_pos = None
                for i in range(n):
                    for j in range(n):
                        if test_grid[i][j] == num:
                            current_pos = (i, j)
                        elif test_grid[i][j] == num + 1:
                            next_pos = (i, j)
                if current_pos is None or next_pos is None:
                    return False
                i1, j1 = current_pos
                i2, j2 = next_pos
                manhattan_distance = abs(i1 - i2) + abs(j1 - j2)
                if manhattan_distance != 1:
                    return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str, strict=False):
        """从模型回答中提取网格"""
        del strict
        try:
            pattern = "\\[\\s*\\[\\s*\\d+.*?\\]\\s*\\]"
            matches = re.finditer(pattern, test_solution, re.DOTALL)
            match = None
            for m in matches:
                match = m
            if not match:
                return None
            grid_text = match.group(0)
            grid_text = grid_text.replace("'", "").replace('"', "")
            grid = ast.literal_eval(grid_text)
            if not isinstance(grid, list) or not all((isinstance(row, list) for row in grid)):
                return None
            if not all((isinstance(cell, int) for row in grid for cell in row)):
                return None
            return grid
        except Exception:
            return None
