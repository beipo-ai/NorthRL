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


class KukurasuVerifier(Verifier):
    """
    Verifier for Kukurasu puzzle
    数独拼图验证器
    """

    def verify(self, data: Data, test_answer: str, **kwargs) -> bool:
        del kwargs
        try:
            grid = self.extract_answer(test_answer)
            row_sums = data.metadata["row_sums"]
            col_sums = data.metadata["col_sums"]
            n = data.metadata["n"]
            m = data.metadata["m"]
            if len(grid) != n:
                return False
            for row in grid:
                if len(row) != m:
                    return False
                for cell in row:
                    if cell not in ["1", "X"]:
                        return False
            calculated_row_sums = []
            for i, row in enumerate(grid):
                row_sum = 0
                for j, cell in enumerate(row):
                    if cell == "1":
                        row_sum += j + 1
                calculated_row_sums.append(row_sum)
            calculated_col_sums = []
            for j in range(m):
                col_sum = 0
                for i in range(n):
                    if grid[i][j] == "1":
                        col_sum += i + 1
                calculated_col_sums.append(col_sum)
            if calculated_row_sums != row_sums:
                return False
            if calculated_col_sums != col_sums:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """Extract the answer grid from the model's response
        从模型的响应中提取答案网格"""
        response = test_solution
        grid_pattern = '\\[\\s*\\[(?:\\s*"[X1]"\\s*,\\s*)*\\s*"[X1]"\\s*\\]\\s*(?:,\\s*\\[(?:\\s*"[X1]"\\s*,\\s*)*\\s*"[X1]"\\s*\\]\\s*)*\\]'
        match = re.search(grid_pattern, response)
        if match:
            try:
                grid_str = match.group(0)
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass
        return ""
