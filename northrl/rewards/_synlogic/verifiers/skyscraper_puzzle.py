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


class SkyscraperPuzzleVerifier(Verifier):
    """
    摩天楼游戏验证器，用于验证模型提供的解答是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否符合摩天楼游戏的规则

        @param data: 包含游戏信息的Data对象
        @param test_answer: 游戏类提取的网格数据
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            metadata = data.metadata
            n = metadata["n"]
            top = metadata["top"]
            bottom = metadata["bottom"]
            left = metadata["left"]
            right = metadata["right"]
            self.n = n
            test_answer = self.extract_answer(test_solution)
            grid = test_answer
            if isinstance(grid, str):
                return False
            for row in grid:
                pass
            if len(grid) != n or any((len(row) != n for row in grid)):
                return False
            for i in range(n):
                for j in range(n):
                    if not isinstance(grid[i][j], int) or grid[i][j] < 1 or grid[i][j] > n:
                        return False
            for i in range(n):
                if len(set(grid[i])) != n:
                    return False
            for j in range(n):
                column = [grid[i][j] for i in range(n)]
                if len(set(column)) != n:
                    return False
            for j in range(n):
                visible_count = self._count_visible_skyscrapers([grid[i][j] for i in range(n)])
                if visible_count != top[j]:
                    return False
            for j in range(n):
                visible_count = self._count_visible_skyscrapers(
                    [grid[i][j] for i in range(n - 1, -1, -1)]
                )
                if visible_count != bottom[j]:
                    return False
            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i])
                if visible_count != left[i]:
                    return False
            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i][::-1])
                if visible_count != right[i]:
                    return False
            return True
        except Exception:
            return False

    def _count_visible_skyscrapers(self, heights):
        """
        计算从一个方向看过去能看到的摩天楼数量

        @param heights: 从观察方向依次排列的摩天楼高度列表
        @return: 可见的摩天楼数量
        """
        visible_count = 0
        max_height = 0
        for height in heights:
            if height > max_height:
                visible_count += 1
                max_height = height
        return visible_count

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取网格数据

        @param test_solution: 模型的完整回答
        @return: 提取的解答网格数据
        """
        try:
            n = self.n
            code_block_pattern = "```python\\s*\\n([\\s\\S]*?)\\n\\s*```"
            code_blocks = re.findall(code_block_pattern, test_solution)
            if code_blocks:
                code_block = code_blocks[0].strip()
                try:
                    grid = ast.literal_eval(code_block)
                    if (
                        isinstance(grid, list)
                        and len(grid) == n
                        and all((isinstance(row, list) and len(row) == n for row in grid))
                    ):
                        return grid
                except Exception:
                    code_without_comments = re.sub("#.*$", "", code_block, flags=re.MULTILINE)
                    try:
                        grid = ast.literal_eval(code_without_comments.strip())
                        if (
                            isinstance(grid, list)
                            and len(grid) == n
                            and all((isinstance(row, list) and len(row) == n for row in grid))
                        ):
                            return grid
                    except Exception:
                        pass
            return test_solution
        except Exception:
            return test_solution
