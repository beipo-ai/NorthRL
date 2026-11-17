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
from typing import List

from ..base import SynLogicData as Data
from ..base import Verifier
from ..safe_eval import literal

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class ArrowMazeVerifier(Verifier):
    """
    箭头迷宫游戏验证器

    验证条件:
    1. 判断answer grid的大小是否和question grid一致
    2. 判断answer grid中数字格子是否和question grid中数字格子一致
    3. 判断question grid空格（"X"）在answer grid中是否被箭头填满
    4. 判断箭头符号是否合法：
       上（↑）、下（↓）、左（←）、右（→）或对角线方向（↖、↗、↘、↙）
    5. 判断answer grid中非空格（"X"）和非数字的部分，即预填的箭头，是否和question grid一致
    6. 迷宫有个隐藏的条件是所有箭头都能被射线箭头串覆盖到
    7. 每个数字起点发出的射线箭头串总长度等于该数字
    """

    VALID_ARROWS = {"↑", "↓", "←", "→", "↖", "↗", "↘", "↙"}
    ARROWS_DIRECTIONS = {
        "↑": (-1, 0),
        "↓": (1, 0),
        "←": (0, -1),
        "→": (0, 1),
        "↖": (-1, -1),
        "↗": (-1, 1),
        "↘": (1, 1),
        "↙": (1, -1),
    }

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证箭头迷宫的答案是否正确

        @param data: 游戏数据
        @param test_solution_str: 测试答案字符串 (JSON格式的二维数组)
        @return: 答案是否正确
        """
        test_answer_str = self.extract_answer(test_answer)
        if not test_answer_str:
            return False
        try:
            answer_grid = json.loads(test_answer_str)
            question_grid = data.metadata["maze"]
            if not self._verify_grid_size(answer_grid, question_grid):
                return False
            if not self._verify_number_positions(answer_grid, question_grid):
                return False
            if not self._verify_all_blanks_filled(answer_grid, question_grid):
                return False
            if not self._verify_arrow_symbols(answer_grid):
                return False
            if not self._verify_prefilled_arrows(answer_grid, question_grid):
                return False
            if not self._verify_arrow_rays(answer_grid):
                return False
            if not self._verify_number_rays(answer_grid):
                return False
            return True
        except Exception:
            return False

    def _verify_grid_size(
        self, test_answer: List[List[str]], question_grid: List[List[str]]
    ) -> bool:
        """
        验证答案网格大小是否与题目一致

        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 网格大小是否一致
        """
        if len(test_answer) != len(question_grid):
            return False
        for i, answer_row in enumerate(test_answer):
            if len(answer_row) != len(question_grid[i]):
                return False
        return True

    def _verify_number_positions(
        self, test_answer: List[List[str]], question_grid: List[List[str]]
    ) -> bool:
        """
        验证答案中数字位置和值是否与题目一致

        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 数字位置和值是否一致
        """
        for i, question_row in enumerate(question_grid):
            for j, question_cell in enumerate(question_row):
                if question_cell.isdigit():
                    if test_answer[i][j] != question_cell:
                        return False
        return True

    def _verify_all_blanks_filled(
        self, test_answer: List[List[str]], question_grid: List[List[str]]
    ) -> bool:
        """
        验证所有空格是否都被填满

        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 所有空格是否被填满
        """
        for i, question_row in enumerate(question_grid):
            for j, question_cell in enumerate(question_row):
                if question_cell == "X" and test_answer[i][j] == "X":
                    return False
        return True

    def _verify_arrow_symbols(self, test_answer: List[List[str]]) -> bool:
        """
        验证箭头符号是否合法

        @param test_answer: 测试答案网格
        @return: 箭头符号是否合法
        """
        for answer_row in test_answer:
            for cell in answer_row:
                if not cell.isdigit() and cell != "X" and (cell not in self.VALID_ARROWS):
                    return False
        return True

    def _verify_prefilled_arrows(
        self, test_answer: List[List[str]], question_grid: List[List[str]]
    ) -> bool:
        """
        验证预填的箭头是否与题目一致

        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 预填箭头是否一致
        """
        for i, question_row in enumerate(question_grid):
            for j, cell in enumerate(question_row):
                if not cell.isdigit() and cell != "X":
                    if test_answer[i][j] != cell:
                        return False
        return True

    def _verify_arrow_rays(self, test_answer: List[List[str]]) -> bool:
        """
        验证所有箭头是否都能被射线箭头串覆盖到

        @param test_answer: 测试答案网格
        @return: 所有箭头是否都能被射线覆盖
        """
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0
        covered = [[False for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    covered[i][j] = True
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
                        ni, nj = (i + di, j + dj)
                        while 0 <= ni < n and 0 <= nj < m and (test_answer[ni][nj] == arrow_symbol):
                            covered[ni][nj] = True
                            ni += di
                            nj += dj
        for i in range(n):
            for j in range(m):
                if test_answer[i][j] in self.VALID_ARROWS and (not covered[i][j]):
                    return False
        return True

    def _verify_number_rays(self, test_answer: List[List[str]]) -> bool:
        """
        验证每个数字起点发出的射线箭头串总长度是否等于该数字

        @param test_answer: 测试答案网格
        @return: 每个数字的射线箭头串是否符合要求
        """
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    number = int(test_answer[i][j])
                    arrow_count = self._count_arrow_rays(test_answer, i, j)
                    if arrow_count != number:
                        return False
        return True

    def _count_arrow_rays(self, grid: List[List[str]], i: int, j: int) -> int:
        """
        计算从数字出发的所有射线箭头串中箭头总数

        @param grid: 网格
        @param i: 数字行索引
        @param j: 数字列索引
        @return: 箭头总数
        """
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        count = 0
        for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
            ni, nj = (i + di, j + dj)
            ray_length = 0
            while 0 <= ni < n and 0 <= nj < m and (grid[ni][nj] == arrow_symbol):
                ray_length += 1
                ni += di
                nj += dj
            count += ray_length
        return count

    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取答案

        @param test_solution: 模型的完整回答
        @return: 提取的答案 (JSON格式的二维数组)
        """
        if not test_solution:
            return ""
        code_block_patterns = [
            "```python\\s*\\n(.*?\\[.*?\\].*?)\\n```",
            "```\\s*\\n(.*?\\[.*?\\].*?)\\n```",
            "```(.*?\\[.*?\\].*?)```",
        ]
        for pattern in code_block_patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                code_block = matches[-1].strip()
                try:
                    grid = literal(code_block)
                    if isinstance(grid, list) and all((isinstance(row, list) for row in grid)):
                        return json.dumps(grid)
                except Exception:
                    continue
        list_pattern = "\\[\\s*\\[.*?\\]\\s*\\]"
        matches = re.findall(list_pattern, test_solution, re.DOTALL)
        if matches:
            try:
                grid = literal(matches[-1])
                if isinstance(grid, list) and all((isinstance(row, list) for row in grid)):
                    return json.dumps(grid)
            except Exception:
                pass
        return ""
