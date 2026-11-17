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
from ..safe_eval import comparison

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


class MathPathVerifier(Verifier):
    """
    验证器用于检查math_path填充游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的运算表达式
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
        except Exception:
            return False
        try:
            metadata = data.metadata
            ref_expr = metadata["ref_expr"]
            query_expr = metadata["query_expr"]
            test_tmp = test_answer.replace(" ", "").strip()
            query_tmp = query_expr.replace(" ", "").strip()
            _ref_tmp = ref_expr.replace(" ", "").strip()
            query_nums = [x for x in query_tmp if "0" <= x <= "9" or x == "?"]
            test_nums = [x for x in test_tmp if "0" <= x <= "9"]
            if len(query_nums) != len(test_nums):
                return False
            else:
                for ind, x in enumerate(query_nums):
                    if x == "?":
                        continue
                    if x != test_nums[ind]:
                        return False
            query_symbols = [x for x in query_tmp if x in ["+", "-", "*", "/", "%"]]
            test_symbols = [x for x in test_tmp if x in ["+", "-", "*", "/", "%"]]
            if len(query_symbols) != len(test_symbols):
                return False
            else:
                for ind, x in enumerate(query_symbols):
                    if x != test_symbols[ind]:
                        return False
            try:
                tmp = test_tmp.replace("=", "==")
                if not re.fullmatch(r"[\d\s+\-*/%().=<>!]*", tmp or ""):  # [0717] 表达式白名单守卫
                    return False
                if not comparison(tmp):
                    return False
            except Exception:
                return False
            return True
        except Exception:
            return False

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案（字符表达式）

        @param test_solution: 模型的完整回答
        @return: 提取的矩阵答案字符串
        """
        if not test_solution:
            return ""
        code_block_pattern = "\\[\\[(.*?)\\]\\]"
        code_matches = re.findall(code_block_pattern, test_solution)
        if code_matches:
            operation_expression = code_matches[-1].strip()
            return operation_expression
        return ""
