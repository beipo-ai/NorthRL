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


class CryptarithmVerifier(Verifier):
    """
    验证器用于检查密码算术游戏的答案是否正确
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        test_solution = test_answer
        try:
            test_answer = self.extract_answer(test_solution)
            correct_answer = data.answer
            test_answer = test_answer.strip()
            test_answer = test_answer.replace(" ", "")
            correct_answer = correct_answer.replace(" ", "")
            is_correct = test_answer == correct_answer
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
        if not test_solution:
            return ""
        test_solution = test_solution.replace("THE ANSWER IS", "The answer is")
        test_solution = test_solution.replace("答案是：", "答案是:")
        test_solution = test_solution.replace("答案：", "答案:")
        equation_patterns = [
            "(\\d+(?:\\s*(?:\\+|\\-|\\*)\\s*\\d+)+\\s*=\\s*-?\\d+)",
            "(\\d+(?:\\s*(?:\\+|\\-|\\*)\\s*\\d+)+\\s*=\\s*-?\\d+)[.。]*$",
            "(\\d+\\s*(?:\\+|\\-|\\*)\\s*\\d+\\s*=\\s*-?\\d+)",
        ]
        for pattern in equation_patterns:
            matches = re.findall(pattern, test_solution)
            if matches:
                return matches[-1].strip()
        cn_patterns = [
            "答案是[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "答案[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "我的答案是[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "正确答案[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "数字等式是[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "数字等式为[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "等式为[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "等式是[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "结果是[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "结果为[：:]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
        ]
        en_patterns = [
            "[Tt]he answer is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]he answer[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Aa]nswer[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Mm]y answer is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]he final answer is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]he equation is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]he result is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]he numeric equation is[：:=]\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Tt]herefore,\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
            "[Ss]o,\\s*([0-9\\s\\+\\-\\*=]+)[.。]*$",
        ]
        patterns = cn_patterns + en_patterns
        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                answer = matches[-1].strip()
                answer = answer.replace("$", "").replace("。", "").replace(".", "")
                if re.match("\\d+(?:\\s*(?:\\+|\\-|\\*)\\s*\\d+)+\\s*=\\s*-?\\d+", answer):
                    return answer
        lines = test_solution.strip().split("\n")
        for line in reversed(lines):
            equation_match = re.search("\\d+(?:\\s*(?:\\+|\\-|\\*)\\s*\\d+)+\\s*=\\s*-?\\d+", line)
            if equation_match:
                return equation_match.group(0)
        general_equation_pattern = "\\d+\\s*(?:\\+|\\-|\\*)\\s*\\d+\\s*=\\s*-?\\d+"
        all_equations = re.findall(general_equation_pattern, test_solution)
        if all_equations:
            return all_equations[-1]
        return ""
