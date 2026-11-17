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


class DyckLanguageReasoningErrorsVerifier(Verifier):
    """
    Dyck语言推理错误识别验证器
    """

    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确

        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案字符串
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
            correct_indices = data.metadata["error_indices"]
            _expected_answer = self._format_answer(correct_indices)
            if (
                "不确定" in test_answer
                or "不知道" in test_answer
                or "unclear" in test_answer.lower()
            ):
                return False
            cleaned_test_answer = self._standardize_answer(test_answer)
            if not correct_indices and (
                cleaned_test_answer == ""
                or cleaned_test_answer.lower()
                in ["无问题", "no", "无错误", "no error", "no errors", "no mistakes", "all correct"]
            ):
                is_correct = True
            else:
                test_error_indices = self._extract_error_indices(cleaned_test_answer)
                expected_error_indices = set(correct_indices)
                is_correct = test_error_indices == expected_error_indices
            if is_correct:
                pass
            else:
                pass
            return is_correct
        except Exception:
            return False

    def _standardize_answer(self, answer: str) -> str:
        """
        标准化答案字符串

        @param answer: 原始答案字符串
        @return: 标准化后的答案字符串
        """
        if not answer or answer.strip() == "":
            return ""
        if answer.lower() in [
            "无问题",
            "no",
            "无错误",
            "no error",
            "no errors",
            "no mistakes",
            "all correct",
        ]:
            return ""
        answer = answer.replace("，", ",")
        answer = re.sub("[^0-9,]", "", answer)
        return answer

    def _extract_error_indices(self, answer: str) -> set:
        """
        从答案字符串中提取错误索引集合

        @param answer: 答案字符串
        @return: 错误索引集合
        """
        if not answer or answer.strip() == "":
            return set()
        answer = answer.replace("，", ",")
        try:
            return set((int(idx) for idx in answer.split(",") if idx and idx.strip().isdigit()))
        except ValueError:
            return set()

    def _format_answer(self, error_indices: list) -> str:
        """
        格式化答案字符串

        @param error_indices: 错误步骤的索引列表
        @return: 格式化的答案字符串
        """
        if not error_indices:
            return ""
        sorted_indices = sorted(error_indices)
        return ",".join(map(str, sorted_indices))

    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取错误索引

        @param test_solution: 模型的完整回答
        """
        answer_str = test_solution
        return answer_str
