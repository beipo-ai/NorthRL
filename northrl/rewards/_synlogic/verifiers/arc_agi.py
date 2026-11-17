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

from ..base import SynLogicData as Data
from ..base import Verifier
from ..safe_eval import literal

# Adapted from the MIT-licensed MiniMax-AI SynLogic project:
# https://github.com/MiniMax-AI/SynLogic


def _preprocess_answer(answer: str) -> object | None:
    """预处理答案文本"""
    if not answer:
        return None

    processed = answer.strip().lower()
    processed = processed.split("```json")[-1]
    processed = processed.split("```")[0].strip()
    try:
        processed = json.loads(processed)
    except (json.JSONDecodeError, TypeError):
        try:
            processed = literal(processed)
        except (SyntaxError, TypeError, ValueError):
            return None

    return processed


class ArcAGIVerifier(Verifier):
    """ArcAGI任务的验证器"""

    def verify(self, data: Data, test_answer: str) -> float:
        try:
            # 提取和预处理模型答案
            parsed_answer = self.extract_answer(test_answer)

            # 获取并预处理参考答案
            ground_truth = data.answer

            # 进行模糊匹配
            correct = parsed_answer == ground_truth

            return 1.0 if correct else 0.0
        except Exception:
            return 0.0

    def extract_answer(self, test_solution: str) -> object | None:
        """从完整答案中提取最终答案"""
        # if THOUGHT_DELIMITER_END in test_solution:
        #     test_solution = test_solution.split(THOUGHT_DELIMITER_END)[1]

        return _preprocess_answer(test_solution)
