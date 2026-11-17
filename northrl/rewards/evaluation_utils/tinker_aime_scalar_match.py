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

"""tinker-style AIME answer extraction + integer scalar match.

Replicates tinker's AIME grading (https://github.com/thinking-machines-lab/tinker):
strip ``<think>...</think>``, take the first ``\\boxed{...}``, extract the integer,
and compare ``int(float(pred)) == int(gold)``. ``RewardMathFn`` / LightEval do a
richer sympy/latex match; this is the exact-integer matcher selected by
``judge_mode="tinker_aime_scalar_match"`` in ``northrl.rewards.math``.
"""

from __future__ import annotations

import math
import re


def strip_thinking(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<\|im_end\|>.*$", "", stripped, flags=re.DOTALL)
    return stripped


def extract_boxed(text: str) -> str | None:
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def extract_number(text: str) -> str:
    cleaned = re.sub(r"\\text\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def extract_gsm8k_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return extract_number(boxed)

    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return extract_number(hash_match.group(1))

    answer_match = re.search(r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    bold_numbers = re.findall(r"\*\*\$?([-]?\d+[,\d]*\.?\d*)", text)
    if bold_numbers:
        return bold_numbers[-1].replace(",", "").rstrip(".")

    numbers = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def is_correct(prediction: object, ground_truth: object) -> bool:
    """True iff the prediction's tinker-extracted integer equals the gold integer."""
    response = strip_thinking("" if prediction is None else str(prediction))
    boxed = extract_boxed(response)
    extracted_str = extract_number(boxed) if boxed else extract_gsm8k_answer(response)
    try:
        extracted_float = float(extracted_str)
        if not math.isfinite(extracted_float):
            return False
        extracted_val = int(extracted_float)
        expected = int(str(ground_truth).strip())
        return extracted_val == expected
    except (OverflowError, ValueError, TypeError):
        return False
