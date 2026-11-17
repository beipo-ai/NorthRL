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

"""最小 E2B 任务：调用一次模型，再用后上传的隐藏答案评分。"""

from pathlib import Path

import httpx

from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)


async def run_agent(
    input_data: RolloutInput,
    workspace: Path,
    model_environment: dict[str, str],
) -> AgentResult:
    """通过平台 gateway 调用模型；平台负责启动服务和记录训练轨迹。"""
    del workspace
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(
            model_environment["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + model_environment["OPENAI_API_KEY"]},
            json={
                "model": "northrl",
                "messages": [{"role": "user", "content": input_data.task["instruction"]}],
            },
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    return AgentResult(final_answer=answer)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
    """平台在 agent 结束、trace 保存、隐藏答案上传后调用此函数。"""
    del input_data, workspace
    try:
        expected = (evaluation_assets / "expected.txt").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise RolloutEnvironmentError("cannot read expected.txt evaluation asset") from exc

    answer = agent_result.final_answer
    matched = answer is not None and answer.strip() == expected
    return EvaluationResult(
        reward=1.0 if matched else 0.0,
        status="success" if matched else "task_failed",
    )
