[English](../../../en/developer-guide/05-rollout-workers/custom-workers.md) | 简体中文 | [文档目录](../../index.md)

<a id="creating-custom-rollout-workers"></a>

# 创建自定义 Rollout 工作进程

任务需要自定义提示、多次模型调用或自己的评分逻辑时，在 `BaseRolloutWorker`
上实现 `async rollout(task)`。单次响应配合内置奖励可使用
`rollout.type: single_turn`；沙箱执行使用
[E2B 任务接入](e2b-task-integration.md)中的任务函数。

<a id="recipe-integration"></a>

## 配方集成

将任务代码放在配方旁边：

```text
recipe/my_task/
├── train.yaml
└── worker.py
```

把以下选择字段合并到完整训练配置：

```yaml
rollout:
  type: custom
  custom_module_path: worker.py
  custom_class_name: MySimpleWorker
```

`custom_module_path` 相对于配方 YAML 解析。默认类名为 `CustomRolloutWorker`，
使用其他名称时设置 `custom_class_name`。控制器导入该文件，并用完整配置构造实例。
任务参数可放在 `config.rollout.params`。相邻目录不会自动加入 `sys.path`；
共享辅助包应安装到运行环境中，少量辅助函数可以直接放在 worker 模块里。

<a id="single-response-worker"></a>

## 单响应工作进程

以下实现使用现有 completion 接口，保留实际 token ID 和采样 logprobs，
将答案与任务的 `answer` 比较。将其放在 `worker.py`：

```python
from northrl.rollout_worker import BaseRolloutWorker
from northrl.northrl_types import Trajectory
from typing import Any
import logging

logger = logging.getLogger(__name__)

class MySimpleWorker(BaseRolloutWorker):
    """Custom worker for simple tasks."""

    async def rollout(self, task: dict[str, Any]) -> str | None:
        # 1. Extract and validate task data
        if "question" not in task:
            logger.error("Task missing 'question' field")
            return None

        question = task["question"]
        context = task.get("context", "")

        # 2. Format prompt
        prompt = self._format_prompt(question, context)

        # 3. Call LLM
        # Sampling parameters come from rollout.sampling in the configuration.
        result = await self._inference_client.completion(prompt)
        train_fields = result.get("northrl_train", result)
        choices = result.get("choices") or [{}]
        response = result.get("response", choices[0].get("text", ""))

        # 4. Process response
        answer = self._extract_answer(response)

        tokens = train_fields["prompt_tokens"] + train_fields["response_tokens"]
        loss_mask = [0] * len(train_fields["prompt_tokens"]) + [1] * len(train_fields["response_tokens"])
        response_log_probs = train_fields.get("response_old_logprobs")
        if response_log_probs is None:
            response_log_probs = train_fields["response_logprobs"]
        old_log_probs = [0.0] * len(train_fields["prompt_tokens"]) + response_log_probs
        tokens, loss_mask, old_log_probs, is_truncated = self._check_and_truncate(
            tokens, loss_mask, old_log_probs
        )

        # 5. Evaluate the retained result
        reward = (
            0.0
            if is_truncated or result.get("finish_reason", choices[0].get("finish_reason")) == "length"
            else self._compute_reward(answer, task)
        )

        # 6. Create one ordinary trajectory with direct token fields
        trajectory = Trajectory(
            tokens=tokens,
            loss_mask=loss_mask,
            reward=reward,
            is_val=task.get("is_val", False),
            extra_fields={
                "ground_truth": task.get("answer", ""),
                "group_id": task.get("group_id", ""),
                "run_id": task.get("run_id", 0),
                "extracted_answer": answer,
                "old_log_probs": old_log_probs,
                "is_truncated": is_truncated,
            }
        )

        # 7. Submit
        return self._put_trajectory(trajectory)

    def _format_prompt(self, question: str, context: str) -> str:
        """Format task-specific prompt."""
        if context:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        # Task-specific extraction logic
        import re
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def _compute_reward(self, answer: str, task: dict) -> float:
        """Compute reward."""
        ground_truth = task.get("answer", "")
        return 1.0 if answer.strip() == ground_truth.strip() else 0.0
```

控制器在运行前初始化推理客户端和组件引用。采样参数来自 `rollout.sampling`，
验证使用对应的采样配置。保持 `tokens`、`loss_mask`、`old_log_probs` 对齐；
prompt 位置的 logprobs 使用零占位，随后接模型返回的响应 logprobs，
不要在 worker 中把它们移位到 target-token 位置。

<a id="multi-step-processing"></a>

## 多步骤处理

多次调用不一定需要多条训练序列。只训练最后一次调用时，保留该调用实际使用的
prompt 和 response，构造普通 `Trajectory`。需要训练多条独立序列时，
在一个完整 trajectory 的内部保存普通字典：

```python
trajectory = Trajectory(
    segments=[
        {"tokens": first_tokens, "loss_mask": first_mask, "old_log_probs": first_log_probs},
        {"tokens": second_tokens, "loss_mask": second_mask, "old_log_probs": second_log_probs},
    ],
    reward=final_reward,
    is_val=task.get("is_val", False),
)
return self._put_trajectory(trajectory)
```

该片段假设每条序列已经完成处理和截断。最终 reward 只保存在完整轨迹对象上。
不要拼接无关 prompt，也不要为同一次执行的各个 turn 分别提交 trajectory。
如果截断删除了已评分答案的一部分，提交前应使该奖励失效。
详见[轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)。

<a id="error-handling"></a>

## 提交与错误处理

训练时，`execute_rollout` 暂存一次 `_put_trajectory` 提交，等 `rollout` 返回后发布。
重复提交会报错。不提交而返回 `None` 会记录 skipped；返回 `"fail"` 会记录失败。
逃出的异常会用空的 failed trajectory 替换暂存结果，并继续传播。
wrapper 不会重试任务执行。

验证直接调用 `rollout`，将 `is_val=True` 的结果交给 validator，
不会自动补充 skipped/failed 对象。始终保留任务的 `is_val` 标记，
完成序列处理和评分后再提交。

<a id="testing"></a>

## 测试与调试

先独立检查答案提取和奖励计算，再验证 worker 保留 token/mask/logprob 长度、
验证路由和单次最终提交。覆盖截断输出、缺失任务数据和模型调用失败。
通过 [rollout 调试指南](../../debug_rollout.md)和
[数据 dump](../../debug_data_dump_usage.md)检查实际模型响应。

<a id="related-documentation"></a>

## 相关文档

- [Rollout 工作进程基类](base-rollout-worker.md)
- [奖励与评估](evaluators.md)
- [Rollout 配置](../11-configuration-reference/rollout-config.md)
