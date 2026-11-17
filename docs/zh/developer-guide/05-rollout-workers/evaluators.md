[English](../../../en/developer-guide/05-rollout-workers/evaluators.md) | 简体中文 | [文档目录](../../index.md)

<a id="rewards-and-evaluation"></a>

# 奖励与评估

评分逻辑属于任务，接入方式取决于 rollout 路径：

| 路径 | 评分接口 | 返回值 |
| --- | --- | --- |
| `single_turn` | `northrl.rewards.REWARD_FNS` 中的命名函数 | 有限标量或包含 `reward` 的字典 |
| 自定义 worker | `rollout(task)` 内调用的任务代码 | 一个完整 `Trajectory` 上的最终奖励 |
| `e2b` | 异步 `evaluate(input_data, workspace, agent_result, evaluation_assets)` | `EvaluationResult` |

<a id="single-turn-rewards"></a>

## 单轮奖励

把奖励选择合并到完整配方中：

```yaml
rollout:
  type: single_turn
  reward_fn: math_answer_only
```

样本中非 null 的 `reward_fn` 优先于 `rollout.reward_fn`。未知名称（包括显式空字符串）
会报错。函数接收 `(task, result)`；`result["response"]` 是生成答案，
`result["finish_reason"]` 表示补全状态。详见
[支持的奖励与输入字段](../11-configuration-reference/rollout-config.md#single-turn-reward-selection)。

`evaluate_reward` 将标量规范化为 `{"reward": value}`。字典必须包含 `reward`，
可包含布尔 `correct`、字符串 `reason`，以及从字符串映射到有限数值的 `metrics`。
奖励必须有限，接口不限制在 `[0, 1]` 范围内。选择适合任务的量表，
并让可比较样本的奖励含义保持一致。

同步评分在线程中运行；异步函数和返回的 awaitable 在调用者的事件循环中执行。
取消等待不会中断已经开始的同步 scorer，judge 故障会继续传播。
`finish_reason == "length"` 会直接返回零而不调用评分函数；
如果评估后又截断训练序列，`SingleTurnRolloutWorker` 也会清零奖励。

<a id="custom-worker-scoring"></a>

## 自定义工作进程评分

自定义 worker 可以直接调用任务函数，也可以显式传入函数映射，复用 `evaluate_reward`：

```python
from northrl.rewards import evaluate_reward


def exact_match(task, result):
    correct = result["response"].strip() == task["answer"].strip()
    return {"reward": float(correct), "correct": correct}


async def score_response(task, result):
    return await evaluate_reward(
        task,
        result,
        functions={"exact_match": exact_match},
        default_reward_fn="exact_match",
    )
```

这个映射是 Python API 参数，不是 YAML 插件配置。内置 single-turn worker 使用
`REWARD_FNS`；把任意文件路径填入 `rollout.reward_fn` 不会加载该文件。
要在不修改内置映射的情况下使用配方目录里的评分代码，可选择
[自定义 worker](custom-workers.md)，在其中调用评分函数。
完整 trajectory 上只保留一个最终 reward。

<a id="sandbox-evaluation"></a>

## 沙箱评估

任务入口实现 `run_agent`、`evaluate` 和可选的 `prepare`。平台完成 agent 阶段、
保存轨迹、上传隐藏材料，再在同一 sandbox 的独立进程中调用 `evaluate`。
模型轨迹由 runtime 记录，`AgentResult` 携带可选答案和产物引用。
下面的评分函数与 `recipe/e2b_minimal/rollout.py` 的最小任务一致：

```python
from pathlib import Path
from northrl.remote_rollout.contracts import (
    AgentResult,
    EvaluationResult,
    RolloutEnvironmentError,
    RolloutInput,
)


async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult:
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
```

`EvaluationResult.reward` 必须有限，可选 `metrics` 也必须是有限数值。
`status="task_failed"` 表示已完成评估但任务未通过，要求 reward 为零；
`status="success"` 接受任何有限奖励，包括零。材料缺失或评测故障应抛出
`RolloutEnvironmentError`，不要伪造已评分结果。入口配置和隐藏材料处理详见
[E2B 任务接入](e2b-task-contract.md#scoring-and-failures)。

<a id="testing-evaluation"></a>

## 测试评估逻辑

覆盖正确与错误答案、缺失或非法字段、空输出、多个合法参考答案和非有限分数。
详细诊断与标量指标分开保存。评分依赖外部服务时，应区分服务故障与有效错误答案，
并明确超时时限。

<a id="related-documentation"></a>

## 相关文档

- [自定义工作进程](custom-workers.md)
- [轨迹术语说明](../04-trajectory-pool/trajectory-semantics.md)
- [Rollout 调试](../../debug_rollout.md)
