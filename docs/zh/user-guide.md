[English](../en/user-guide.md) | 简体中文 | [文档目录](index.md)

<a id="northrl-user-guide"></a>

# NorthRL 用户指南

NorthRL 连接任务执行、奖励计算和训练。从已有配方开始，再定制任务所需的接口。
运行时组件负责任务接入、轨迹收集、权重同步和训练服务调用。

<a id="quick-start"></a>

## 快速开始

安装、凭据和启动命令见[快速开始](developer-guide/01-getting-started/quick-start.md)，
服务凭据和数据路径见[本地配置](developer-guide/01-getting-started/configuration-setup.md)。
在准备好的本地 Python 环境中运行驱动。

按任务选择起点：

| 任务 | 入口 | 指南 |
| --- | --- | --- |
| 单次模型响应与命名奖励 | `rollout.type: single_turn` | [数学配方](recipe/math/README.md) |
| 自定义提示、多次调用或自定义评分 | `BaseRolloutWorker` 子类 | [自定义 worker](developer-guide/05-rollout-workers/custom-workers.md) |
| 在沙箱中执行 agent 或工具 | `rollout.type: e2b` | [E2B 任务接入](developer-guide/05-rollout-workers/e2b-task-integration.md) |
| 监督微调 | `algorithm.type: sft`，省略 `rollout` | [SFT 配方](recipe/sft_quickstart/README.md) |

内置 `simple` worker 也支持 completion，并对 `<answer>` 内容进行精确匹配评分。
它使用固定评分逻辑；选择已注册奖励时使用 `single_turn`。

<a id="recipe-preparation"></a>

## 准备配方

将任务配置与代码放在一起：

```text
recipe/my_task/
├── train.yaml
├── worker.py       # Only for a custom local worker
└── rollout.py      # Only for a sandbox task
```

完整训练配方包含 `experiment`、`models`、`data`、`validate`、`algorithm`、
`rollout`、`training`、`backend` 和 `runtime` 配置块，SFT 省略 `rollout`。
复制仍在维护的配方，调整模型、数据路径、训练预算和运行环境。
下面的片段只展示部分配置，不能单独作为完整训练配置运行。

单轮数学任务使用：

```yaml
rollout:
  type: single_turn
  reward_fn: math_answer_only
```

数据加载器要求 `prompt` 为聊天消息列表，例如：

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42"}
```

`label` 为这个奖励函数提供参考答案。样本中可选的 `reward_fn` 优先于配置默认值。
其他 scorer 可能需要不同字段，详见[数据格式](developer-guide/03-data-loader/data-loader.md)
和[奖励选择](developer-guide/11-configuration-reference/rollout-config.md#single-turn-reward-selection)。

自定义 worker 路径相对于配方 YAML 解析。E2B 入口 target 和配置后的
`task_files_root` 当前要求使用宿主机绝对路径，准备时参考
[E2B 配置指南](developer-guide/05-rollout-workers/e2b-task-integration.md)。凭据放在运行环境中，
隐藏评测材料不要作为题目输入发送给沙箱中的 agent。

<a id="implementing-rollout-workers"></a>

## 实现 Rollout 工作进程

自定义 worker 继承 `BaseRolloutWorker`，实现 `async rollout(task)`，
使用已初始化的推理客户端，并通过 `_put_trajectory` 提交一个完整 `Trajectory`。
配置方式：

```yaml
rollout:
  type: custom
  custom_module_path: worker.py
  custom_class_name: MyWorker
```

控制器相对于配方导入 `worker.py`，使用完整配置构造 `MyWorker`。
任务参数可放在 `rollout.params`。
[自定义 worker 指南](developer-guide/05-rollout-workers/custom-workers.md)
提供完整实现并解释模块导入方式。

普通 trajectory 直接保存 `tokens` 和 `loss_mask`。需要保留多条独立训练序列时，
将序列字典放入一个完整 trajectory 的 `segments`。多轮交互本身不要求分段。
最终奖励保存在完整轨迹对象上，同时保留任务的 `is_val` 和按 token 对齐的
`old_log_probs`，其中 prompt 位置使用零占位。提交前处理截断；如果已评分答案被截断，
应使该奖励失效。详见[轨迹术语说明](developer-guide/04-trajectory-pool/trajectory-semantics.md)。

训练时，wrapper 在 `rollout` 返回后发布已提交结果。不提交会记录 skipped；
返回 `"fail"` 会记录失败；逃出的异常会发布空的 failed 结果并继续传播。
验证将 `is_val=True` 的对象直接交给 validator，不会自动创建上述结果。
详见 [worker 生命周期](developer-guide/05-rollout-workers/base-rollout-worker.md)。

<a id="implementing-evaluators"></a>

## 实现奖励与评估

内置单轮路径调用 `evaluate_reward(task, result)`。奖励名称在
`northrl.rewards.REWARD_FNS` 中选择普通 Python callable。返回值可以是有限标量，
或包含 `reward` 及可选 `correct`、`metrics`、`reason` 的字典。
配方目录里的评分代码可以在自定义 worker 内调用；
`rollout.reward_fn` 本身不会加载任意 Python 文件。

E2B 任务脚本实现异步 `run_agent`、`evaluate` 和可选的 `prepare`。
runtime 记录模型交互，并在 agent 阶段结束后上传隐藏评测材料。
评估函数返回 `EvaluationResult`；任务未通过时可以返回
`status="task_failed"` 和零奖励。基础设施故障应抛出 `RolloutEnvironmentError`，
不要伪造评分。详见[奖励与评估](developer-guide/05-rollout-workers/evaluators.md)。

<a id="trainer-architecture"></a>

## 训练器架构

通过 `algorithm.type: grpo`、`opd` 或 `sft` 选择对应 trainer。
调整任务评分通常不需要修改 trainer。trainer 准备完整 trajectories，
计算算法训练信号，再把内部序列转换为服务需要的 Datums。

从[算法组件](developer-guide/06-trainers/algorithm-components.md)进入 estimator、
loss 和后处理的选择说明及各自的详细指南。仅在改变优化行为时扩展这些组件或
trainer，并在转换前保留完整 trajectories。训练器接口与执行流程见
[Remote API 训练器](developer-guide/06-trainers/remote-api-trainers.md)。

<a id="configuration-guide"></a>

## 配置指南

允许的字段见[配置参考](developer-guide/11-configuration-reference/rollout-config.md)。
生成参数放在 `rollout.sampling`，任务参数放在所选 worker 的配置中，
优化器参数放在 `algorithm`。运行 `python -m northrl.main` 前，
请在本地驱动环境中准备好依赖和环境变量，详见[本地配置](developer-guide/01-getting-started/configuration-setup.md)。

<a id="best-practices"></a>

## 最佳实践

- 将任务提示、奖励规则和辅助代码放在对应配方中。
- 校验必要输入字段，区分错误答案和服务故障。
- 保持奖励与标量指标有限，并说明量表及含义。
- 保留实际模型调用返回的 token ID、mask 和采样 logprobs。
- 只提交一个完整结果，保留执行身份和验证路由。
- 在分布式训练前先测试任务逻辑。

<a id="debugging-tips"></a>

## 调试建议

先用 [rollout 调试](debug_rollout.md)检查生成和奖励，无需启动完整训练。
通过[数据 dump](debug_data_dump_usage.md)检查 tokens、masks、分数与 trajectory 分组。
覆盖正确、错误、空和截断响应，再在训练中检查奖励分布与非有限数值。
启动和服务问题参考[启动要求](developer-guide/01-getting-started/quick-start.md)。
