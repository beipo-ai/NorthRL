[English](../../../en/developer-guide/11-configuration-reference/e2b-execution-budgets.md) | 简体中文 | [文档目录](../../index.md)

# E2B 执行时限

E2B 为环境准备、Agent 执行、Agent 收尾、评测准备、评测和数据恢复分别分配时间。
某个阶段较慢，不会消耗为后续阶段预留的时间。环境准备、Agent 执行、收尾和评测
可以配置；数据恢复和单次 I/O 使用框架固定的上限。

以下字段位于 `rollout.execution`：

| 字段 | 默认值 | 用途 |
| --- | ---: | --- |
| `preparation_timeout_seconds` | 600 | 准备 sandbox、输入和运行环境。 |
| `agent_timeout_seconds` | 3600 | 执行 Agent 及其工具调用。 |
| `agent_finalize_timeout_seconds` | 600 | Agent 停止执行后，完成输出处理并保存轨迹。 |
| `evaluation_timeout_seconds` | 600 | 评测材料就绪后，运行评测并生成结果。 |

可配置的阶段时限必须是正整数。布尔值、浮点数、字符串、显式 `null`、零和负数
都会在 Worker 初始化时被拒绝，execution profile 中的值也遵循相同规则。

```yaml
rollout:
  execution:
    preparation_timeout_seconds: 600
    agent_timeout_seconds: 3600
    agent_finalize_timeout_seconds: 600
    evaluation_timeout_seconds: 600
```

评测材料的上传和解压共用独立、固定的 **600 秒**准备时限，包含重试和等待就绪标记。
材料就绪后才开始计算配置的评测时限。准备阶段超时会将本次执行记为环境故障，
不会启动评测，也不会重新运行 Agent。性能记录分别统计
`driver.evaluation-preparation`、`driver.evaluation-upload`、
`driver.evaluation-extract` 和 `driver.evaluator-command`；成功耗时以 DEBUG 记录，
失败以 ERROR 记录。这个准备时限是内部常量，没有对应的 YAML 配置项。

框架预留 **900 秒（15 分钟）**来恢复已有的结果和轨迹数据。普通传输或恢复 I/O 的
单次上限为 **60 秒**。最终 `result.json.gz` 可能远大于控制文件，因此下载它时，
读取和 SDK 请求的单次上限为 **120 秒**。两者同时受当前阶段剩余时间限制。
这些上限均为内部常量，不能通过 YAML 或 execution profile 修改。

每个阶段的时限包含该阶段内的重试。重试传输不会重置恢复截止时间，也不会重新获得
一份完整的阶段时间。Agent 启动前的 sandbox 重试共用准备阶段的同一个截止时间。
恢复阶段只取回本次执行已经产生的数据，不重新运行 Agent。`env_error_retries` 只允许在
**Agent 命令下发之前**因故障创建新的 sandbox。命令下发后，无法恢复的样本以环境故障
结束，不产生任务评分，executor 不会自动重跑 Agent。Worker 仍提交终态的完整轨迹记录，
使对应 rollout 组能够结束。

<a id="recovery-behavior"></a>

## 恢复行为

- 宿主下载 Agent 完成记录超时会输出警告。Agent 命令完成后，评测端仍会在 sandbox
  内验证该记录；缺失、格式错误或属于其他执行的记录不会被接受。
- Agent 命令连接断开后，宿主会在原 sandbox 中继续观察，直到原有截止时间。只有确认
  当前 attempt 的可信完成状态后才启动评测。观察超时不等同于已评分的任务失败。
- 评测材料上传在原 sandbox 内重试。确认解压命令失败退出后可重新解压；如果只是传输
  断开，则观察就绪标记，避免同时启动多个解压进程。
- 评测器只启动一次。命令流断开后，executor 等待原评测器的结果。最终结果下载可以在
  剩余恢复时间内重试，不重新启动评测器或 Agent。
- 平台 supervisor 将评测 runtime 放入独立进程组。评测截止时间到达或 runtime 退出时，
  supervisor 终止该进程组，再最多等待 5 秒回收 runtime 进程。这也会清理任务 hook
  返回或取消后遗留的测试进程。supervisor 继承输出流，不等待后代进程持有的管道关闭。
  脱离该进程组的子进程由后续 sandbox 销毁清理。未完成的评测记为环境故障，不作为零分
  测试结果。任务入口无需为此修改。
- 故障恢复先确定可信的结果或失败状态，再下载轨迹。轨迹或诊断信息读取失败不会覆盖
  已确定的结果。远程读取超时后可使用已有本地轨迹；未产生模型交互的 Agent 不需要为
  一个不存在的轨迹文件等待 15 分钟。
- sandbox 销毁先尝试 10 秒。失败后，宿主保留 sandbox ID，安排后台控制面检查，并最多
  再尝试销毁两次，总时限为 60 秒。销毁成功或明确返回 sandbox 不存在，才确认已关闭；
  网络错误仍视为状态未知。后续确认关闭会从实时指标中移除未知 sandbox，不改变 rollout
  结果。如果重试耗尽，或 Worker 在确认前关闭，会记录 ID 供排查；后台重试要求 Worker
  事件循环继续运行。
- 取得结果后，额外日志、指标回收和 sandbox 清理仍按尽力完成处理。缺少必需的结果或
  训练数据时会报告故障，不会用虚构的分数或训练片段替代。

Agent 收尾阶段默认有独立的 600 秒，因此完成输出和保存轨迹不必挤在 Agent 剩余执行
时间内。即使 Agent 已耗尽执行时间，恢复阶段仍可继续下载轨迹。

sandbox runtime 对运行中的 hook 执行 Agent 时限控制，随后等待进程清理完成，再提交
完成状态。执行超时会记录 `agent_exit_code=124` 和 `e2b_timed_out=1`，评测仍可以给已修改
的 workspace 打分。收尾失败或超时则不允许进入评测。runtime 指标包含
`agent_execution_seconds` 和 `agent_finalize_seconds`。宿主还对整条命令设置上限，
兜底处理不响应取消的 hook；最终由 sandbox 销毁保证清理。

sandbox 存活时间预留四个可配置阶段时限之和，再加上固定的 600 秒评测准备、900 秒
数据恢复和 30 秒清理时间。按默认值计算为 **6930 秒**。单次 I/O 上限不再额外计入
sandbox 存活时间。

<a id="existing-configurations-and-execution-profiles"></a>

## 已有配置与 execution profile

使用 `rollout.execution.agent_timeout_seconds` 设置 Agent 执行时限。MiMo hook 不读取
任务数据中的 `agent_timeout_sec`，也不另设 1800 秒的默认 Agent 时限。旧数据可以保留
这个未使用字段；新 recipe 应从 `input_fields` 中省略它。runtime 在配置时间耗尽时取消
Agent，再使用独立的收尾时间完成清理。各 verifier 自己的评测超时设置保持原义。

旧字段 `rollout.execution.timeout_seconds` 仍可用，但含义是 **Agent 执行时限**，不再用
一个共同截止时间限制全部阶段。例如 `timeout_seconds: 3600` 给 Agent 一小时，环境准备、
收尾和评测继续使用各自默认时限，数据恢复仍有固定的 15 分钟。

两个字段同时存在时，`agent_timeout_seconds` 优先。直接调用 executor 时仍可传入
`timeout_seconds` 参数：配置没有显式 `agent_timeout_seconds` 时，它提供 Agent 时限。
没有该调用参数时使用旧配置字段；两者都没有则默认为 3600 秒。

execution profile 继承公共 `rollout.execution` 设置，可以分别覆盖环境准备、Agent
执行、Agent 收尾和评测这四个时限。任何 rollout 启动前都会验证所有合并后的 profile。
选中的 profile 决定这些阶段时限；恢复和 I/O 上限对所有 profile 都固定不变。

接入流程见 [Agent 任务接入](../05-rollout-workers/e2b-task-integration.md)，执行和结果恢复
细节见 [E2B 运行机制](../05-rollout-workers/e2b-runtime.md)。
