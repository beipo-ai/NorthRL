[English](../../../en/developer-guide/08-features/live-metrics.md) | 简体中文 | [文档目录](../../index.md)

# E2B 运行监控日志

`live/*` 是主进程的运行监控快照，写入普通日志 `[rollout-monitor] elapsed_seconds=... metrics={...}`，**不再写 W&B**。训练 step 的字段、口径和 W&B 查看方式见 [训练 step 指标与 W&B 查看方式](../../metrics.md)。E2B 沙箱不需要 W&B 凭据。

<a id="update-intervals"></a>

## 默认行为与更新时间

指标采集默认开启，沙箱心跳约每 30 秒更新；启动、阶段切换和结束时也更新状态。主进程启动轮询时立即尝试读取 `metrics.json`，每次读取结束后等待 5 分钟再读，结束清理时额外读取一次；单次读取超时为 5 秒。监控日志仍约每 30 秒发布，使用最近收到的指标快照。后台快照发布与控制线程分离，中间快照可能合并。是否配置 W&B 后端不改变 `Tracking.log_live` 只写普通日志的规则。

`elapsed_seconds` 表示从监控器启动起经过的时间。这里的累计值覆盖本进程内的所有 attempt，包括失败和重试；活动值是当前快照。它们不归入某个 `training/train_step`，不会创建 W&B history 行。

<a id="terminology"></a>

## 基本概念

- **rollout / trajectory**：一次完整的逻辑任务，由 `rollout_id` 标识；按[轨迹契约](../04-trajectory-pool/trajectory-semantics.md)提交一个完整 `Trajectory`，包含零个或多个 `segments`。
- **attempt**：一次执行尝试。同一个 rollout 发生重试时，会产生多个 attempt。
- **活动 rollout**：已经开始首次执行尝试、尚未收到最终结果的逻辑任务。环境准备、agent 执行、评估、清理和重试等待期间都计入，最终成功、任务失败、环境错误或取消时退出统计。
- **模型轮次（model turn）**：推理客户端返回一次模型结果。
- **轨迹分段（segment / sequence）**：一个 token 序列，通过 `RemoteSegment` 返回并保存在完整 `Trajectory.segments` 中；最终是否训练取决于主轨迹状态和整组有效性。一个 rollout 可以包含多个分段，一个分段也可以包含多个模型轮次。
- **活动 attempt**：已经开始、尚未结束的执行尝试；其中只有已收到指标快照的 attempt 才参与活动长度统计。

<a id="sandbox-liveness"></a>

## E2B 沙箱与存活观测

| 指标 | 含义 |
| --- | --- |
| `live/e2b/active_sandboxes` | 已创建、尚未确认清理完成，且未超过创建时配置 TTL 的沙箱数量。这个数量本身不能证明沙箱当前可连接或仍正常运行。 |
| `live/e2b/active_rollouts` | 当前尚未结束的逻辑 rollout 数量，按 `rollout_id` 去重；从首次尝试开始计入，直到整个 rollout 结束。 |
| `live/e2b/unknown_sandboxes` | 尚未超过 TTL、但指标读取失败或未确认清理成功的沙箱数量。首次心跳前指标文件尚不存在时只计 pending。 |
| `live/e2b/heartbeat_pending_sandboxes` | 尚未观察到首次运行时心跳的沙箱数量，包括仍在准备环境的沙箱。 |
| `live/e2b/heartbeat_age_seconds_max` | 距离上次观察到心跳推进的最长时间，单位为秒。重复读取内容未变化的文件不会刷新这个时间。 |
| `live/e2b/sandbox_age_seconds_max` | 尚未确认清理完成的沙箱中，从主进程观察到创建至今的最长时间，单位为秒。 |
| `live/e2b/ttl_remaining_seconds_min` | 按配置的沙箱生命周期估算的最短剩余时间，单位为秒；启动和环境准备也消耗这段时间。 |
| `live/e2b/attempt_count` | 累计执行尝试数，包括未产生轨迹的尝试。 |
| `live/e2b/retry_count` | 累计额外重试次数，不含每个 rollout 的首次尝试。 |
| `live/e2b/completed_rollouts` | 已结束的逻辑 rollout 数，包括环境错误和取消；每个 rollout 只计一次。 |
| `live/e2b/successful_rollouts` | 最终 E2B 状态为 `success` 的 rollout 数，表示任务执行完成，并不代表 SWE resolved。 |
| `live/e2b/task_failed_rollouts` | 最终 E2B 状态为 `task_failed` 的有效任务失败样本数。 |
| `live/e2b/valid_rollouts` | 最终状态为 `success` 或 `task_failed` 的 rollout 数，包括零分段结果；不代表这些样本所在的 GRPO 组一定可训练。 |
| `live/e2b/overlength_ratio` | 最终因 `context_length_exceeded` 结束的 rollout 数 ÷ 已结束的逻辑 rollout 数；不等同于实际裁剪 token 的比例。 |
| `live/e2b/env_error_rate` | 最终以环境错误结束的 rollout 数 ÷ 已结束的逻辑 rollout 数。 |
| `live/e2b/timeout_count` | 累计超时的 attempt 数，包括沙箱创建前发生的超时。 |
| `live/e2b/timeout_sandbox` | 累计已创建、且所属 attempt 以超时结束的沙箱数，在 attempt 结束时计入；重试创建的新沙箱分别计数，清理成功与否不影响累计值，沙箱创建前的超时不计入。沿用 attempt 的超时判定，包含执行或请求超时，并非仅指沙箱 TTL 到期。 |
| `live/e2b/timeout_rate` | 超时的 attempt 数 ÷ 已结束的 attempt 数；重试单独计数。 |

例如，一个 rollout 首次尝试失败后等待重试，`active_rollouts` 始终计为 1；新尝试开始时，`attempt_count` 和 `retry_count` 各增加 1，整个 rollout 结束后 `active_rollouts` 才减少 1。

沙箱确认关闭或达到创建时配置的 TTL 后，会退出沙箱活动数量、心跳年龄和剩余 TTL 的统计。清理失败累计保留在 `live/e2b/cleanup_error_count` 中，不因活动记录移除而清零；attempt 结束后收到的关闭确认仍可清理沙箱记录。

心跳用于观测运行状态。缺少心跳可能来自环境尚未启动、指标读取失败等情况，不能仅凭这一项判定沙箱已经退出。尚未观察到任何有效心跳时，`heartbeat_age_seconds_max` 也会为 0，应结合 `heartbeat_pending_sandboxes` 查看。

<a id="trace-and-tools"></a>

## Trace 长度与工具调用

| 指标 | 含义 |
| --- | --- |
| `live/e2b/active_attempts_with_metrics` | 已收到指标快照的活动 attempt 数，即下面活动长度统计的样本数。 |
| `live/trace/active_model_turns_mean`、`live/trace/active_model_turns_max` | 上述活动 attempt 已完成模型轮次的平均值和最大值。 |
| `live/trace/active_generated_tokens_mean`、`live/trace/active_generated_tokens_max` | 上述活动 attempt 已生成响应 token 数的平均值和最大值。 |
| `live/trace/active_segments_mean`、`live/trace/active_segments_max` | 上述活动 attempt 当前已记录轨迹分段数的平均值和最大值。分段数与模型轮次数不同。 |
| `live/trace/max_context_tokens` | 已观察到的单次模型请求中，提示词 token 数加响应 token 数的最大值。 |
| `live/tool/active_calls_emitted_mean`、`live/tool/active_calls_emitted_max` | 上述活动 attempt 已输出工具调用数的平均值和最大值。 |
| `live/tool/format_unknown_turns` | 尚无法判断是否存在工具调用解析/格式错误的模型回复累计次数。未提供原文诊断的 parser，即使返回空调用或结构正常的调用，也不会据此视为无错误。 |
| `live/tool/call_counts_unknown_turns` | NorthGate 诊断中 `detected` 或 `invalid` 数量未知的回复数；与格式是否已确认错误独立，不表示旧 parser 的完整计数覆盖率。 |
| `live/tool/format_reason_<reason>_turns` | NorthGate 诊断包含该原因的回复数，同一原因每轮最多计一次；不是无效调用个数。原因列表见下文。 |
| `live/tool/format_error_turn_rate` | 检测出工具调用解析/格式错误的回复次数 ÷ 能判断结果的回复次数；分母为 `live/trace/model_turns - live/tool/format_unknown_turns`。它表示已知结果子集的错误比例，不代表未知回复也没有错误。 |
| `live/tool/result_error_rate` | 明确失败的工具结果数 ÷ 已观察到的工具结果数。 |

`live/trace/model_turns`、`live/trace/generated_tokens`、`live/trace/segments` 和 `live/tool/*` 中的计数类指标，会累计本次任务进程的所有 attempt，包括失败的尝试和重试。重复快照只贡献新增部分，不会重复累计。这些累计状态保存在当前进程内存中，重启任务后重新计数。

没有已收到快照的活动 attempt 时，活动长度的平均值和最大值归零。工具错误比率、环境错误率和 attempt 超时率在分母为零时不写入该次记录；`live/e2b/overlength_ratio` 则保留 0 初始值。

例如，10 次模型回复中有 2 次明确出错、3 次已检查无错、5 次无法判断，则 `format_unknown_turns=5`，`format_error_turn_rate=2/5=40%`。如果全部回复都无法判断，保留未知次数，不发布错误率。

<a id="tool-errors"></a>

## 工具错误分类与环境问题线索

`live/tool/error_<category>` 是已观察到的工具错误累计次数，沿用约 30 秒的普通日志快照，按 `elapsed_seconds` 定位。工具结果带 `is_error=true`、明确的非零 `exit_code`，或已关联 Bash 调用的返回以 `Exit code N`（N 非零）开头时计为失败。普通成功读取的文件正文即使包含报错文本也不计为错误。

| category | 含义 |
| --- | --- |
| `command_not_found` | shell 找不到命令，如 `cargo: command not found` |
| `dependency_missing` | 明确缺少模块、包或动态库 |
| `network` | 连接、下载或网络超时错误 |
| `permission` | 权限不足 |
| `resource` | 明确的磁盘空间、内存等资源不足 |
| `file_not_found` | 文件或路径不存在 |
| `edit` | 编辑匹配失败或无效编辑 |
| `test_or_compile` | 测试或编译失败 |
| `other` | 已确认失败，但原因无法归入上述类别 |

每个工具调用 ID 的结果只计一次，一条错误只进入一个类别，各类别之和等于 `live/tool/result_errors`。明确的命令/依赖/网络/权限/资源错误优先于一般测试或文件错误。首次分类后保持不变，避免重复历史或后续补充文本导致累计计数下降。前五类作为“环境问题线索”；它们不改变 reward、任务状态或重试行为，也不证明环境是任务失败的唯一原因。指标只包含固定类别与数值，命令、路径和错误原文保留在 trace 中。

| 指标 | 含义 |
| --- | --- |
| `live/tool/observed_rollouts` | 已结束且任意 attempt 至少观察到一个工具返回的 rollout 数 |
| `live/tool/unobserved_rollouts` | 已结束但没有工具返回观测的 rollout 数，包含尚未执行工具的任务 |
| `live/tool/observation_coverage` | `observed_rollouts` ÷ 全部已结束 rollout 数；不是单个 trace 的完整率 |
| `live/tool/env_issue_rollouts` | 已结束且任意 attempt 出现环境问题线索的 rollout 数 |
| `live/tool/env_issue_rollout_rate` | `env_issue_rollouts` ÷ `observed_rollouts` |
| `live/tool/timeout_observed_rollouts` | 已结束、有工具返回观测且任意 attempt 超时的 rollout 数 |
| `live/tool/timeout_env_issue_rollouts` | 上述 rollout 中，任意 attempt 出现环境问题线索的数量 |
| `live/tool/timeout_env_issue_rollout_rate` | `timeout_env_issue_rollouts` ÷ `timeout_observed_rollouts`，表示共现比例，不是因果判断 |

错误次数保留失败和重试 attempt 的工作量；影响比例按逻辑 rollout 去重，任意一次尝试的线索都会保留到 rollout 结束。比例的分母为零时不发布；未观测样本不作为“没有工具错误”进入环境问题比例。用于去重的历史完成 ID 沿用最近 4096 条窗口，进程重启后累计值重置。

运行时 HTTP relay 从后续模型请求的历史里观察工具返回；末次工具结果没有再随模型请求发送时，可能一直未被观察到。模型请求处理阶段会保存 trace，运行时退出时进行最终 flush；executor 结束 attempt 前的最终指标补报可以保留已采集但未被轮询到的观测，不能恢复从未被运行时观察到的工具结果。超时或断连后保留此前已经收到的累计值。

训练指标 `timeout/tool_count` 只计已观测工具返回中确认的超时。`tool/unobserved_call_count` 记录已发出但未观察到结果的调用数，不等同于 timeout；工具 timeout 计数为 0 也不代表所有已发出的调用均已完成。

这些运行期累计量只在监控日志中查看；W&B 的 `tool/*` / `timeout/*` 字段展示 trainer 消费批次中最终执行的统计，口径不同。

<a id="segment-distribution"></a>

## 一个 rollout 被拆成几段

| 指标 | 含义 |
| --- | --- |
| `live/trace/segments_per_trajectory_mean` | 已结束、状态为 `success` 或 `task_failed` 且明确返回分段数的 rollout，其最终分段数的平均值，包括零分段结果。 |
| `live/trace/segments_per_rollout_max` | 上述 rollout 最终分段数的最大值。 |
| `live/trace/segments_per_rollout_p95` | 上述 rollout 最终分段数的第 95 百分位数。 |
| `live/trace/split_trajectory_ratio` | 上述 rollout 中，最终分段数大于 1 的比例。 |
| `live/trace/rollouts_with_segments` | 已结束、最终返回至少一个分段且未以环境错误结束的 rollout 数。 |
| `live/trace/rollouts_without_segments` | 已结束、明确返回零个分段且未以环境错误结束的 rollout 数。 |

<a id="segment-counting"></a>

### 统计口径

- 以最终结果中的 `len(result.segments)` 为准，每个逻辑 `rollout_id` 只统计一次。即使最后一次心跳快照滞后于最终结果，也能记录最终分段数。
- 一个 rollout 重试多次时，仅最终 attempt 的结果参与“每个 rollout 分成几段”的统计。此前尝试产生的分段仍计入累计工作量 `live/trace/segments`。
- 状态为 `task_failed` 的有效任务失败样本也参与统计，包括明确返回零分段的结果。
- 环境错误、取消执行以及缺失最终分段数的结果不参与统计，不能把它们当作零分段。
- 明确返回零个分段的有效结果计入 `rollouts_without_segments`，同时以值 0 参与平均值、最大值、P95 和拆分比例，保持“一次完整 rollout 一个样本”的语义。
- P95 使用最近秩方法：将有效 rollout 的分段数升序排列，取第 `ceil(0.95 × 样本数)` 个值，每个 rollout 权重相同。

这些最终统计在 rollout 结束后保留，并按相同的实时上报间隔更新。 累计计数和分布保留整个任务进程的结果；用于过滤重复终态和迟到启动事件的 ID 只保留最近 4096 个已结束 rollout，避免内存随历史执行数持续增长。正常 executor 回调按顺序执行；超出该窗口的历史事件重放不保证去重。

<a id="segment-example"></a>

### 分段示例

当新请求的提示词不再以当前分段完整的 token 序列为前缀时，会开始一个新分段，例如发生上下文压缩之后。

分段分布的样本数为 `live/trace/rollouts_with_segments + live/trace/rollouts_without_segments`，不再额外发布同义总数；executor 的正常结果都携带明确分段数，因此这个和等于 `live/e2b/valid_rollouts`。缺失或非法分段数仍不参与分布计算，不能把缺失值补成零。

如果一个 rollout 最终返回 3 个 `RemoteSegment`，它对“每个 rollout 的分段数”贡献 **1 个值为 3 的样本**。

<a id="training-step-comparison"></a>

## 与训练 step 指标的对应关系

实时监控反映 executor 的任务运行和所有尝试累计量；W&B 的 Batch / Trajectory / Segment / Group 指标反映 trainer 本次实际消费的父对象与保存序列。worker 裁剪、reward 修改、整组跳过和 Datum/mask 过滤不会回写运行累计量。不能用累计量相邻差值代替消费 batch 的统计，尤其是 fully async 模式。

现有训练字段已从旧 `rollout/*`、`critic/*`、`advantage/*` 等迁移，完整口径与页面入口见 [训练 step 指标](../../metrics.md)。

<a id="collection-scope"></a>

## 数据采集方式与观测范围

<a id="stage-metrics"></a>

### 跨阶段的 `metrics` 字段

任务输出与阶段记录的公共字段见 [E2B 任务接入](../05-rollout-workers/e2b-task-contract.md)。这里区分评测指标与运行时采集的指标，三个同名字段的来源和用途不同：

| 字段 | 默认值 | 来源与用途 |
| --- | --- | --- |
| `EvaluationResult.metrics` | `{}` | 任务的 `evaluate` 返回的附加评分信息，例如 `swebench_resolved`、`tests_passed`。训练奖励由同一个结果的 `reward` 字段提供。 |
| `RemoteRolloutTrace.metrics` | `{}` | runtime 随 trace 保存的运行时指标快照，例如模型轮数、生成 token 数、工具调用数。存在 trace 时，evaluate 阶段从这里恢复 agent 阶段的观测值。 |
| `AgentStageResult.metrics` | `{}` | 没有 trace 时携带上述运行时快照，例如第一次请求就遇到上下文上限、没有记录到任何 segment。由框架写入并交给 evaluate 阶段；存在 trace 时这个字段保持为空，使用 `trace.metrics`。 |

失败退出时，最终结果优先保存当前运行时观测快照，即使最后一轮模型结果无法转换、还未更新 trace，或完全没有 trace，也保留模型轮数和工具格式诊断；评测阶段失败仍保留 agent 阶段的观测值。

这三个字段都使用字符串键和有限数值。平台在 agent 阶段的 `trace.metrics` 和 `stage.metrics` 中择一读取，再与 `EvaluationResult.metrics` 合并到最终 rollout 结果；如果键名重复，运行时指标覆盖同名评测指标。`AgentStageResult.metrics` 保存的是执行观测数据，评测是否通过由 `EvaluationResult.status` 和 `reward` 表达。

<a id="heartbeats"></a>

### 沙箱心跳

沙箱在结果文件旁写入一个体积较小、限制文件访问权限的 `metrics.json`，通过原子替换避免读取到写了一半的内容。心跳独立于模型轮次推进，并在 evaluator 执行期间继续更新。

主进程单独轮询这个文件，不依赖调试 trace，因此无需打开 `runtime.debug.enable_data_dump`。读取失败会记录观测错误，不会因此使 rollout 失败或重启。首次有效快照之前，指标文件尚不存在属于正常准备过程，只保留 pending 状态；收到首个快照之后文件丢失，以及任意阶段的连接、沙箱不存在或 JSON 内容错误，才计入 `live/e2b/poll_error_count`。若沙箱突然丢失，最后一次成功读取之后新增的指标可能无法保留。

<a id="model-turns"></a>

### 模型轮次与 token

模型轮次数统计 relay 已收到的完整 NorthGate 结果，包括随后因训练 token 数据无效而拒绝的结果。响应 token 数使用 NorthGate 提供的 usage，避免把重复传入的历史提示词计入生成量。流式请求在收到完整终止事件后统计；中断的流不计为已完成的模型轮次。

<a id="tool-format"></a>

### 工具调用与格式

工具调用统计新输出的调用，不重复统计请求历史中的调用。同一个 attempt 内，已观察到的工具结果按调用 ID 去重。如果工具结果从未随后续模型请求发回 relay，就无法被统计；输出了工具调用也不代表工具执行已经完成。

运行时优先读取 NorthGate 的顶层 `tool_call_diagnostics`，无需额外请求开关。非流式响应直接读取；OpenAI 流式响应从 `choices: []` 的 metadata chunk 读取，Anthropic 流式响应从最终 `message_delta` 读取。`checked=true` 表示原文解析器提供格式检查，`malformed=true` 表示已确认存在错误，不表示完成了全部参数 JSON Schema 校验。缺少该诊断时，读取 `northrl_parse_diagnostics.tool_format_checked/tool_format_error`；若没有 `northrl_parse_diagnostics` 字段，则兼容改名前的 `nexrl_parse_diagnostics`。没有检查依据且本地结构检查未发现错误时记为未知，不能把空工具列表当作已检查无错。

运行时仍对返回的调用执行统一结构检查，包括 `function`、非空工具名与 ID，以及参数能否解析为 JSON 对象。非法参数 JSON 计入 `live/tool/arguments_json_errors`；合法 JSON 但类型不是对象计入 `live/tool/arguments_type_errors`。即使原文诊断未知，只要通用检查发现错误，也能确定本回复有错，不计入未知次数。

- NorthGate 的 `detected/invalid` 覆盖原文中的调用，包括被修复或丢弃的调用，分别进入 `live/tool/calls_detected` 和 `live/tool/calls_invalid`。使用这些计数时不再累加返回消息中的结构错误，避免重复统计。
- `detected/invalid=null` 表示数量未知，不从最终调用列表补估计值；累计计数仅代表已知部分，同时增加 `live/tool/call_counts_unknown_turns`。即使数量未知，`malformed=true` 仍然计为格式错误轮次。
- `live/tool/format_error_turns` 统计已明确存在解析/格式问题的回复，每次回复最多加 1。诊断未说明坏调用数量时，不虚构坏调用个数。
- 只有服务端回包提供的解析诊断才能统计 parser 丢弃的调用；本地结构检查不等于完整 JSON Schema 校验。工具执行时返回的 `is_error` 单独计入 `result_errors`。

`reasons` 按固定原因分别统计：`incomplete_tool_call`、`invalid_arguments_json`、`missing_tool_name`、`invalid_tool_call_structure`、`arguments_not_object`、`duplicate_parameter`、`undeclared_tool`。同一回复可以命中多个原因，因此这些次数不能相加当作错误回复或无效调用总数；未知原因不创建动态指标。`arguments_json_errors/arguments_type_errors` 仍只统计返回调用的本地参数检查，与 NorthGate 的原文原因统计分开。

这些诊断只用于观测，不改变工具调用接受规则、reward 或 advantage。训练指标 `tool/format_unknown_turn_count`、`tool/call_counts_unknown_turn_count`、`tool/format_reason_<reason>_turn_count` 是本批最终执行的次数总和；相应 `live/tool/*_turns` 累计本进程所有 attempt 的已观察工作量。

<a id="request-timing"></a>

## 权重同步与模型请求耗时

所有耗时均以秒为单位，使用单调时钟。

| 训练指标 | 含义 |
| --- | --- |
| `timing_weights/weight_export_seconds` | 导出供 sampler 使用的权重；与 `timing_train/batch_training_seconds` 分开计时。 |
| `timing_weights/checkpoint_seconds` | 本步成功 checkpoint 保存的客户端墙钟耗时，不计入训练总时间；失败时不报。 |
| `timing_weights/weight_sync_seconds` | 创建并切换新的 sampling client，包含调用侧等待；不含权重导出、validation 或日志上报。 |
| `training/weight_export_success_flag`、`training/weight_sync_success_flag` | 对应操作的成功标志，成功为 1，失败为 0。失败耗时改写到 `timing_weights/weight_export_failed_seconds`、`timing_weights/weight_sync_failed_seconds`，不混入成功曲线。 |

同步指标在发布结束后、validation 或下一批放行前，并入产生该权重的同一 `training/train_step` 暂存行。跳过训练、no-sync 或本轮不需要发布时，不产生相应计时，不补零。计时观测失败不会改变同步及批次放行逻辑。

| 实时指标 | 含义 |
| --- | --- |
| `live/inference/request_count`、`completed_count` | 累计开始、已经结束的模型请求数。 |
| `live/inference/success_count`、`error_count`、`timeout_count`、`cancelled_count` | 已结束请求的互斥分类；成功指 NorthGate 响应完整且训练 token 数据成功记录，流式请求还必须收到终止事件。 |
| `live/inference/request_seconds_sum`、`request_seconds_mean`、`request_seconds_max` | 所有已结束请求的累计耗时、按请求等权的平均值、最大值，包含失败、超时和取消。并发请求的累计耗时不能当作 rollout 墙钟时间。 |
| `live/inference/request_seconds_p95` | 合并所有请求的累计直方图后估算的 P95，不平均各 rollout 的 P95。桶上界为 1、5、10、30、60、120、300、600、1800 秒及溢出桶；返回所在桶上界与已观察最大值的较小值，是近似值。 |
| `live/inference/inflight`、`oldest_inflight_seconds` | 活动 attempt 最近快照中的正在执行请求总数、最长已等待时间。按当前值更新，不累计历史最大值；无活动请求时归零。快照滞后于真实状态，应结合心跳年龄判断。 |

这些指标随主进程读取的沙箱快照更新（定期读取间隔为 5 分钟，结束前额外读取），跨本进程所有 attempt 累计，包括环境失败与重试；没有完成请求时不发布 mean/P95。重复或旧心跳不重复累计，也不覆盖更新的活动状态。

一次观测请求对应 sandbox relay 处理的一次逻辑模型调用，耗时覆盖网络、NorthGate 的模板/tokenization 与输出解析、后端处理，以及本地训练 token 记录。流式请求一直计时到收到完整终止事件、确认训练数据；中断、无效 metadata 和缺失终止事件计为失败。relay、NorthGate 或后端内部重试属于同一次观测请求；agent 发起新的 HTTP 重试算新请求。这组指标统计完整调用，没有独立的首 token 或 GPU 计算计时，因此不能仅凭 SSE 事件把它们解释为真实 TTFT 或纯 GPU 推理时间。

运行累计请求指标进入普通日志的 `live/inference/*`；最终执行的逐请求观测还会随父对象进入 batch，计数和长度汇总为 `model_request/*`，耗时汇总为 `timing_rollout/model_request_*_seconds`（`mean`、`min`、`max`）。有效计时分母保留在内部，不额外上报计时计数；现有请求计数保持不变。原始桶和活动状态保留在运行时快照中，不作为训练 reward/score 平均。

每个完成请求另写一行私有 `requests.jsonl`，只含 rollout/attempt/request 身份、API、状态、耗时、token 数和异常类型，不包含 prompt 或响应文本。开启既有 debug rollout trace 保存时，主进程在沙箱清理前回收为 `rollout_traces/e2b_rollout_<rollout_id>.requests.<attempt_id>.jsonl`；各次重试独立保留。未开启 trace 保存时仍有实时统计，但主进程不回收明细。回收失败不影响 rollout 结果或沙箱清理；沙箱突然丢失时无法保证最后的观测或明细可恢复。

正常 `[PROFILE]` 开始/完成日志改为 DEBUG；失败仍为 ERROR，超过 60 秒的模型请求输出一条带请求身份的 WARNING。同步保留开始和耗时摘要，后端内部重试继续保留原有告警。设置 `LOG_LEVEL=DEBUG` 可恢复逐阶段诊断，该设置会传到 E2B 沙箱；关闭详细日志不会关闭计时采集。
