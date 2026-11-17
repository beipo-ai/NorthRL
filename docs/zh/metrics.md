[English](../en/metrics.md) | 简体中文 | [文档目录](index.md)

# 训练 step 指标与 W&B 查看方式

现有 NorthRL 训练指标以 **`training/train_step`（trainer 已消费并完成的 batch 轮次）** 为唯一横轴。一次 step 可以有零次或多次 optimizer 更新。运行中的 `live/*` 只写 `[rollout-monitor]` 普通日志，不增加 W&B history 行。

字段、单位、定义和统计阶段见 [metric_definitions.py](../../northrl/utils/metric_definitions.py)；[metric_catalog.py](../../northrl/utils/metric_catalog.py) 负责字段规范化与 W&B 轴/summary 设置。字典登记不代表每种 worker 或服务都支持全部字段，只有有效观测才会记录。

## 在普通 Run / Workspace 中查找

训练正常写入 W&B 后，直接查看该 run 或项目 workspace，无需额外命令。各类指标的规范前缀如下：

| 字段前缀 | 内容 |
| --- | --- |
| `batch/` | 输入与实际可训练的 group、trajectory、segment、token 总量与 response token 总量，沙箱重试 |
| `trajectory/` | 保存 response 长度、段数、工具/请求数 |
| `segment/` | 保存段 response 长度、可靠归属的工具调用和 timeout、实际裁剪 |
| `group/` | 各组数量总和的 min/max、有效组与单成员组数 |
| `algorithm/` | Reward、Advantage、Reference KL、熵、PPO/IS、采样 logprob/PPL、MIS 和 OPD |
| `training/` | 实际更新和 forward/backward 次数、跳过与权重发布状态 |
| `model_request/` | 单次模型请求输入/输出长度和请求数 |
| `tool/` | 工具调用、已观测返回、错误和未观测返回的调用数 |
| `timeout/` | 工具、模型请求、整条 rollout 三层 timeout 与受影响对象 |
| `verifier/` | reward 判题结果、失败和 E2B 超时/重试 |
| `reward_fn/` | 按有效 reward function 拆分的判题计数 |
| `outcome/` | 任务/环境结果、上下文超限、裁剪及附加任务分数 |
| `weaver/` | 服务 loss、grad norm、token/样本计量和适用的更新状态 |
| `timing/` | 批次完成间隔和 trainer 等待 |
| `timing_rollout/` | trajectory/group/verifier/model request 耗时分布，包含按 reward function 拆分的统计 |
| `timing_train/` | 训练总耗时、logprob/诊断前向、FB 和 optimizer 的 wall/compute 耗时 |
| `timing_weights/` | 权重导出、checkpoint、同步及失败耗时 |
| `timing_val/` | 验证耗时分布，包含按 reward function 拆分的统计 |
| `val/` | 本次验证的数量、分数和 pass@k |

本文表格使用代码内部的规范名称。实际写入 W&B 时会增加固定编号，例如 `algorithm/trajectory/reward_mean` 对应 `01_algorithm/01_trajectory/reward_mean`，`timing_rollout/` 对应 `06_timing_rollout/`；其他日志后端保留规范名称。

W&B 自动 workspace 按第一个 `/` 前缀组织图表，因此带编号的算法指标归入 `01_algorithm` 分区；参见 [W&B 分区组织方式](https://docs.wandb.ai/models/track/workspaces#section-organization)。数据字段均设置为不隐藏，只有作为横轴的 `training/train_step` 不单独生成自动图。少量关键指标保留 latest summary，其他指标仍写 history，但不生成 summary 汇总列。

### 启动时设置 workspace 顺序

`wandb.init` 完成后，NorthRL 会设置当前实验所属 entity/project 下、**`WANDB_KEY` 所属用户的项目页和 Run 页个人 workspace**。这两个页面分别保存布局。Run 页布局由该用户在项目内的多个实验共用，因此查看旧实验时也可能看到新顺序；其他用户的个人 workspace、另存的命名视图和其他项目不受影响。

初始化会预留指标目录中的 25 个编号分区（`01_algorithm` 到 `25_group`），让后出现的指标进入已有位置；按名称重排可见分区、取消分区置顶、启用第一层前缀分组及分区内部的图表字母排序，并隐藏空分区。同时预留 `Charts`、`System`，W&B 的隐藏图表区始终保留在最后。已有图表定义、折叠状态、筛选条件、run 选择、隐藏图表和自定义网格坐标会保留，已有手动 workspace 的自动生成图表开关也会保留；mean/min/max 仍是独立图表。

每次启动 W&B 日志记录时都会自动设置排序，无需额外配置。

无需额外依赖或命令。Offline/disabled 模式跳过接口；服务端接口不兼容、没有写入权限或请求失败时只记录警告，训练继续。重复启动只在顺序需要更新时保存。这里使用 W&B 内部视图 API，不修改训练参数或历史数据。保存前会重新读取，保存后会验证，但接口没有原子版本检查：同时编辑或旧浏览器标签页之后自动保存，仍可能覆盖排序。它只在启动时初始化，不会持续强制重排页面。

### Timing 分区

Timing 使用五个分区。固定训练计时共 27 项：2 项总览、8 项训练、5 项权重/保存和 12 项 rollout 耗时分布；另按适用范围生成 reward function 和验证耗时曲线。每个字段只有一个 `/`；分布中的 `*` 表示 `mean`、`min` 或 `max`。

| W&B 字段 | 统计口径 |
| --- | --- |
| `timing/batch_completion_interval_seconds` | 相邻两次 `train()` 返回之间的时间：第 `n` 步为 `finish(n) - finish(n-1)`，使用 trainer 进程的 monotonic 时钟。 |
| `timing/trainer_wait_for_batch_seconds` | 为完整 batch 等待的已观测时间，包含轮询和调度延迟；立即领取为 0。排除初始化、初始验证、训练、保存和同步。 |
| `timing_train/batch_training_seconds` | 标准 trainer 从进入训练到开始保存前的墙钟时间，包含数据处理、重算和 FB/optimizer，排除权重导出、checkpoint 和权重同步。 |
| `timing_train/recompute_logprobs_seconds` | 训练端重算 logprob 的客户端墙钟耗时，含请求等待。 |
| `timing_train/ref_logprobs_seconds` | reference 模型计算 logprob 的客户端墙钟耗时。 |
| `timing_train/mismatch_forward_seconds` | 独立 mismatch 诊断前向的客户端墙钟耗时，不含随后统计。 |
| `timing_train/forward_backward_wall_seconds` | 本步全部 FB 阶段的客户端墙钟耗时，含 Datum 准备、提交、排队和结果轮询；流水线请求按整体阶段跨度计。 |
| `timing_train/forward_backward_compute_seconds` | 本步完整有效的 FB 服务 `compute_ms` 观测合计除以 1000。 |
| `timing_train/optimizer_step_wall_seconds` | optimizer 提交到完成的客户端墙钟耗时，多次更新相加；在全部 FB 成功后开始。 |
| `timing_train/optimizer_step_compute_seconds` | 本步完整有效的 optimizer 服务 `compute_ms` 观测合计除以 1000。 |
| `timing_weights/weight_export_seconds` | 采样权重成功导出的客户端墙钟耗时。 |
| `timing_weights/checkpoint_seconds` | 本步成功完成的 checkpoint 保存客户端墙钟耗时；保存失败时不报此字段。 |
| `timing_weights/weight_sync_seconds` | `train()` 返回后的采样权重同步客户端墙钟耗时。 |
| `timing_weights/weight_export_failed_seconds` | 采样权重导出失败前的耗时。 |
| `timing_weights/weight_sync_failed_seconds` | 采样权重同步失败前的耗时。 |
| `timing_rollout/trajectory_*_seconds` | 单次完整 rollout 的执行耗时分布。 |
| `timing_rollout/group_*_seconds` | 完整采样组执行跨度分布。 |
| `timing_rollout/verifier_*_seconds` | reward 判题耗时分布，不含答案生成。 |
| `timing_rollout/model_request_*_seconds` | 被观测的答案生成请求耗时分布。 |

completion interval 在 sync、batch-async、fully-async、no-sync 和 SFT 中定义相同，覆盖上一轮同步、验证、日志和剩余数据等待，以及本轮完整 `train()` 调用（含权重导出和 checkpoint）；本轮返回后的工作计入下一间隔。启动或恢复后的首次返回只建立基准，不上报间隔。filtered/skipped 批次仍算完成步，比较实际更新吞吐时需结合 optimizer 更新次数。

等待取批期间的验证暂停不计入 trainer 等待，暂停前已观测的等待仍保留。自定义 trainer 未单独提供训练阶段计时时，回退为完整调用耗时，因为其没有标准保存边界。

训练总时间不等于图中阶段之和：它还包含不单独生成曲线的数据处理等工作，部分阶段也有包含或重叠关系。服务 compute 和客户端 wall 是同一操作的不同观测，不能相加。`compute_ms` 仅表示服务报告的计算耗时，尚不能解释为纯 GPU 时间。compute 观测缺失、无效或不完整时省略本步合计，不补 0；成功跳过训练的 batch 可以记录零客户端工作，但没有 compute 观测。

按 reward function 拆分的耗时仍归所属执行阶段，使用 `timing_rollout/<reward_fn>_<object>_<reducer>_seconds`；验证使用 `timing_val/<object>_<reducer>_seconds` 和 `timing_val/<reward_fn>_<object>_<reducer>_seconds`。有效观测数保留为内部分母元数据和普通诊断日志，不单独上报为 W&B timing 字段。

Batch rollout span 和单独的 MIS/mismatch 统计、prepare、other 计时已退役，相关算法仍照常运行。原始 producer 和旧 timing 名称只有语义相同时才规范化到新分区。旧 `timing/batch_training_seconds` 和 `timing/training_time` 包含保存，规范化时直接丢弃，不冒充新的纯训练总时间。旧 `timing/mismatch_forward` 还包含随后统计，也直接丢弃，不映射为新的纯前向计时。采集变更需要重启进程；不回填历史数据。详见 [rollout 与 trainer 耗时](developer-guide/08-features/rollout-timing-metrics.md)。

### Algorithm 分区

算法指标统一使用 `algorithm/` 作为第一层前缀。第二层保留统计对象或来源，便于判断分母和采集阶段：

| 字段 | 内容与口径 |
| --- | --- |
| `algorithm/trajectory/reward_*` | 每条完整轨迹的有限最终 reward；mean/min/max/std 与有效观测数 |
| `algorithm/group/reward_std_min,max` | 先计算每个有效 GRPO 组内 reward 样本标准差，再比较各组极值 |
| `algorithm/trajectory/advantage_*` | 有效 GRPO 完整轨迹的 advantage 分布、正负/零计数、零值百分比与 token 加权均值 |
| `algorithm/kl/actor_ref_mean`、`actor_ref_loss`、`coefficient` | 实际训练 forward 的 Reference KL、服务明确返回的 loss 贡献和系数 |
| `algorithm/weaver/entropy_coeff`、`entropy_loss` | 服务使用的熵系数与返回的熵 loss 项 |
| `algorithm/weaver/entropy`、`entropy:sum`、`entropy:tokens` | 温度 1 的全词表策略熵（nats）、熵总和与有效 mask token 数 |
| `algorithm/weaver/ppo_clip/*` | PPO 裁剪比例与当前策略对固定分母策略的 KL 估计 |
| `algorithm/weaver/is:*` | 服务的概率比、log 比、平方、极值与相关统计 |
| `algorithm/weaver/ratio:*`、`algorithm/weaver/logprob_diff:*` | 自定义 loss 的概率比与 logprob 差异诊断，按各字段定义解读 |
| `algorithm/sampling/*` | 更新前原始采样 logprob 分布与逐 segment 计算的 PPL，不是更新后模型效果 |
| `algorithm/mis/*` | 更新前 trainer 重算概率与原采样概率的差异、权重和过滤情况 |
| `algorithm/opd/*` | 对齐后的 student/teacher 概率差异与有效 token 数 |

上表中的简写后缀沿用同一行的完整前缀；全部精确字段、单位和分母见指标字典。只上报新名称，不额外保留同值的旧曲线；统计计算和提交时序不变。正式 step 日志也使用新名称，服务原始响应诊断仍保留服务原字段。`weaver/loss`、`weaver/grad_norm` 与 `training/*` 保持原名，不移入 Algorithm。

Weaver 的 `entropy` 诊断独立于 `entropy_coeff` 和 `loss.reduction.mode`；支持该指标的服务版本在系数为零时也会返回。跨 forward/backward 请求和 optimizer 更新时，NorthRL 分别累加 `entropy:sum` 与 `entropy:tokens`，再用总和相除，不直接平均各请求的均值。服务明确返回空 mask 时保留三个零值，缺失观测仍留空。W&B 字段为 `01_algorithm/56_weaver/entropy`、`01_algorithm/57_weaver/entropy:sum` 和 `01_algorithm/58_weaver/entropy:tokens`。原有 `entropy_loss` 和 `entropy_coeff` 的名称与含义不变。

使用新代码启动的训练，在 workspace 启用自动图表时，算法指标会集中显示在 `01_algorithm` 分区。启动时的布局设置也会对分区内带编号的算法图表排序。正在运行的旧进程不会自动加载代码变更，历史指标数据不回填。

轴与可见性设置参考 [W&B Run.define_metric](https://docs.wandb.ai/models/ref/python/experiments/run)。这些设置不会迁移历史 run 或重排已有 workspace。

## 长度、次数和分母

`batch/tokens_total`（W&B：`15_batch/tokens_total`）按当前 step 本地保存的全部序列
完整长度求和，包含 response token，以及 prompt、工具上下文等非 response 位置。
不同 segment 重复保存的上下文按每段分别计数。它与 `batch/response_tokens_total`
使用相同的原始消费批次口径，在重采样、训练筛选和 optimizer 补齐前统计，每步重新计数。
空序列及不携带本地 tokens 的托管 SampleRef 贡献零，因此该值不代表托管 SFT 的服务端工作量。

| 对象 | 长度/时间含义 |
| --- | --- |
| `model_request/input_tokens_*` | 每次请求的完整输入，包含历史与工具返回；有效请求等权 |
| `model_request/output_tokens_*` | 仅本次请求新增输出；缺 usage 的请求不补 0 |
| `model_request/total_tokens_*` | 同一次请求 input + output，不是 trajectory 总长度 |
| `segment/response_tokens_*` | 保存段内 `loss_mask > 0` 的位置数 |
| `trajectory/response_tokens_*` | 完整轨迹各保存段 response token 数之和；不是运行过程中所有生成 token |
| `group/response_tokens_min,max` | 先对组内完整轨迹求和，再对各完整观测组取极值；组内数量总和没有 mean |
| `timing_rollout/trajectory_*_seconds` | 首次执行开始至最终结果返回，包含准备、模型/工具、评估、清理、内部等待与重试；排除执行前排队和完成后的 pool 等待 |
| `timing_rollout/group_*_seconds`（`mean`、`min`、`max`） | 每组最早进入 worker 到最晚结束的跨度；只使用完整组与有效、可比较时间戳，均值按有效组等权，不相加成员耗时 |

group 是同一 `(model_tag, group_id)` 下的采样组，不同模型的同名组分开统计；它不是独立的 task 对象。组耗时要求已提供的 `expected_rollouts` 与组大小一致；提供 `run_id` 时要求完整、唯一的采样槽位。

fully-async 下，`training/train_step` 表示 group/trajectory 被哪个已完成的 batch 消费，不表示它们在哪一步执行；采样可以与前面的训练步重叠。因此执行耗时不等于该步 trainer 等待取批的时间；等待看 `timing/trainer_wait_for_batch_seconds`。

输入计数和原始 rollout 指标在失败槽位替换前记录，包含失败、跳过和没有训练序列的完整轨迹。
普通非空单轮轨迹贡献一段，空对象贡献零段。advantage 统计使用修复后的 GRPO 组。
`batch/resampled_group_count` 统计本批次修复组数，`batch/resampled_trajectory_count`
统计被补齐的失败槽位数。可训练集合包含替换成员，在 Datum/mask 处理之后、optimizer
尾批复制之前统计。三种集合不能混用分母。

`*_count` 是个数；`*_seconds` 是秒；新 `*_percentage` 是 0–100。既有 MIS `fraction`、自定义 loss `frac` 和验证比例仍为 0–1，概率比 `ratio` 是倍数，不能统一乘 100。比例的精确分母在字段说明中给出。

缺失、不适用或没有有效分母时省略值；适用且确认没有发生时才记录 0。均值只使用有效观测，不把缺失补 0；不完整组不以部分事件和冒充完整组总数。Reward、Advantage、Sampling logprob 的 std 使用 `ddof=0`；组内 Reward std 使用 `ddof=1`，单成员组为 0。

## 模型生成、reward 判题与 rollout 超时

先区分执行阶段：生成答案的模型请求、reward 判题、整条 rollout 的总期限分别观测。reward 内调用的 LLM judge 属于判题，即使它也调用模型 API，仍不计入生成阶段的 `timeout/model_request_count`。

| 发生阶段 | 看哪个指标 | 统计限制 |
| --- | --- | --- |
| 模型生成请求 | `timeout/model_request_count` | 已观测逻辑 `generate` 调用最终被识别为 timeout；后端内部超时重试后成功不计。当前 SingleTurn 验证未汇总此项。 |
| reward 的 E2B 操作 | `verifier/timeout_count` | 显式上报的判题沙箱超时事件；CodeE2B 仅 `run`，IfVerify/VerInstruct hard checker 包含创建、上传、执行。 |
| reward 超时重试耗尽 | `verifier/timeout_after_retry_count` | 因判题流程超时给 0 的完整轨迹数，不是事件次数。 |
| reward 的 LLM judge | `verifier/judge_error_count` | 最终返回 `judge_error` 的结果；Reasoning360 尚未区分 API timeout、HTTP 400、网络错误等类型。 |
| 整条执行达到总期限 | `timeout/rollout_count` | 只在 worker 有明确观测时记录；不等于生成和判题超时之和。 |

判题 `run` 超时可能来自生成程序过慢、judge/checker 卡住或沙箱通信问题，指标标明发生阶段，不能单凭它认定是模型质量还是 reward 服务故障。judge 正常返回的程序测例 TLE 也没有独立汇总，需查看原始测试结果；`finish_reason=length` 是 token 上限而不是 timeout。详见[模型生成与 reward 判题的失败、超时统计](developer-guide/08-features/verifier-metrics.md)及[group、trajectory 与 verifier 耗时](developer-guide/08-features/rollout-timing-metrics.md)。验证的 `val/verifier/*` 使用相同判题口径；目前没有 `val/timeout/model_request_count`。抛异常并触发停止时可能没有完成步的 W&B 行，字段缺失不能当作 0。

工具/模型请求及其 timeout 只统计最终交付数据所属的执行；沙箱重试次数和完整 trajectory 耗时覆盖全部尝试。工具结果按 rollout、执行编号和调用身份去重。无法关联到 segment 的事件保留完整轨迹的信息，不复制或分摊到所有段。

- `timeout/tool_count`：已观测工具返回中明确确认的 timeout 事件。
- `timeout/model_request_count`：模型请求 timeout 事件。
- `timeout/rollout_count`：最终整条 rollout 超时的完整轨迹数。

三者可能来自同一个故障，不能相加。受影响 trajectory / segment 另用去重计数；`segment_from_timed_out_rollout_count` 表示超时完整轨迹对象携带的保存段，不表示每段各发生一次 timeout。三层 timeout 总数记录在 `timeout/*`，不在 `batch/*` 重复创建同义字段。没有观察到工具结果只会进入 `tool/unobserved_call_count`，不能推定 timeout。

运行时 API adapter 从后续模型请求携带的历史消息中观察工具结果。末次工具结果如果没有再随模型请求发送，可能一直未被观察到。`timeout/tool_count` 只统计已观测工具返回中确认的超时；值为 0 不代表所有已发出的工具调用都已完成或都没有超时。`tool/unobserved_call_count` 是未观察到结果的已发出调用数，不是 timeout 次数。没有可靠元数据的自定义 worker 不会得到虚构的零工具/零 timeout 曲线。

运行时在模型请求处理阶段保存 trace，并在退出时进行最终 flush。最终 flush 只能持久化已经采集的观测，不能恢复此前未观察到的末次工具结果。

## NorthGate 工具调用诊断

E2B rollout 从 JSON 和 SSE 响应读取顶层 `tool_call_diagnostics`。`checked` 和 `malformed` 接入 `tool/format_unknown_turn_count`、`tool/format_error_turn_count`；本地结构错误也能确认该轮有错。这是解析/结构检查，不是完整 JSON Schema 校验。缺少该字段时保留旧 parser 诊断路径。

`detected`、`invalid` 分别接入 `tool/calls_detected_count`、`tool/calls_invalid_count`。原文计数包含被修复或丢弃的调用，不再从返回工具列表重复累加。数量为 null 时不补估计调用数，每轮增加一次 `tool/call_counts_unknown_turn_count`，即使 `malformed=true` 已确认该轮有错。这项覆盖计数只针对 NorthGate 诊断；调用累计数代表已知部分。

`tool/format_reason_<reason>_turn_count` 统计包含各个固定原因的回复数：`incomplete_tool_call`、`invalid_arguments_json`、`missing_tool_name`、`invalid_tool_call_structure`、`arguments_not_object`、`duplicate_parameter`、`undeclared_tool`。同一原因每轮最多一次，原因可重叠，不能当作无效调用个数。未知原因不创建动态指标。现有 `tool/arguments_json_error_count`、`tool/arguments_type_error_count` 仍表示对返回参数的本地检查。

这些 step 计数覆盖本批消费的完整轨迹所对应的最终执行。相应 `live/tool/*_turns` 覆盖所有已观察到的 attempt 和重试，详见[运行监控日志](developer-guide/08-features/live-metrics.md)。诊断只影响 metrics，不修改 reward、advantage 或工具接受规则。

## 服务指标、MIS 和验证

Weaver 原始响应先留诊断日志。正式曲线要求可合并语义：计数求和、极值取极值、均值/比例使用实际权重或分子分母，std 需要可合并充分统计量。多分块 PPO/IS 均值缺少明确权重时省略，不以“各块平均值再平均”替代；全局 min/max 可以独立保留。

Reference KL 的 `algorithm/kl/actor_ref_mean`、`algorithm/kl/actor_ref_loss`、`algorithm/kl/coefficient` 依赖实际 forward 输出和对应 loss 支持。`actor_ref_mean` 在本轮实际训练 forward 的 target 对齐、有效 response 位置计算配置的 estimator，跨请求/更新按 token 加权；服务实际收到的 optimizer 补齐或重复样本也计入，因此与 Batch 唯一数据量的口径不同。`actor_ref_loss` 仅接受服务明确给出的实际 loss 贡献，不用 coefficient × mean 推算。未返回真实值时不合成曲线，原始含义不明的 `kl_loss` 不会直接改名。熵与 PPO/IS 使用 `algorithm/weaver/*` 保留服务来源；loss 和 grad norm 继续使用 `weaver/*`，不创建重复字段。`weaver/grad_norm` 在一轮多次 optimizer 更新时，按成功更新的观测值等权取均值；服务的梯度裁剪前/后定义需按服务契约解读。

三个比较对象分别为 actor 对 reference、训练 current 对固定 denominator（PPO/IS），以及更新前 trainer 重算概率对 rollout 原采样概率（MIS）。只在策略对、mask、权重和 reducer 都确认一致时去重 PPO/IS 同义项；catalog 不按数值近似或正负号自动合并。

内部已确认的 `mis/log_ppl_diff` 对外归一到 `algorithm/mis/kl`；`val/<dataset>/avg@k` 不再单独写 W&B，与 `pass@1` 重复。自定义验证字段继续保留，不使用全局白名单。验证均值各自使用有限 score 的观测分母；验证数据不是训练 batch，初始验证可以落在 step 0。

MIS 的分布与过滤比例没有有效观测/分母时留空。`algorithm/mis/optimizer_steps` 是过滤后的计划更新数，实际完成数量看 `training/optimizer_step_count`。OPD 仅在 teacher/student logprob 对齐且启用对应算法时记录。

## 提交时序与普通日志

旧 `timing/batch_release_to_claim_seconds` 和 `timing/batch_claim_interval_seconds` 已退役，不再采集或登记，规范化时也会丢弃旧 producer 提交的这两个字段。完成吞吐看 `timing/batch_completion_interval_seconds`，trainer 等待取批看 `timing/trainer_wait_for_batch_seconds`，首批也记录等待。旧进程已漏报的时间不会自动回填。

worker 元数据随完整轨迹进入 pool，trainer 在取 batch 时固定成员。prepare/recompute/MIS/Datum 与训练服务返回值合并到该批次；训练完成后以 completed step 暂存，权重同步追加同一步，验证触发时等待验证并提交，否则 flush 同一步。正常路径每个训练轮次一条 W&B history，已提交旧步的迟到写入会被忽略。全批跳过仍提交输入数据量和实际更新次数 0，并推进 step。

`LocalLogger` 使用同一批次的指标，整数不打印成三位小数，极小 loss 使用有效数字/科学计数法。阶段日志应标明阶段，尚未计算的训练量不能作为最终 0 展示。约 30 秒的运行快照继续使用 `[rollout-monitor]`，其累计工作量覆盖进程内所有尝试，与训练 step 的最终执行集合不同。详细说明见 [运行监控日志](developer-guide/08-features/live-metrics.md)。

旧 `rollout/*`、`datum/*`、`round/*`、含混长度前缀、reference 覆盖计数与同步计数、recompute 数量、MoE、scale、loss denominator 和未知服务字段不会进入新 step 曲线。此迁移不回填旧 run，也不根据旧聚合量推算新 cohort。

调试 dump 文件名继续使用历史的零起始 batch index；round manifest 另有 `training_step` 标明完成轮次。对外阶段日志与 W&B 使用完成轮次，因此不能只凭文件名数字直接等同 W&B step。

## 模式兼容性

| 模式 | 通用 worker | E2B worker |
| --- | --- | --- |
| sync | 按本批完整轨迹对象/保存段统计，已适配采集通用耗时 | 支持；最终执行事件与全程重试/耗时范围分开 |
| batch-async | 指标按实际消费 batch 归属；字段以 worker 观测能力为准 | E2BRolloutWorker 仍拒绝该模式 |
| fully-async | 按实际取走的完整组归属 step，不用运行累计差值 | 已放开；step 指标归属训练实际消费的完整组 |
| SFT | Batch / Training / Timing 和适用服务字段；不创建 GRPO group/advantage | 不属于当前 E2B 配方 |

自定义 worker 缺少工具/请求观测时相应字段留空。组耗时只在时间戳可比较时记录，不能相减来自不同机器的 monotonic 绝对值。上述兼容性描述指标通路和现有运行限制，不代表已经完成所有服务配置的端到端测试。

## 增量 Weaver metrics-v1

使用 Weaver 后端且启用 W&B 时，所有返回的 v1 观测自动写入 `weaver_metrics_v1/`，
复用 SDK 的标量、布尔值、状态表和专家直方图展示。它们不会进入 NorthRL 旧指标的
聚合逻辑。每次操作保留独立观测行，不会把多个优化器窗口的 top-10% 熵均值再平均。

v1 横轴为 `weaver_metrics_v1/step`（Trainer 优化器尝试次数，包含跳过的更新），
不同于 `training/train_step`（NorthRL 完成的批次）。仅前向操作使用操作序号横轴；
不同模型或重启的 Trainer 使用独立短前缀，身份信息存入 run config。现有 NorthRL
图表不变。W&B 内部 `_step` 只是历史行游标，不是训练步数；每个 NorthRL 批次现在
可能推进多行。

SDK 回调只将观测入队，不执行 W&B I/O。NorthRL 在原有轮次提交后或退出时发布，
避免提前提交尚未完整的 RL 数据。没有只允许默认指标的过滤。本地 JSONL 独立保存，
保留原始名称、数值、标签和状态。
`backend.metrics_store` 和注册模型的 Trainer 采集 YAML 详见
[配置说明](developer-guide/11-configuration-reference/trainer-config.md#weaver-metrics-v1)。
