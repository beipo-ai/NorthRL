[English](../../../en/developer-guide/08-features/rollout-timing-metrics.md) | 简体中文 | [文档目录](../../index.md)

# Batch rollout、trainer 等待与执行耗时

本页说明[训练指标](../../metrics.md)中各类耗时的统计范围。每个训练步汇总该步领取的原始 batch，在 GRPO 过滤和 optimizer 补齐之前统计。每个父 trajectory 计一次，多 segment 不重复计数。fully-async 下，`training/train_step` 表示数据被哪个已完成的训练 batch 消费，不表示采样在该 step 开始或完成；采样可以与前面的训练步重叠。

<a id="batch-timing"></a>

## 大盘先看完成间隔和等待

`timing/` 只保留 `batch_completion_interval_seconds` 和 `trainer_wait_for_batch_seconds`。前者是相邻两次 `train()` 返回时间差，覆盖上一轮同步、验证、日志、数据等待及本轮完整训练与保存；首次返回建立基准，不补 0。后者是 trainer 发现完整 batch 尚未就绪后，直到成功领取的已观测等待，含轮询和调度延迟；立即领取为 0，初始化、验证暂停、训练和权重同步不计入。

`timing_train/` 记录纯训练总时间与内部阶段，`timing_weights/` 记录导出、checkpoint 和同步，`timing_rollout/` 与 `timing_val/` 分别记录采样和验证执行分布。固定训练计时共 27 项：2 项总览、8 项训练、5 项权重/保存和 12 项 rollout 分布。NorthRL 启动时会配置个人 workspace 的分区排序；生效范围与保留设置见[启动时设置 workspace 顺序](../../metrics.md#启动时设置-workspace-顺序)。

<a id="execution-distributions"></a>

## 诊断单条、组和判题耗时

本文使用规范指标名；W&B 会添加稳定的显示序号，具体见[训练 step 指标](../../metrics.md)。另按适用范围生成 reward function 与验证耗时曲线。耗时字段仅以一个 `/` 分隔分区与名称，分布名称为 `<object>_<mean|min|max>_seconds`。原始 worker 元数据和 `[timing-metrics]` 日志可以保留 producer 名称。

| W&B 指标 | 统计口径 |
| --- | --- |
| `timing_rollout/trajectory_mean_seconds` | 每条父 trajectory 的完整执行耗时均值，包括生成、工具、判题、转换、清理、内部等待和重试；不包含进入 worker 前的排队及完成后的 pool 等待。 |
| `timing_rollout/group_mean_seconds` | 同一 `(model_tag, group_id)` 采样组内最晚结束减最早开始，再对有效完整组等权求平均；包含同组成员错开开始造成的跨度。不同模型的同名 group 分开统计。 |
| `timing_rollout/verifier_mean_seconds` | SingleTurn worker 的一次完整 reward evaluation 耗时，包括 reward function 内的 LLM judge、沙箱创建、执行、清理和重试，不包含模型答案生成。 |
| `timing_rollout/model_request_mean_seconds` | 被观测的答案生成请求耗时，包含请求内等待、传输和重试；每次被观测请求等权。 |
| `verifier/skipped_length_count` | 明确因 `finish_reason=length` 跳过判题的 trajectory 数量。这些 trajectory 仍计总耗时，不作为零秒 verifier 样本参与均值。 |

四类耗时分布还报告 `_min_seconds`、`_max_seconds`。均值只使用有效、有限、非负的实际观测；没有观测时省略耗时，不补 0。有效计时观测数保留在内部和原始诊断日志中作为计算分母，但正式 W&B 不再上报这些计时计数，也不新增计时计数分区。已有 batch、group、model request、reward 等数量字段保持不变，不能替代有效计时观测数作均值分母。

group 是采样组，不是独立的 task 抽象。group 耗时要求所有成员起止时间有效且时钟可比较，已提供的 `expected_rollouts` 与组内数量一致；提供 `run_id` 时要求完整、唯一的采样槽位。不完整或重复的组不参与组耗时统计。缺失、空或非有限数值的组 ID 不视为真实 group；SingleTurn 为默认单次采样、没有有效组 ID 的验证任务生成独立组身份。`timing_rollout/group_mean_seconds` 按组等权，而 `group/response_tokens_*` 等组内数量总和仍只报告各组 min/max，没有 mean。

例如一个 group 的两条 trajectory 执行区间为 `[100, 110]` 和 `[105, 120]`：group 耗时为 20 秒，trajectory 平均耗时为 12.5 秒。不能用两条耗时之和 25 秒或整个 batch 的跨度除以条数代替。

按有效 reward function 拆分的执行耗时仍放在 rollout 分区，使用 `timing_rollout/<reward_fn>_<object>_<mean|min|max>_seconds`，例如 `timing_rollout/code_e2b_group_mean_seconds`、`timing_rollout/if_verify_group_max_seconds`。全 batch 的 trajectory/verifier 均值按各实际观测等权计算，group 均值按有效完整组等权计算，不平均各 reward function 的均值。

验证执行耗时使用相同的简短名称，放在 `timing_val/`，例如 `timing_val/trajectory_mean_seconds`、`timing_val/group_mean_seconds` 和 `timing_val/code_e2b_verifier_mean_seconds`。当前 SingleTurn 验证采集总耗时和 verifier 耗时，未汇总 `timing_val/model_request_*_seconds`；其他 worker 没有明确 verifier 计时的结果不参与 verifier 均值。

<a id="training-and-weights"></a>

## 训练、权重保存和重叠关系

| W&B 指标 | 统计口径 |
| --- | --- |
| `timing_train/batch_training_seconds` | 标准 trainer 从进入训练到开始保存前的客户端墙钟耗时，含数据处理、重算、FB/optimizer；不含权重导出、checkpoint 或同步。无标准分段计时的自定义 trainer 回退完整调用。 |
| `timing_train/recompute_logprobs_seconds` | 训练端重算 logprob 的客户端墙钟耗时，含请求等待。 |
| `timing_train/ref_logprobs_seconds` | reference 模型前向的客户端墙钟耗时。 |
| `timing_train/mismatch_forward_seconds` | 独立 mismatch 诊断前向的客户端墙钟耗时，不含随后统计。 |
| `timing_train/forward_backward_wall_seconds` | 全部 FB 阶段的客户端墙钟耗时合计，含 Datum 准备、提交、服务排队、网络和结果轮询。流水线按首个提交前到所有结果完成的整体跨度计，不重复累加重叠等待。 |
| `timing_train/forward_backward_compute_seconds` | 完整有效 FB 服务 `compute_ms` 观测之和除以 1000。 |
| `timing_train/optimizer_step_wall_seconds` | 全部 FB 成功之后，optimizer 提交到结果完成的客户端墙钟耗时，多次更新相加。 |
| `timing_train/optimizer_step_compute_seconds` | 完整有效 optimizer 服务 `compute_ms` 观测之和除以 1000。 |
| `timing_weights/weight_export_seconds` | 成功导出采样权重的客户端墙钟耗时。 |
| `timing_weights/checkpoint_seconds` | 本步成功完成的 checkpoint 保存客户端墙钟耗时；保存失败时不报此字段。 |
| `timing_weights/weight_sync_seconds` | `train()` 返回后的采样权重同步耗时。 |
| `timing_weights/weight_export_failed_seconds` | 采样权重导出失败前耗时。 |
| `timing_weights/weight_sync_failed_seconds` | 采样权重同步失败前耗时。 |

训练总时间不等于阶段之和：数据处理等其他工作没有单独曲线，部分阶段包含其他工作，采样也与训练重叠。服务 compute 与客户端 wall 是同一操作的不同口径，不能相加。服务 `compute_ms` 尚不能保证为纯 GPU 时间；缺失、非有限、负数或不完整观测不补零，也不以部分观测冒充本步总计算耗时。

<a id="legacy-fields"></a>

## 旧字段与缺失观测

Batch rollout span、旧 batch release/claim 间隔，以及单独的 MIS/mismatch 统计、prepare、other 计时已退役，不再采集或登记；相关算法和统计继续运行。规范化会丢弃旧 producer 的这些字段。旧 `timing/batch_training_seconds` 和 `timing/training_time` 包含保存，语义不同，直接丢弃而非迁移为新的训练总时间。旧 `timing/mismatch_forward` 含随后统计，直接丢弃，不映射为新的纯前向计时；其他同义旧名迁移到当前分区。

耗时长不等于 timeout；`finish_reason=length` 也不是 timeout。生成请求超时看 `timeout/model_request_count`，reward 的 E2B 超时看 `verifier/timeout_count`。LLM judge 属于 verifier，不属于答案生成。详见[模型生成与 reward 判题的失败、超时统计](verifier-metrics.md)。

这些是 batch/验证快照统计，不是进程累计值；训练中断可能没有成功完成步的 W&B 行，初始验证可落在 step 0。使用新代码需要重启训练进程。正式规范化将原始 producer 名和上一版 `timing/<object>/duration_seconds_*`、`timing/reward_fn/...`、`val/timing/...` 等名称映射为对应分区的新名，不重复上报旧耗时或历史 `task` 耗时别名，也不回填历史曲线。启动时的 workspace 排序是另一项操作，见上文链接。

批次边界与采样版本的关系见[同步训练语义](../06-trainers/sync-rollout-semantics.md)；运行期累计指标见[运行监控](live-metrics.md)。
