[English](../../../en/developer-guide/08-features/verifier-metrics.md) | 简体中文 | [文档目录](../../index.md)

# 模型生成与 reward 判题的失败、超时统计

一次 SingleTurn 执行先请求训练模型生成答案，再调用 reward function 判题。判题用的 LLM（例如 Reasoning360 STEM judge）属于 **reward 阶段**，与生成答案的模型请求分开。指标首先区分故障发生在哪个阶段；具体原因还需要结合错误类型、测试结果和日志判断。

<a id="failure-stages"></a>

## 先区分故障阶段

| 要区分的问题 | 训练指标或诊断来源 | 如何解释 |
| --- | --- | --- |
| 生成答案的模型请求超时 | `timeout/model_request_count` | 被观测的逻辑 `generate` 调用最终被识别为超时；不包含 reward 的 LLM judge 或 E2B 请求。不能单凭它判断是推理服务、网络还是客户端等待超时。 |
| 模型生成达到 token 上限 | `trajectory/finish_reason_length_count` | `finish_reason=length`，直接给 0 并跳过判题；这是长度限制，不是 timeout。 |
| reward 的 E2B 判题流程超时 | `verifier/timeout_count` | 沙箱操作未在预算内完成，包含重试后恢复的超时；不包含模型生成请求。支持的操作阶段见下文。 |
| reward 超时重试后仍未返回正常结果 | `verifier/timeout_after_retry_count` | 最终因判题流程超时给 0 的 trajectory 数；不是超时请求次数，也不代表已确定服务端根因。 |
| reward 的 LLM judge 请求失败 | `verifier/judge_error_count` | 当前包含最终返回 `judge_error` 的结果；Reasoning360 将 timeout、网络错误、HTTP 400 等合并在这里，没有单独的 judge API timeout 计数。 |
| 模型生成的程序在某个测例中超时（TLE） | 暂无独立汇总；检查 CodeE2B 返回的 `sandbox_result.tests` | judge 正常返回失败测例时，当前归入 `failed_tests`。不能用 `verifier/timeout_count` 当作程序 TLE 数量；原始测试字段由沙箱 judge 决定。 |
| 整条 rollout 达到总执行期限 | `timeout/rollout_count` | 需要 worker 明确报告整条执行超时。它不是模型请求和 reward 超时之和，当前 SingleTurn 没有对应总期限观测。 |
| 训练服务的 forward / backward 操作失败 | trainer / Weaver 错误日志 | 属于生成和判题完成之后的训练阶段；不计入上述生成或判题超时。 |

E2B 的 `stage=run` 超时只说明判题命令未及时完成。模型生成的程序太慢、checker/judge 卡住、沙箱通信故障都可能导致这个现象；需要结合 `stage`、`error_type`、沙箱探测和原始测试结果定位。`verifier/service_error_count` 是历史字段名，表示**判题流程未返回正常结果**，不能据此认定是 reward 服务实现的问题。

模型请求的观测也有边界：当前训练 SingleTurn 能采集逻辑 `generate` 的最终超时；Weaver 内部重试中间发生超时、最后成功时，不会增加这个计数。验证路径未汇总模型请求超时，因此目前没有 `val/timeout/model_request_count`；下文的 `val/verifier/*` 只描述验证的 reward 判题。缺少字段不表示超时为 0。

<a id="verifier-counters"></a>

## 判题结果与重试计数

每步训练在原始 batch 进入 GRPO 过滤之前统计判题结果。验证使用相同口径，并加 `val/` 前缀。每条父 trajectory 计一次，多 segment 不重复计数。聚合时输出普通日志 `[verifier-metrics]`；W&B 随完成的训练步或验证快照提交，提前终止时可能未提交。

| 指标 | 含义 |
| --- | --- |
| `verifier/result_count` | 收到的可识别 reward score 或非空 reason 数量，包含 `finish_reason_length` 跳过结果；不是实际判题调用次数。 |
| `verifier/service_error_count` | 判题流程最终未返回正常结果、reason 为 `judge_error` 或 `timeout_after_retry` 的数量；不用于认定故障根因。普通答错、生成达到长度限制、缺少代码不计入。 |
| `verifier/judge_error_count` | 判题 API 重试后仍失败、返回 `judge_error` 的结果数，包括 `reasoning360/.../judge_error`。一次逻辑判题内部的多个请求不会重复计为多个失败结果。 |
| `verifier/timeout_count` | reward function 显式报告的 E2B 操作超时事件总数，包括随后恢复的超时；不包含模型生成、LLM judge API timeout 或正常返回的测例 TLE。 |
| `verifier/retry_count` | reward function 实际报告的重试总次数。 |
| `verifier/retried_trajectory_count` | 经历判题重试的 trajectory 数量，每条计一次。 |
| `verifier/retry_completed_count` | 重试后返回正常判题结果的 trajectory 数量。结果可以是答对给 1，也可以是答错给 0。 |
| `verifier/timeout_after_retry_count` | 重试耗尽、最终因超时给 0 的 trajectory 数量。 |
| `verifier/no_code_count` | 提取不到非空代码、以 `no_code` 给 0 的结果数。 |

例如一条 IfVerify 结果第一次超时、重试后判错，会记录 `timeout_count=1`、`retry_count=1`、`retried_trajectory_count=1`、`retry_completed_count=1`，而 `service_error_count=0`。如果允许的三次 `run` 尝试中判题命令都超时，则记录 `timeout_count=3`、`retry_count=2`、`timeout_after_retry_count=1`、`service_error_count=1`。

<a id="reward-coverage"></a>

## 各 reward function 的观测范围

CodeE2B 一条 trajectory 可能包含多个测试 batch，因此超时和重试事件数可能大于 trajectory 数。它的通用 `timeout_count` / `retry_count` 对应 `run` 阶段的 judge 命令超时和重试。创建、上传、warmup 的可恢复故障使用各自独立的重试预算；恢复后的 `create_*`、`upload_*`、`warmup_*` 诊断保存在原始 `score.metrics`，不合入通用判题超时与重试计数。预算耗尽或遇到不可恢复错误时抛异常。部分整任务执行路径还会对可恢复的非超时命令故障重试，因此 `retry_count` 也不等同于 timeout 次数。

IfVerify 和 VerInstruct hard checker 上报沙箱创建、上传、执行超时；它们的通用重试计数覆盖这些阶段，并另存阶段计数。只有耗尽 `run` 预算且该阶段每次故障都是命令超时，才返回 `timeout_after_retry`；不可恢复的基础设施错误、整次 attempt 到期或清理时仍未终止的命令会抛异常。清理阶段的超时只记录清理日志，不作为失败判题或判题重试统计。

LLM judge API 的 timeout 和内部请求重试次数尚未单独上报。Reasoning360 最终返回的 `judge_error` 计入判题失败结果；VerInstruct soft judge 使用 Recast，其 API 重试耗尽后直接抛异常，不产生上述正常 reward 结果。

同一套计数按实际使用的 reward function 拆分，例如 `reward_fn/code_e2b/verifier/timeout_count`、`reward_fn/reasoning360/verifier/judge_error_count`。验证对应 `val/verifier/service_error_count`、`val/reward_fn/if_verify/verifier/retry_completed_count` 等。

<a id="aggregation-limits"></a>

## 汇总边界与缺失数据

这些是当前 batch 或验证快照的数量，不是进程累计值。计数读取父 trajectory 的 `score.reason` 和明确报告的 `score.metrics.timeout_count/retry_count`，不展开任意 judge 数值分数，也不混入远程 rollout 的 `sandbox_retries`。非法的负数、非整数、布尔值和非有限计数会被忽略；缺失事件计数不根据 reason 猜测。

这套 verifier 计数需要父 trajectory 提供可识别的 reward score 或 reason。训练 wrapper 可为抛出异常的执行保存 failed 空 trajectory 和模型请求观测，但不会凭异常补造 reward 结果；`stop_on_error` 提前终止时，完整 batch 或训练步可能尚未完成，W&B 不保证记录到这次故障。此时需要查看普通错误日志，不能从缺少曲线推断没有超时。修改 worker 或 reward function 后，运行中的训练进程需要重启才能加载新逻辑。

训练页面入口见[训练 step 指标](../../metrics.md)；耗时定义见[执行耗时](rollout-timing-metrics.md)。Agent 沙箱的运行期累计量见[运行监控](live-metrics.md)，任务完成和评分规则见[任务接口参考](../05-rollout-workers/e2b-task-contract.md#scoring-and-failures)。
