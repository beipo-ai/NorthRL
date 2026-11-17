[English](../../../en/developer-guide/05-rollout-workers/e2b-runtime.md) | 简体中文 | [文档目录](../../index.md)

# E2B 执行机制与轨迹存储

本文说明平台如何运行任务、连接模型服务、保存轨迹和回收评测日志。任务接入步骤见[接入指南](e2b-task-integration.md)，函数与返回值见[接口参考](e2b-task-contract.md)。

实时沙箱与模型请求计数见[运行监控](../08-features/live-metrics.md)，已完成训练步的图表见[训练指标](../../metrics.md)。

<a id="components"></a>

## 组件分工

| 组件 | 运行位置 | 职责 |
| --- | --- | --- |
| `e2b_rollout_worker.py` | 宿主侧 worker | 从训练样本构造请求、取得已发布的 sampler gateway、调用 executor、把最终结果转换为训练轨迹 |
| `e2b_executor.py` | 宿主机 | 创建 sandbox、分阶段下发命令、上传评测包、取回结果、重试和销毁 |
| `e2b_runtime.py` / `E2BRolloutRuntime` | sandbox | 平台统一入口、NorthGate HTTP 转发、轨迹记录、调用任务函数、保存阶段和最终结果 |
| 任务模块，例如 `swerebench_claude_code_rollout.py` | sandbox | 准备任务环境、执行 agent、给任务产物评分 |
| `e2b_bootstrap.sh` / `e2b_bootstrap.py` | sandbox | 准备公共 Python 和依赖；由 executor 在第一阶段前调用 |
| `contracts.py` | 宿主机和 sandbox | 定义请求、agent 产物、阶段记录和评分结果的数据类型 |

训练由 worker 调用 executor。任务方无需复制上述平台脚本。

<a id="sampler-gateway"></a>

## Sampler 发布、版本路由与恢复

模型调用经过 `agent → sandbox 内 HTTP relay → 服务端 NorthGate → Weaver sampler`。宿主继续使用 Weaver SDK 管理训练和 sampler，并查询 sampler 的 `gateway.state`，等待 `published` 后自动取得 `gateway.base_url` 和 `gateway.model`。首次启动在共享 sampler 预热后集中等待发布，再启动 workers；后续权重同步在新 sampler 发布完成后才切换 client 和版本并放行下一批任务。同一 session 的成功结果持续复用，普通 rollout 不再定期查询发布状态。`rollout.execution.gateway_publication_timeout_seconds` 限制每次集中等待，默认 300 秒。服务端必须启用 sampler 自动发布，并配置对应模型的 tokenizer、chat template 和响应解析；recipe 无需手动填写 NorthGate URL 或模型名。

成功缓存不会自动探测空闲退役。空闲下架的 sampler 需要由宿主集中调用 `warmup_sampling(refresh_gateway=True)`，通过 Weaver 直接短采样唤醒并重新确认发布；也可以先直接采样或计算 logprobs，再调用 `refresh_gateway_snapshot()`，或创建新 sampler。仅重试 NorthGate 请求、查询 sampler 或发送 heartbeat 不能唤醒入口。发布检查遇到 `retiring` / `retired` 时明确报错，不自动发起额外采样；sampler 已过期或终止时需创建新 sampler。恢复只影响后续任务，已运行的 rollout 保持原 session 和版本。

宿主配置 `WEAVER_API_KEY` 和 `NORTHGATE_API_KEY`。

agent 使用平台提供的本地 OpenAI 或 Anthropic API 地址，并继续发送 `model: northrl`；relay 把模型名改成当前 sampler 的 `weaver-<sampling-session-uuid>`。Anthropic 格式的响应（包括流式的 `message_start`）会把模型名改回客户端请求时用的名字：Claude Code 发现响应模型与请求模型不一致时，会删掉历史里的 thinking 块，平台重新渲染的历史就与采样不一致，每次调用都会切成新的训练片段。recipe 显式配置 `rollout.sampling.max_tokens`（验证时使用 `validate.sampling`）时覆盖 agent 预算，不按剩余上下文裁剪；未配置或为 `null` 时保留 agent 预算。NorthGate 负责分词和上下文限制。

训练轨迹取自 NorthGate 回包中的 `token_data`，保留服务端实际使用的输入/输出 token、log probabilities 以及按训练配置请求的附加字段。

可读 trace 的 `message_format`、`message_system`、`messages` 和 `tools` 对应 relay 转发的最后一次成功模型请求的原生协议内容，其中 `messages` 还包含该次模型响应。`tools` 保存工具定义，包括描述、参数 schema 等原生字段；请求未携带工具定义时为 `null`，显式传空列表时为 `[]`。Anthropic 请求中的协议级 `cache_control` 会在转发和历史匹配前移除，因此这些缓存标记也不进入对应的 trace 请求字段。

<a id="lifecycle"></a>

## 执行顺序

```text
宿主等待当前 sampler 的 NorthGate gateway 发布完成
  → 创建 sandbox，上传平台代码和任务模块
  → 平台调用 prepare（可选）
  → 平台启动 gateway，调用 run_agent
  → run_agent 清理并等待 agent 及子进程退出
  → 平台关闭 gateway，保存最终 trace 和 agent 阶段记录
  → executor 校验本次 rollout / attempt 的阶段记录
  → executor 在同一个 sandbox 上传隐藏评测包
  → 平台调用 evaluate
  → 平台合并轨迹和 reward，写最终结果
  → executor 下载结果并销毁 sandbox
```

`trace.json.gz` 在执行中也会更新，不能用它是否存在来判断 agent 是否结束。上传评测包依赖平台验证的阶段记录。评测阶段复用 agent 留下的 workspace，不再运行 agent 或重置代码仓库。

各阶段时限与故障恢复见[执行时限与恢复](../11-configuration-reference/e2b-execution-budgets.md)。

<a id="trace-schema"></a>

## 轨迹文件与 v2 格式

`trace.json.gz`、`agent-stage.json.gz` 和 `result.json.gz` 使用必填的 `schema_version: 2`，只接受这一版本；没有版本字段的旧文件不再支持读取。宿主 executor 和 sandbox runtime 必须使用匹配的版本。内存中的训练契约 `RemoteSegment` 仍保存完整数组，持久化和传输使用 `northrl.remote_rollout.wire` 中的紧凑模型。

新增 `turn_annotations` 后仍使用 v2：新 reader 可读取缺少该字段的旧 v2 记录，按空列表处理；旧 reader 使用 `extra="forbid"`，会拒绝含该字段的新记录，即使值为空列表。这是单向读取兼容，不能据此混用新旧运行版本；回滚到旧代码后也不能直接读取新产物。

每个 segment 只保存以下内容：

| 字段 | 含义 |
| --- | --- |
| `base_segment` | 当前文件内更早的 segment 索引；没有共享前缀时为 `null`。编码器与上一段比较实际 token ID。 |
| `prefix_len` | 从引用段完整 tokens 开头复用的 token 数；无引用时为 `0`。 |
| `suffix_tokens` | 接在共享前缀之后的本段 token ID。 |
| `response_spans` | 本段完整 token 坐标中的响应区间，使用从 0 开始的半开区间 `[start, end)`，支持多段回答。 |
| `response_log_probs` | 按响应区间顺序保存的原始 logprob，不保存上下文位置的零。 |
| `old_response_log_probs` | 同样按响应位置排列；与本段 `response_log_probs` 完全相同时，可使用 `{"ref":"response_log_probs"}`。 |
| `sampling_mask` | 保留原有 inline/ref 表示，按本段全部响应 token 对齐。 |
| `segment_metrics` | 原有的分段计数。 |
| `turn_annotations` | 本段每轮生成输出的原始诊断和结束信息；缺字段默认 `[]`。 |

<a id="turn-annotations"></a>

## 逐轮诊断

每条 annotation 包含 `version: 1`、非空且本段唯一的 `turn_id`、`response_span: [start, end]`、`diagnostics`、`finish_reason`、`truncated`、`tools_declared` 和 `source: "northgate"`。`response_span` 使用本段完整 tokens 的左闭右开坐标，必须有序、不重叠，且只包含生成位置；它不是压缩后 token 或 response logprob 数组的下标。相邻轮次即使合并为一个 compact `response_spans` 区间，其 annotations 仍分别保留。

Recorder 自动为每轮记录 annotation；`diagnostics` 保留 NorthGate 的原始 `tool_call_diagnostics` 字典或 `null`，不从已接受调用或聚合计数反推。`finish_reason` 保留原始结束原因或 `null`，`truncated` 表示截断或完成状态不确定；`tools_declared` 表示本轮工具声明已知，显式空列表也为 `true`。新 segment 不继承仅作为上下文复制的旧轮次 annotations。训练轨迹裁切时保留之前完整轮次，只裁短受影响轮次的 span 并标记 `truncated: true`，删除完全超出保留范围的轮次。

<a id="trace-decoding"></a>

## 解码与压缩

例如，S0 的 tokens 为 `[10,21,22,23]`，S1 为 `[10,21,22,30,41,50,61,62]`，则 S1 可以保存 `base_segment=0`、`prefix_len=3` 和 `suffix_tokens=[30,41,50,61,62]`。若 S1 只有 `41` 和 `61,62` 是回答，其 `response_spans` 为 `[[4,5],[6,8]]`。解码时这些位置的 `loss_mask` 为 1，其他位置的 mask 和两种 logprob 都补零。共享的 token 在不同 segment 中可以分别作为回答或上下文，因此不能继承引用段的 mask 或 logprob。

所有引用都在同一个文件内，`result.json.gz` 不依赖另一个 `trace.json.gz` 文件才能恢复。元数据、原生消息、工具定义、reward 和状态保留原义。周期性 trace 同步直接校验并保存紧凑格式，不展开完整训练数组；executor 在返回训练结果时解码并校验 `RemoteSegment`。

读取新 trace 的示例：

```python
from pathlib import Path
from northrl.remote_rollout.wire import decode_trace, load_trace_gzip

compact = load_trace_gzip(Path("trace.json.gz").read_bytes())
trace = decode_trace(compact)  # Expand only when full training arrays are needed.
```

这一格式减少存储和传输量，不减少模型计算量；展开后的训练数组大小与原来一致。

三个轨迹文件使用 gzip level 1 压缩，宿主通过二进制下载，并将本地 trace 保存为 `e2b_rollout_<rollout_id>.json.gz`。只接受 gzip 包装的 v2 数据，不回退读取未压缩文件。状态、metrics、request journal 和评测日志仍保持原来的文本格式。

<a id="downloads"></a>

## 下载并发与后台处理

每个 rollout 事件循环共用一个下载队列，实际创建的共享队列最多执行 128 个下载任务，其中 8 个名额保留给心跳和小型控制文件，其余 120 个由普通下载共享。周期 trace 使用普通下载名额，不另设并发子限制。等待队列不设容量上限，下载不会因队满被丢弃；排队项只保存下载操作，不预先拉取文件内容。最终结果、阶段交接和恢复请求优先于周期 trace。每个 rollout 的周期 trace 由一个任务串行同步，上一次同步结束后才等待下一轮。等待执行名额时只挂起调用协程。

单次网络超时从实际开始读取文件时计时，排队仍受阶段或恢复的整体截止时间约束。重试退避期间释放下载名额，恢复的总截止时间不重置。流式评测日志持有名额直到流关闭。普通 I/O 和最终结果下载采用不同的单次时限，见[执行时限与恢复](../11-configuration-reference/e2b-execution-budgets.md)。

宿主对诊断后台 I/O 和最终结果分别限流：诊断工作最多 4 个并发回调，最终结果处理最多 2 个。周期 trace 的 gzip 解析、完整校验和原子落盘交给同一事件循环共享的最多 2 个 spawn 子进程，只回传少量计时数据，不把解析后的 token 对象带回训练进程。成功保存过的相同压缩快照跳过重复校验与写入；无效快照不覆盖上一次有效文件。最终结果默认在线程中解码；`rollout.execution.result_processing: process` 可改用共享的单个 spawn 子进程，仍受两个结果处理名额限制，返回完整的 packed 训练数组。用于恢复训练数据的部分 trace 保持原有解码路径。

下载名额保持到对应处理结束，避免积压已下载的大文件；小型心跳和阶段状态 JSON 直接解析。取消会等待正在运行的后台任务收尾，最后一个使用者退出时关闭对应进程池，防止旧写入覆盖后续尝试。后台限流控制并发和内存，并不保证所有调度都没有延迟；线程中的文件系统操作不能强制中断，文件系统卡住可能使清理超过阶段截止时间。

sandbox 的周期 trace 使用单个后台写入线程，只保留一个最新待写快照；序列化、压缩和原子落盘都在后台进行。agent 收尾时等待最新快照写完，再发布最终 trace 和阶段记录。

<a id="evaluator-logs"></a>

## 最终评测 stdout / stderr

评测包根目录包含 `run-tests.sh` 时，executor 会在打包上传副本时自动加入日志包装。宿主上的原始评测文件和 `rollout_entrypoints/` 无需修改。评分执行前会在原安装路径恢复脚本原文，再以相同参数、工作目录和环境启动 Bash；脚本的 stdout/stderr 一边交给现有评分逻辑，一边按原始字节分别保存。脚本自己的 `set -e`、退出码、相对路径引用和 EXIT trap 保留。

训练已启用 `runtime.debug.enable_data_dump` 并设置 `runtime.debug.dump_dir` 时，日志随现有 E2B trace 回传到：

```text
<dump_dir>/rollout_traces/e2b_rollout_<rollout_id>.evaluation.<attempt_id>/
  stdout.log
  stderr.log
  status.json
```

`attempt_id` 区分环境重试，不覆盖前一次失败的日志。`status.json` 记录 rollout/attempt 身份、评分脚本是否结束、`returncode`（受信号终止时为负数）、采集错误和两个文件的下载状态。评分脚本成功退出的 `returncode: 0` 不代表任务通过；任务成绩仍由现有评测逻辑决定。命令超时或取消后，`finished: false`、`returncode: null` 表示尚未取得脚本最终退出状态，日志可能只有已产生的部分。

executor 在销毁 sandbox 前尝试回收日志，成功、任务失败、评分异常、超时和取消均走同一收尾路径。文件流式下载，不做末尾 2000 字符截断；整个日志回收最多额外占用 10 秒，与最终指标读取并行，仍在 sandbox 的清理宽限期内。连接中断或预算耗尽时保留已经下载的字节，并在 `downloads` 中标明未完成；日志回收失败不会修改 reward，也不会跳过 sandbox 清理。sandbox 已不可访问时无法补回未传出的内容。

其他文件名的自定义评分程序不自动套用这一脚本包装；现有外层命令日志仍正常输出。

<a id="request-cancellation"></a>

## 模型请求取消

agent 退出时，可能还有模型请求没完成。例如，agent 已经修改了文件，退出时取消了仍在等待响应的请求。relay 会记录 `model_requests_cancelled`，但不会仅凭取消信号就认定模型服务故障；如果没有其他执行或清理错误，平台仍可对沙箱中的结果进行评测。

只有在取消发生时满足以下任一条件，relay 才会把这次取消同时记为模型服务故障：

1. 请求此前遇到 HTTP 429、5xx 或连接错误，正在等待重试或重试尚未成功。
2. 已连续至少 300 秒没有收到成功响应头或流式数据块。

这 300 秒从请求开始处理时计时；每次收到成功响应头或流式数据块，都会重新计时。relay 只在收到取消信号时检查这个时长，不会因为达到 300 秒就主动停止请求。
