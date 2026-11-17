[English](../../../en/developer-guide/11-configuration-reference/rollout-config.md) | 简体中文 | [文档目录](../../index.md)

<a id="rollout-worker-configuration"></a>

# Rollout Worker 配置

Rollout worker 的配置选项。

E2B rollout 可设置 `rollout.sampling.reasoning_effort: high`，覆盖每次请求中
agent 的 effort。relay 将其写入 OpenAI 的 `reasoning_effort` 或 Anthropic 的
`output_config.effort`；支持的值由网关和模型决定。省略或设为 `null` 时保留
agent 原设置。验证使用独立的 `validate.sampling.reasoning_effort` 配置，
其他 rollout 类型不支持此选项。

<a id="sampling-mask-retention"></a>

## 保留采样 Mask

`rollout.keep_sampling_mask` 默认为 `false`，直接控制 Weaver 请求中的
`return_sampling_mask`。设为 `true` 时，请求并在 rollout 轨迹中保留 mask，
供后续训练使用。`top_p`、`top_k`、`min_p` 等采样过滤参数不会自动开启此功能。

```yaml
rollout:
  keep_sampling_mask: true
  sampling:
    top_p: 0.95
```

全量微调时开启此选项，需要 Weaver SDK 支持 `sampling_mask_transport="ref"`。
使用 `algorithm.mis.enabled` 或 `algorithm.mis.get_mismatch_metrics` 时，
此选项必须关闭，因为 MIS 暂不支持 sampling mask。

<a id="single-turn-reward-selection"></a>

## 单轮奖励选择

数据集字段、JSONL/Parquet 示例以及逐行混合奖励的说明，参见
[单轮奖励数据](../03-data-loader/data-loader.md#single-turn-reward-data)。

单轮 rollout worker 会立即评估每个生成的响应。如果样本顶层
提供了 `reward_fn`，则使用该值；否则使用 `rollout.reward_fn`。
数据值缺失或为 null 时，使用配置的默认值。未知名称或
空字符串会引发错误，而不会回退。文件数据集中 pandas 的缺失值
视为未提供。CSV 解析保留 pandas 的默认
缺失值规则，包括空单元格。

```yaml
rollout:
  num_workers: 128
  reward_fn: math_answer_only
  type: single_turn
```

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2?"}], "label": "42", "reward_fn": "math_with_format"}
```

虽然配置的默认值是 `math_answer_only`，此样本仍使用 `math_with_format`。
`label` 包含参考答案；仍然支持现有的 `ground_truth` 和
`reward_model.ground_truth` 数据。

| 奖励名称 | 评分规则 |
| --- | --- |
| `math_with_format` | 原始数学规则评分，包含 0.1 的格式奖励 |
| `math_answer_only` | 基于完整响应中方框答案的数学评分，二元 0/1 |
| `math_llm` | 使用 LLM 评判思考块后的答案 |
| `math_rule_llm` | 先进行规则评分，再按需使用 LLM 评判 |
| `math_rule_llm_answer_only` | 对最后一个方框答案进行 MathD/SymPy 评分，再使用兼容 bp 的二元 LLM 回退 |
| `math_boxed_llm` | 使用 LLM 评判提取出的方框答案 |
| `tinker_aime_scalar_match` | 现有的 AIME 标量比较 |
| `rule_nemotron_mcqa` | 在本地提取 Nemotron 多选题答案并进行精确比较，二元 0/1 |
| `rule_dr_sci` | 提取 `<answer>...</answer>` 后进行 MathD/SymPy 等价判定，二元 0/1 |
| `rule_reasoning_gym` | 使用本地 ReasoningGym 规则判题；除 `boxnet` 保留连续分数外，奖励为二元 0/1 |
| `reasoning360` | 按数据来源调用本地规则或 STEM LLM 裁判，二元 0/1 |
| `synlogic` | 使用本地规则评测 35 类 SynLogic 任务，二元 0/1 |
| `ifeval` | 使用本地规则检查全部 IFEvalG 约束，二元 0/1 |
| `if_verify` | 在 E2B 沙箱执行数据行的 `evaluate(response)`，二元 0/1 |
| `code_e2b` | 使用 SandboxFusion 评测代码，全部用例通过为 1，否则为 0 |
| `recast` | 使用 LLM 对完整响应进行指令遵循评判，二元 0/1 |
| `verinstruct` | 先在沙箱中检查硬约束，再使用 LLM 进行指令遵循评判，二元 0/1 |

四种支持 LLM 的模式（`math_llm`、`math_rule_llm`、
`math_rule_llm_answer_only` 和 `math_boxed_llm`）
要求在 rollout 进程环境中设置 `LLM_JUDGE_URL` 和 `LLM_JUDGE_MODEL`。
`LLM_JUDGE_API_KEY` 是可选项。`math_llm` 还要求渲染后的提示词
与响应中包含思考起止标记；`math_rule_llm` 在规则检查未通过时
也有此要求。`math_boxed_llm` 评判提取出的方框
答案，无需思考标记。`math_rule_llm_answer_only`
选择最后一个 `</think>` 或 `###Response` 标记中较晚出现者
之后的区域；两个标记都不存在时，则选择完整响应。它首先使用
MathD/SymPy 将最后一个方框答案与 `label` 比较。规则判定为错误、
缺少方框答案或规则异常时，会回退到 LLM 评判器。SymPy 直接在
奖励协程中运行，不设置软超时，与 `math_rule_llm` 一致。
回退使用 `[[1]]` / `[[0]]` 判定，所有
成功的评判路径都返回二元奖励。如果重试后仍无法
初始化、请求或解析该回退，则发出警告并返回
0；其他 LLM 模式保留原有失败行为。

SimpleRolloutWorker 和 E2B 使用各自的评估路径。

<a id="nemotron-mcqa-reward"></a>

### Nemotron MCQA 奖励

`rule_nemotron_mcqa` 是同步的本地奖励，要求 `label` 为非空
字符串。它首先使用最后一个完整且大小写敏感的
`<answer>...</answer>` 块。没有此格式时，则使用最后一个
不区分大小写的 `Answer:` 标记之后的文本。如果选中片段包含
`\boxed{...}` 命令，则使用该片段中第一个方框的内容。在与 `label`
进行不区分大小写的精确比较前，会规范化边缘
标点和反斜杠。

缺少答案标记、提取的答案为空、选项错误或答案
格式无效时得 0 分。选项匹配时得 1 分。该奖励不
调用 LLM、沙箱、SymPy 或外部服务，也不读取
`metadata.style` 或 `metadata.rm_type`；请通过数据行的
`"reward_fn": "rule_nemotron_mcqa"` 或 `rollout.reward_fn` 显式选择它。

<a id="code-e2b-reward"></a>

### Code E2B 奖励

`code_e2b` 支持预处理后的 LeetCode、LiveCodeBench、TACO 等代码数据。
设置数据行的 `reward_fn: code_e2b`，或用 `rollout.reward_fn: code_e2b`
作为默认值。每行的 `label` 提供完整判题请求，例如 Python 标准输入输出题：

```jsonl
{"prompt": [{"role": "user", "content": "Read two integers and print their sum. Return Python code."}], "reward_fn": "code_e2b", "label": {"language": "python", "input_mode": "stdio", "cases": [{"stdin": "2 3\n", "expected_stdout": "5\n"}, {"stdin": "-2 2\n", "expected_stdout": "0\n"}], "run_timeout": 2, "compile_timeout": 10}}
```

`label` 必需字段为 `language`、`input_mode`（`stdio` 或 `function_call`）和
非空 `cases`。可选字段包括 `fn_name`、`generated_checker`、
`compile_timeout`、`run_timeout`、`checker_timeout`，以及用于包装代码或
内嵌测试的 `code_template`。函数调用模式和代码模板仅支持 Python。

源码取最后一个 `</think>` 后的最后一个代码块；没有代码块时使用剩余文本。
`code_template` 必须包含 `__MODEL_CODE__`，以源码的 `repr(code)` 替换。
其他判题字段原样传递。
旧版 `metadata.test_cases` 或 `metadata.test_cases_path` 数据需转换为
`label`，并将 `reward_fn: leetcode` 改为 `code_e2b`。

在 rollout 进程环境中设置以下变量：

| 变量 | 含义 |
| --- | --- |
| `E2B_API_KEY` | 必需的 E2B 凭据。 |
| `E2B_TEMPLATE` | 必填且非空，指定你的 E2B 账号有权访问、已准备好的代码判题模板 ID 或别名。 |
| `E2B_DOMAIN` | 可选的 E2B 服务域名。 |
| `E2B_REQUEST_TIMEOUT` | SDK 请求超时秒数，须为正数，默认 60。 |
| `E2B_UPLOAD_REQUEST_TIMEOUT` | 上传基础超时秒数，须为正数，默认取 300 与 `E2B_REQUEST_TIMEOUT` 中的较大值。每次上传重试增加基础值的 50%，最高为基础值的 3 倍。由本地计时器约束，并传给 SDK。 |
| `E2B_COMMAND_TIMEOUT` | 可选的每批命令超时秒数，须为正数；默认根据用例数量及编译、运行和 checker 预算估算，并将首次尝试的默认沙箱存活时间限制为一小时。 |

上传使用独立的本地截止时间，SDK 传输层合并读写超时不会延长这段预算。
创建沙箱及其他 SDK 请求仍使用 `E2B_REQUEST_TIMEOUT`，判题命令保留独立超时。
每次沙箱尝试在命令预算之外预留 `2 * rpc + upload + min(rpc, 30) + 30` 秒，
其中 `rpc` 是请求超时配置，`upload` 是当前重试对应的上传超时。整个尝试的预算和沙箱存活
时间均为 `max(1200, ceil(command + reserve))` 秒：短批次获得 20 分钟预算，
较长批次保留按命令与预留时间计算的预算。每次重试都会重新开始计算尝试预算。
在默认请求和上传基础超时配置下，旧版上传分批的命令预算使首次尝试的沙箱存活时间保持在
一小时以内。上传重试会增加预留时间，判题命令的预算保持不变；重试或显式配置
更长的超时可能需要支持更长存活时间的沙箱。

需要自行准备代码判题模板，并显式设置 `E2B_TEMPLATE`。NorthRL 不包含模板构建
文件或沙箱端判题程序。模板必须提供 `/root/miniconda3/bin/python` 和
`/opt/e2b-sandboxfusion/judge.py --stdio`，并支持预处理请求及结果协议。
这些沙箱内路径是评测器的接口约定，E2B 默认基础模板无法直接满足要求。
`E2B_TEMPLATE` 缺失或仅包含空白字符时，在创建沙箱前报错；模型答案为空时，
仍直接返回零奖励，不创建沙箱。

所有用例都会参与评测。未提供共享引用时，同一回答的各批顺序执行，每批目标上限为 100 个用例
或 10 MiB。单个不可拆分的用例可以超过字节目标；每个请求和结果的硬上限
均为 1 GiB。全部用例通过时 `reward` 为 1，否则为 0。
`sandbox_result.score` 保留原始判题分数供诊断，不影响训练奖励。

上传遇到可恢复的临时故障时最多尝试五次，每次新建沙箱并上传相同内容。
四次重试采用带随机扰动的指数退避，等待范围依次为 5–10、10–20、20–40、
30–60 秒。默认上传超时在五次尝试中依次为 300、450、600、750、900 秒。
只有上传失败会递增这段超时，每个新批次都从基础超时开始。
创建沙箱和命令预热各保留两次重试；旧版分批判题命令超时重试两次，
第三次仍超时则该回答得零分。重试只涉及失败的批次，不改变判题输入和评分规则。

调用方取消、worker 停止或整个尝试的截止时间到期时，不再发起下一次尝试。
准备阶段重试耗尽、其他 E2B 服务故障、输入无效、结果漏测或协议字段矛盾时，
抛出异常而不返回奖励。批次诊断日志包含 `upload_timeout_s`，以及上传字节数、
尝试序号和当前阶段耗时。

### 代码测例的共享存储

数据行保留完整 `label.cases`，新增 `metadata.cases_ref` 后，一份候选答案
只使用一个沙箱评完整道题，不再按 100 个用例或 10 MiB 反复创建沙箱。
`metadata.cases_ref` 的值如下：

```json
{"protocol_version":1,"manifest_path":"/mnt/data/mixrl-eval/v1/manifests/<sha256>.json","manifest_sha256":"<sha256>","case_count":29}
```

数据准备和 vePFS 上传在训练前于 NorthRL 之外完成。新版 JSONL 保留原始 label
和全部 cases，非代码行保持原样。每题 manifest 记录测例文件、各测例的字节位置
和 SHA256、原判分配置及明确的资源预算。JSON 哈希使用规范化 UTF-8 编码；
单条测例的哈希包括行末换行。这些不可变文件须完成上传和校验后再用于训练。
共享卷名属于部署配置，不写死在数据中。

协议 v1 的读取程序在 NorthRL 之外维护和部署。NorthRL 不分发读取程序或发布工具，
需要将这些资源准备到你的 E2B 账号有权访问的卷中。
挂载点 `/mnt/data` 和 `/mnt/data/mixrl-eval/v1` 下包含内容哈希的路径由协议 v1
固定，它们并不代表一个可直接访问的公共共享卷。

使用已发布并校验的 `/mnt/data/mixrl-eval/v1/readers/<sha256>.py`，在训练端配置
共享卷、路径和 SHA256：

| 变量 | 含义 |
| --- | --- |
| `E2B_VEPFS_VOLUME` | 已完成校验的共享卷名称。 |
| `E2B_VEPFS_READER_PATH` | 上述包含内容哈希的读取脚本路径。 |
| `E2B_VEPFS_READER_SHA256` | 已发布脚本的 SHA256。 |

E2B 服务须支持按名称配置 `volume_mounts`，客户端使用 `e2b>=2.50.0,<2.51`。
每次只传生成的代码和引用；小请求经 base64 编码后随安全引用的启动命令传递，
较大的请求上传为小 JSON 文件。不会重复上传 `label.cases` 或读取脚本；
`code_template` 中内嵌的测试仍作为包装后代码的一部分发送。配置缺失、引用错误、
判分设置不匹配、哈希错误、结果漏测都直接报错，不静默退回其他数据。

读取程序将整题文件流式复制到沙箱私有目录，校验内容、关闭共享文件并卸载 vePFS，
随后逐个解析和评测完整测例，保持原顺序。候选程序与 checker 使用不同的非特权身份，
参考答案文件保持私有。仍由原 SandboxFusion 判分逻辑比较完整输出；新的执行层
并发读取程序输出，避免大输出堵住管道。判分后去掉回包里的重复测例原文和大输出值，
保留逐测例判定、分数、执行状态、资源信息及有界的错误摘要。stdout 和答案文件的
硬上限为 1 GiB，超出则判输出超限；stderr 持续读取但最多保留 16 MiB，
不会仅因 stderr 较多就把正确答案判错。

manifest 分别记录运行、编译、checker 的内存预算，单位为 MiB；CPU 默认总配额为 1 核。
原题明确的运行内存上限优先，缺失时使用协议 v1 固定的 1024 MiB；
编译和 checker 分别使用 1024 MiB。数据准备时须明确将原始内存单位换算为 MiB。
运行和编译等时间限制沿用原题。
超内存根据本次执行的 cgroup OOM 计数增量判断，避免历史 OOM 误伤后续测例。

整题命令预算按全部测例计算，不沿用旧流程默认一小时的截断；
`E2B_COMMAND_TIMEOUT` 仍可显式指定。超过现有 24 小时尝试上限时明确报错。
整题命令超时和可重试的执行故障共用三次执行失败预算，每次重试使用新沙箱。
预算耗尽时，如果最后一次失败是执行阶段超时，则该回答得零分，并返回
`reason: "timeout_after_retry"` 和 `timed_out: true`。缺失的测例结果仍记为未评测
（`sandbox_result.score: null`、`tests: []`），指标保留超时和重试次数。
最后一次失败为非超时故障、准备阶段重试耗尽或整个尝试的截止时间到期时，仍抛出评测异常。
沙箱清理失败会记录日志，不阻止符合条件的重试。
单个测例真实超时或超内存则判该测例失败，
后续测例继续执行。训练奖励仍然是全部通过为 1，否则为 0，平均判题分数仅供诊断。

<a id="single-turn-worker-configuration"></a>

### 单轮 Worker 配置

设置 `rollout.type: single_turn` 使用 `SingleTurnRolloutWorker`。它生成单个响应，
按上述规则选择奖励，并构建训练轨迹。Python 集成可以从 `northrl.rollout_worker`
导入此类。

Worker 选项直接配置在 `rollout` 下。`rollout.num_workers` 控制 worker 并发数。
自定义 worker 使用 `rollout.custom_module_path`，默认类名为 `CustomRolloutWorker`；
若使用其他类名，请显式设置 `rollout.custom_class_name`。

SFT 通过 `algorithm.type: sft` 选择内部 worker，应省略整个 `rollout` 块。
单轮调试轨迹的类型为 `single_turn_rollout`，生成结果字段为 `rollout_result`。

<a id="recast-reward"></a>

### Recast 奖励

`recast` 是 `northrl/rewards/recast.py` 中的普通异步奖励函数。
它使用 bp-rl-judge 的 RECAST 指令遵循提示词，并直接调用配置的
评判模型。无需 bp 评判网关。

数据行可以完全省略 `label`：

```json
{"prompt": [{"role": "user", "content": "Give exactly three suggestions for learning Python, in Chinese."}], "reward_fn": "recast"}
```

要将其作为未指定奖励名称的数据行的默认奖励，请在
现有训练配方中设置以下字段：

```yaml
rollout:
  type: single_turn
  reward_fn: recast
```

在运行 NorthRL 的进程环境中配置评判器：

```bash
export LLM_JUDGE_URL=http://judge-host:8000/v1
export LLM_JUDGE_MODEL=your-served-judge-model
# Optional: LLM_JUDGE_API_KEY and LLM_JUDGE_TIMEOUT_S
```

| 变量 | 含义 |
| --- | --- |
| `LLM_JUDGE_URL` | 必填，兼容 OpenAI 的基础 URL，通常以 `/v1` 结尾。多个端点的 URL 以换行分隔；所有端点必须提供配置的模型。 |
| `LLM_JUDGE_MODEL` | 必填，服务中的模型名称。 |
| `LLM_JUDGE_API_KEY` | 可选 API 密钥；默认为 `_`。 |
| `LLM_JUDGE_TIMEOUT_S` | 可选，以秒为单位的有限正数请求超时；`recast` 默认为 180。 |

评判请求使用温度 0，最多输出 8192 个 token。
请求会发送数据集提示词和完整生成响应，包括所有
think 或 EOS 文本。聊天消息数组会转换为列表，并表示为
字符串，保留原始角色和消息顺序。它不使用
`label`、`metadata` 或经分词器渲染的 `prompt_str`，也不要求
方框答案或思考标签。

评判响应中最后一个 `[[0]]` 或 `[[1]]` token 决定奖励；
方括号内允许有空白。如果响应内容为空，则使用
评判器的 `reasoning_content`。返回的评分包含 `correct` 和
`reason`：`pass`、`fail_llm` 或 `no_verdict`。没有判定时返回 0，不重试。

请求失败或 API 响应格式错误时，总共最多尝试两次。
第一个端点随机选择，下一次尝试轮换到下一个
端点。持续失败会抛出带原始原因的 `RuntimeError`；无效
配置会在发送请求前失败。这遵循 NorthRL 的异常
传播策略。bp 的网关则将未捕获的适配器失败转换为
零奖励；这里不复现该网关策略。每个请求客户端
都会关闭，评判请求会绕过代理环境变量。

Rollout worker 在本地 NorthRL 进程中运行并读取其环境。请在当前 shell 中导出评判器变量，或在运行 `python -m northrl.main` 前手动 source 私有环境脚本。

<a id="verinstruct-reward"></a>

### VerInstruct 奖励

`verinstruct` 是
`northrl/rewards/verinstruct.py` 中的普通异步奖励函数。它通过异步 E2B SDK，在
E2B 沙箱中运行数据行的 Python 硬约束检查器，然后直接调用 LLM
检查软约束。无需 bp 评判网关。可以通过数据行的
`"reward_fn": "verinstruct"` 选择它，也可以在现有训练配方中设置默认值：

```yaml
rollout:
  type: single_turn
  reward_fn: verinstruct
```

`metadata.instruction` 和 `metadata.hard_functions` 的用法见[数据示例](../03-data-loader/data-loader.md#single-turn-reward-data)。
每个检查器定义
`check_following(instruction, response)`。硬约束指令取
`metadata.instruction` 或 `metadata.prompt_text` 中第一个真值；否则，
普通字符串使用 `task["prompt"]`，聊天消息列表使用 `""`。使用单轮 rollout worker 时，
请显式提供 `metadata.instruction`。

在两个阶段开始前，奖励函数会保留最后一个 `</think>` 之后的文本；
没有该标记时，则保留最后一个 `###Response` 之后的文本。
它会去除首尾空白，以及尾部重复的 `<|im_end|>`、
`<|endoftext|>` 和 `<|eot_id|>` token。答案内部的 token 保持不变。
硬约束与软约束检查使用同一个规范化响应。不使用
`label`。

在 NorthRL 进程环境中配置 E2B 和评判器：

```bash
export E2B_API_KEY=your-e2b-api-key
# Optional: E2B_TEMPLATE=your-python-template
export LLM_JUDGE_URL=http://judge-host:8000/v1
export LLM_JUDGE_MODEL=your-served-if-verifier-model
# Optional: LLM_JUDGE_API_KEY and LLM_JUDGE_TIMEOUT_S
```

`E2B_API_KEY` 用于托管 E2B 的身份验证。`E2B_TEMPLATE` 可用于选择
沙箱模板；否则使用 SDK 的基础模板。自定义模板
必须提供 `python3`。SDK 还支持通过 `E2B_DOMAIN` 和 `E2B_API_URL`
E2B SDK。

`hard_functions` 缺失、为 null 或为空时，会完全跳过 E2B，因此这些数据行
不需要 E2B 凭据。LLM 使用与 `recast` 相同的 `LLM_JUDGE_*` 设置，
包括默认 180 秒的请求超时。要匹配 bp 的 VerInstruct 评判器，
请配置 IF-Verifier 模型。同一进程中的所有奖励共享这些设置；
选择 `verinstruct` 不会自动选择与
`recast` 或数学 LLM 奖励不同的模型。启动前，请在本地驱动环境中导出 E2B 和评判器设置。

对于每条需要硬约束检查的数据，NorthRL 会创建一个禁用互联网
访问的新 `AsyncSandbox`，将检查执行程序写入 `/tmp/verinstruct_check.py`，并
运行 `python3 -I /tmp/verinstruct_check.py`。所有检查器在此
单个命令中顺序执行。首次返回假值或出现检查器异常时，停止评估，
并返回 `reward: 0`、`correct: false` 和 `reason: "fail_hard"`；不会发送
LLM 请求。成功要求退出码为 0，且 stdout 去除
首尾空白后等于 `True`。因此，检查器的额外输出会导致硬约束阶段失败。
其他非零退出码也会使硬约束阶段失败，但 126 和 127 除外：它们表示
Python 命令或模板不可用，会引发错误。

命令超时时间为 10 秒。沙箱创建、文件上传和命令
执行总共限时 60 秒；SDK 请求限制为创建 30 秒，
上传和命令执行各 10 秒。SDK 输出回调
将 stdout 和 stderr 的总量限制为 2,000,000 个 UTF-8 字节。沙箱
创建后，NorthRL 会在完成后尝试将其关闭，清理操作使用独立的 10 秒
截止时间和请求超时。清理失败会记录到日志，不会替换评分或原始
错误；沙箱 60 秒的生命周期提供兜底清理。创建、上传或执行阶段超时时，
会先清理第一次尝试的沙箱，再新建沙箱重试一次硬约束检查。两次尝试
使用相同的检查器源码，各自独立计算时间和输出限制。重试成功后按正常的
硬约束、软约束规则评分；第二次仍超时则返回 `reward: 0`、`correct: false`
和 `reason: "timeout_after_retry"`，不会发送 LLM 请求。清理阶段超时不会
触发重试；SDK 也可能在内部重试传输请求。IfVerify 的沙箱检查采用相同策略。

执行程序设置以下限制：CPU 时间 10 秒、地址空间 1 GiB、文件大小 64 MiB，
进程数 64，并禁用 Python socket 创建。E2B 提供沙箱
隔离，并在创建沙箱时禁用网络访问。

硬约束检查通过后，软约束阶段复用普通的 `recast_reward`
函数，传入原始数据集提示词和规范化响应。评判回复中最后一个
`[[0]]` 或 `[[1]]` 决定二元奖励。结果的
`reason` 为 `pass`、`fail_soft` 或 `no_verdict`；没有判定时返回零。软约束
请求遵循 [Recast 的重试和错误规则](#recast-reward)。

无效的数据或配置、非超时 E2B API 错误、超过输出限制，以及持续的
LLM 请求失败仍会引发错误。任务取消也会向上传播，不触发奖励层重试。
上述重试策略只将连续两次沙箱超时转换为零奖励。

<a id="rollout-worker-types"></a>

## Rollout Worker 类型

<a id="simplerolloutworker-type-simple"></a>

### SimpleRolloutWorker（`type: "simple"`）

基本的 LLM 补全 worker。


<a id="singleturnrolloutworker-type-single_turn"></a>

### SingleTurnRolloutWorker（`type: "single_turn"`）

负责单响应生成、奖励评估和轨迹构建的 worker。

```yaml
rollout:
  num_workers: 128
  reward_fn: math_with_format
  sampling:
    temperature: 0.7
  type: single_turn
```

<a id="e2brolloutworker-type-e2b"></a>

### E2BRolloutWorker（`type: "e2b"`）

在 E2B 沙箱中执行任务定义的智能体与评估钩子。入口、输入字段和执行预算
通过 `rollout.execution` 配置。

<a id="internal-sftrolloutworker"></a>

### 内部 SFTRolloutWorker

用于监督微调的透传 worker。它将 `SFTStreamingDataLoader` 准备好的
独立 Datum 转换为 Trajectory，无需 LLM
推理；Weaver 在训练时进行打包。

```yaml
algorithm:
  type: sft
```

此 worker 是内部桥接组件，不是公开的 rollout worker 类型。SFT 应省略
整个顶层 `rollout` 配置块。控制器始终仅创建
一个 `SFTRolloutWorker`；如果提供了该配置块，NorthRL 会发出 `UserWarning`
并将其整体忽略。`algorithm.loss.mode` 可以省略，系统会将其推导为
`cross_entropy`。

<a id="configuration-options"></a>

## 配置选项

<a id="type"></a>

### type

**类型：** `string`
**可选值：** `"single_turn"`、`"simple"`、`"e2b"`、`"mock"`
**默认值：** 必填

Rollout worker 实现。

- `simple`：基本 LLM 补全
- `single_turn`：生成单个响应并使用所选奖励函数评分
- `e2b`：在 E2B 沙箱中执行远程 rollout 入口
- `mock`：用于测试的模拟 worker

<a id="num_workers"></a>

### num_workers

**类型：** `int`
**默认值：** 必填

Rollout worker 实例数量。

**指导建议：**
- CPU 受限：4–16 个 worker
- I/O 受限（LLM API）：32–256 个 worker
- Worker 是本地协程；并发数应符合驱动和远端服务的容量限制
- 考虑数据吞吐量需求

**示例：**
```yaml
# Small scale
num_workers: 8

# Medium scale
num_workers: 64

# Large scale
num_workers: 256
```

<a id="need_llm_inference"></a>

### need_llm_inference

**类型：** `bool`
**默认值：** `true`

Worker 是否需要 LLM 推理服务。

**为 `true` 时：**
- Worker 使用 LLMServiceClient
- 需要配置推理服务
- 支持权重同步

**为 `false` 时：**
- Worker 不调用 LLM
- 用于测试或非 LLM 任务

<a id="temperature"></a>

### temperature

**类型：** `float`
**默认值：** `1.0`

LLM 生成的采样温度。

**取值：**
- `0.0`：贪心（确定性）
- `0.1-0.5`：低方差
- `0.7-1.0`：均衡（常用）
- `>1.0`：高方差

**使用场景：**
```yaml
# Deterministic (evaluation)
temperature: 0.0

# Low variance (precise tasks)
temperature: 0.3

# Balanced (general training)
temperature: 0.7

# Exploration (diverse responses)
temperature: 1.0
```

<a id="top_p"></a>

### top_p

**类型：** `float`
**默认值：** `1.0`

LLM 生成的核采样概率。Weaver 会将其与请求级
采样参数一起转发。不支持此过滤器的后端可能会忽略该项并记录警告。

```yaml
top_p: 0.95
```

<a id="top_k"></a>

### top_k

**类型：** `int`
**默认值：** `-1`

LLM 生成的 top-k 采样截断值。在后端支持时，使用 `-1`
禁用 top-k。不支持此过滤器的后端可能会忽略该项并记录警告。

```yaml
top_k: 50
```

<a id="custom-worker-options"></a>

## 自定义 Worker 选项

<a id="custom_module_path"></a>

### custom_module_path

**类型：** `string`
**默认值：** None（可选）

自定义 worker 实现的路径，配置为 `rollout.custom_module_path`。

**格式：** 相对于配方目录的路径或绝对路径

**示例：**
```yaml
custom_module_path: "agent_workspace/my_worker.py"
```

**自定义 Worker：**

继承 `BaseRolloutWorker` 并实现 `async rollout(task)`，使用 `_put_trajectory()`
提交完整轨迹并返回结果。详见[自定义 Worker](../05-rollout-workers/custom-workers.md)。

<a id="custom_class_name"></a>

### custom_class_name

**类型：** `string`
**默认值：** `"CustomRolloutWorker"`

自定义 worker 模块中的类名，配置为 `rollout.custom_class_name`。

**示例：**
```yaml
custom_class_name: "MyWorker"
```

<a id="worker-parameters"></a>

## Worker 参数

<a id="execution"></a>

### execution

**类型：** `dict`
**默认值：** `{}`

`rollout.execution` 保存 E2B 执行选项。E2B worker 从这里读取远程入口、输入字段选择和超时设置。

```yaml
rollout:
  type: e2b
  num_workers: 4
  execution:
    input_fields: [instruction, interface]
    agent_timeout_seconds: 3600
    gateway_publication_timeout_seconds: 300
    token_prefix_reuse: false
    env_error_retries: 3
    volume_mounts:  # 可选的共享卷
      /mnt/shared: shared-data
    entrypoint:
      target: /path/to/NorthRL/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py
      task_files_root: /data/swe/evaluation
```

`execution.volume_mounts` 将沙箱内规范化的绝对路径映射到非空卷名，挂载路径不能为 `/`。
可在全局 `execution` 或执行 profile 中设置。profile 省略该字段时继承全局映射；
显式提供映射时替换整张映射表，`{}` 或 `null` 则清空挂载配置。

卷在每次创建沙箱时挂载，环境故障重试时使用同一映射。挂载的数据共享且可写；
NorthRL 不会自动复制到沙箱本地存储或卸载，可由任务的 `prepare` hook 按需处理。
未配置该选项时保留原有 SDK 调用，不传入卷挂载参数。

可选 agent 工具的环境变量通过 driver 上的 `NORTHRL_E2B_AGENT_ENV` 选择，
值为逗号分隔的宿主环境变量**名称**，例如 `SERPER_API_KEY,SERPER_API_URL`。
只有选中且已定义的变量值会传给以 root 运行的 agent 阶段命令（包括准备 hook），
不会传给 bootstrap 或评估阶段，也不会序列化到 task/config 数据或 trace 文件中。
任务 hook 决定哪些变量继续传给非 root agent。选择器未设置或为空时不额外转发变量；
非法名称、重复名称，以及保留的 Python、运行时和模型网关变量会在创建沙箱前被拒绝。
评估器专用设置仍使用独立的 `NORTHRL_E2B_EVALUATOR_ENV`。
参考 [JobBench recipe](../../../../recipe/jobbench/README.md) 及其 `env.sh`
配置参考资料挂载和 Serper；真实密钥应保存在私有宿主环境设置中，不写入 YAML 或数据集字段。

宿主使用 Weaver SDK 管理当前 sampler，等待 NorthGate 的 `gateway.state` 为
`published` 后，将返回的 `gateway.base_url` 和 `gateway.model` 下发到 sandbox，
无须手动配置网关地址或模型名。`gateway_publication_timeout_seconds` 限制这段
集中等待，默认 300 秒。首次启动在 sampler 预热后确认发布再启动 workers；
权重同步在新 sampler 发布完成后才切换 client 和版本并放行下一批。同一 session
的发布结果持续复用，普通 rollout 不再定期查询发布状态。发布失败保留旧快照，
并清理未激活的候选 sampler。

成功缓存不会自动探测空闲退役。恢复时需由宿主集中调用
`warmup_sampling(refresh_gateway=True)`，通过 Weaver 短采样唤醒后刷新发布结果；
也可以直接采样或计算 logprobs 后调用 `refresh_gateway_snapshot()`，或创建新 sampler。
重试 NorthGate、查询 sampler 和 heartbeat 均不能唤醒。发布检查对
`retiring` / `retired` 明确报错，不自动采样；已过期或终止的 sampler 必须重新创建。

沙箱 relay 内部固定管理 NorthGate 重试：每次请求最多尝试 3 次，从请求开始计算
300 秒失败恢复窗口，用于读取错误响应和退避等待；窗口结束后不再发起重试。
连接超时为 120 秒，且不能超过恢复窗口的剩余时间。收到响应头前的暂时性传输故障、
HTTP 429 和 HTTP 5xx 使用以 1、2 秒为基值的随机抖动退避，并遵守 `Retry-After`；
HTTP 404 直接失败。`Retry-After` 超过剩余窗口时直接失败，不提前重试。
等待成功响应头、成功响应正文和 SSE 生成仍受原有请求与 agent 执行预算约束，
不会因恢复窗口结束被截断。成功响应的正文或流读取失败后不会重放请求。
日志保留真实 HTTP 状态、SSE 错误码/类型和 request ID。

宿主配置 `WEAVER_API_KEY` 和 `NORTHGATE_API_KEY`。HTTP relay 转发 OpenAI 和 Anthropic 请求，
直接记录回包的 `token_data`。agent 继续使用本地 API 地址及 `model: northrl`，
由 relay 选择已发布的 sampler。recipe 显式配置 `rollout.sampling.max_tokens`
（验证时使用 `validate.sampling`）时覆盖 agent 预算，不按剩余上下文裁剪；
未配置或为 `null` 时保留 agent 预算。NorthGate 检查上下文限制。

E2B 不使用 `execution.tokenizer`、`execution.chat_template` 或 `rollout.parsers`。
在服务端 NorthGate codec profile 中配置模型 tokenizer、chat template 和输出
parser，并启用 sampler 自动发布。宿主训练侧的 `models.actor.tokenizer` 仍可使用。

`execution.token_prefix_reuse` 只接受布尔值，默认 `false`，支持 OpenAI Chat
Completions 和 Anthropic Messages 请求（JSON 或 SSE 响应）；relay 不提供 Responses
接口。开启前须确认 NorthGate 已支持所用接口的 `history_tokens`；协议约定见
[NorthGate：复用历史 token](https://northgate.xiaobei.top/docs/backend/features/weaver-sample.html#复用历史-token)。
execution profile 可覆盖公共设置；训练和验证 rollout 使用同一个开关。
配置名仍为 `token_prefix_reuse`，relay 通过 `history_tokens` 复用缓存的历史 tokens。

在匹配或转发 Anthropic 请求前，relay 会移除请求顶层、system/message 的直接
内容块和工具定义外层的 `cache_control`；未开启 `token_prefix_reuse` 时也执行清理。
因此缓存断点在轮次间移动不会使原本相同的历史匹配失败。这些显式 prompt-cache
指令不会转发给 NorthGate，`history_tokens` 复用独立工作。工具 input、input_schema
及其他嵌套业务数据中的同名字段会保留，OpenAI 请求不受影响。

开启 token 复用后，Anthropic 历史比较还允许省略工具调用前的纯换行 text 块。
SGLang 的 `qwen3` reasoning 和 `qwen3_coder` tool parser 会把 `</think>` 与
`<tool_call>` 之间的 `"\n\n"` 保留为 content，而 Claude Code 可能在下一轮请求中
省略对应的 text 块。NorthRL 仅在写入和查询缓存时从比较用副本中移除这些块；不修改
SGLang 或 `llm-t2t-go` 的解析行为、转发的 messages、JSON/SSE 响应，以及录制的
messages、tokens、logprobs 和 mask。复用的仍是包含原始换行 token 的完整历史，
无需新增配置项。

此例外只适用于含 `tool_use` 的 Anthropic assistant content 列表，且要求其中
所有 text 块均位于第一个 `tool_use` 之前、字段只有 `type` 和 `text`、正文非空且
仅含 `\r`/`\n`；满足条件时整体移除这些 text 块。只要存在真实正文、空格/tab、
空字符串、额外块字段或工具调用之后的 text，整个 content 列表仍按原样精确比较。
thinking、工具 input、嵌套内容和 OpenAI messages 继续精确比较。若归一化后相同的
历史对应不同的原始 tokens，缓存仍会标记歧义，不复用该历史，也不回退到更短的匹配。

relay 保存完整成功轮次的 messages 和实际 prompt/response tokens。下一轮请求前，
比较接口、历史 messages 前缀和 system、tools、thinking 等请求设置；Anthropic 的
version、beta 请求头也参与匹配。匹配且原始 tokens 无歧义时，发送
顶层字段 `history_tokens: [...]`，值为上一轮实际 prompt tokens 与 response tokens
顺序拼接后的完整数组，否则省略该字段。完整 messages 仍随请求发送，不发送
`message_count`。只有成功录制且以 OpenAI
`stop`/`tool_calls` 或 Anthropic `end_turn`/`tool_use` 结束的响应可供复用；流式响应
还须完整结束并完整重建 assistant 消息。
OpenAI 请求若配置了非空的 `stop`、`sampling_params.stop` 或
`sampling_params.stop_token_ids`，响应不写入复用缓存，因为完成原因不能证明模型输出了
所需结束符；Anthropic 的 `stop_sequence` 完成原因也不可复用。

历史数组必须非空、各 ID 非负，最后一个 ID 须为上一轮模型停止生成时输出的结束符。
未以结束符结尾的截断输出不能复用；一次 rollout 内须固定模型、tokenizer 和模板配置。

NorthGate 先渲染完整请求，统计历史数组最后一个 ID 在 `history_tokens` 中出现的次数，
再从渲染结果中同样数量的结束符之后切开。切点之后的新消息与生成前缀编码后，接在
原样保留的历史 tokens 之后。模板须保证同一历史在两轮渲染中的结束符个数一致。
NorthGate 不保存会话历史，消息与 tokens 的对应关系由客户端保证。历史不满足约束，
或其中结束符多于渲染结果中的结束符时，服务端在采样前返回 `400 invalid_request`。

relay 会拒绝未保留所传前缀的实际 prompt tokens；recorder 继续记录真实返回的
token 数据，沿用严格 token 前缀检查。详见
[E2B 任务接入](e2b-advanced.md#token-prefix-reuse)。

未启用 profiles 时，`rollout.execution.input_fields` 是 E2B 的必填项，类型为不重复的非空字符串组成的列表。
它选择复制到 `RolloutInput.task` 的 dataloader 字段；`[]` 表示不上传题目字段。
某行中不存在的字段会跳过，由任务脚本校验必需输入并构建 prompt。SWE 选择
`[instruction, interface]`；其他任务可选择 `[question, context]` 等字段。
`task_id` 单独保存在 metadata 中供评测使用，无须选中。未选中的数据字段不会作为
任务输入上传；不要选择隐藏答案或测试字段。

`execution.result_processing` 可选 `thread`（默认）或 `process`。
选择 `process` 后，同一 driver 事件循环的并发 rollout 共享一个常驻 spawn
子进程，执行最终结果的 gzip/JSON 校验及紧凑数组恢复。完整结果及其数组通过
真实 IPC 返回 driver；trajectory pool 的归属和去重仍由 driver 负责。
该模式增加进程启动、内存和 IPC 开销，CPU 回放提速不代表整步训练的收益。
现有两个结果处理槽、下载完成后的时限边界和取消等待语义保持不变，
最后一个 rollout 退出时回收子进程。该选项可按 execution profile 覆盖，
不改变周期快照下载或部分训练 trace 的恢复路径。

在 Driver 环境中设置 `NORTHRL_PROFILE_RESULTS=1`，可启用 INFO 级别的少量结果诊断
（默认关闭）。每条 `driver.result-decode` 记录包含实际生效模式、压缩字节数、
展开后的 token 位置数（各 segment 长度之和）、segment 数，以及解析/恢复的 wall 时间和
执行线程 CPU 时间。进程模式还记录子进程 PID
及主进程往返 wall 时间。往返包含首次调用的进程启动、序列化、传输、子进程排队和调度；
减去子进程解析/恢复时间**不能**得到纯 IPC 耗时。

`driver.result-postprocess` 测量 Driver 将结果转换为 trajectory 的耗时。
`driver.result-pool-insert` 测量训练封装释放暂存 trajectory 后的真实入池操作，包含锁、
分组和去重。两者均记录 wall 时间和执行线程 CPU 时间。下载准入、读取及处理排队时间
继续由 `driver.file-download` 记录。日志只含身份、计数和耗时，不含 tensor 内容；
启用诊断不改变准入限制、取消等待或结果校验。

本地性能验证脚本 `tests/benchmarks/benchmark_result_processing.py` 使用真实的结果下载
适配器、紧凑数组 IPC、分组池以及前台 conversion/SDK builder。它报告每个 fixture
的展开规模、前台完成时间、全部后台完成时间及实际突发到达时间；一致性检查在两段计时
均结束后执行。默认混合 fixture 为合成数据；`--results-dir` 可以读取仅含已保存完整结果
gzip 文件的目录，以回放真实规模和类型。通过 `PYTHONPATH` 选择 NorthRL/SDK 源码，
报告记录源码路径及 SDK 源码版本。远程读取使用内存数据，到达时间表为合成负载，
因此该检查不代表 Server/GPU 提速，也不复现线上后台状态。

`execution.entrypoint.target` 和配置后的 `task_files_root` 必须是宿主机上的绝对路径，
不接受相对路径或 `~`。将示例中的 `/path/to/NorthRL` 替换为实际 checkout 的绝对路径。

E2B 通过 `execution.entrypoint.target` 指定宿主侧 Python 脚本，实现 `run_agent`
和 `evaluate`；评测材料在 agent 退出后上传。`execution.env_error_retries` 控制
环境故障的重试次数。一次 rollout 提交一个包含内部 segments 的完整 trajectory；
解题未通过是有效 0 分结果。详见 [E2B 任务接入](../05-rollout-workers/e2b-task-contract.md#scoring-and-failures)。

Agent 运行时长由 `rollout.execution.agent_timeout_seconds` 控制。MiMo 入口
不再读取数据中的 `agent_timeout_sec`；旧数据可以保留这个无效字段，
但不应继续将它选入 `input_fields`。预算耗尽后框架取消 Agent，
再使用独立的收尾预算清理进程；评测时限单独计算。

<a id="execution-profiles"></a>

#### 执行 Profiles

在同一 E2B worker 池中运行多类任务时，用命名的
`rollout.execution.profiles` 替换全局 `entrypoint` 和 `input_fields`：

```yaml
rollout:
  type: e2b
  execution:
    agent_timeout_seconds: 3600
    env_error_retries: 3
    profiles:
      mimo_code:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/code.py
          task_files_root: /data/mimo/evaluation/code
      mimo_arvo:
        input_fields: [instruction, cwd, agent_user, max_turns, description]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/arvo.py
      mimo_terminal:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/terminal.py
          task_files_root: /data/mimo/evaluation/terminal
      mimo_general:
        input_fields: [instruction, cwd, agent_user, max_turns]
        agent_timeout_seconds: 7200
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/mimo/general.py
          task_files_root: /data/mimo/evaluation/general
      swe:
        input_fields: [instruction, interface]
        entrypoint:
          target: /path/to/NorthRL/northrl/rollout_entrypoints/swerebench_claude_code_rollout.py
          task_files_root: /data/swe/evaluation
```

每个 profile 必须提供独立、完整的 `entrypoint` 和 `input_fields`。
全局 `execution.entrypoint` / `input_fields` 不能与 `execution.profiles`
同时配置。profile 的 `agent_timeout_seconds` 覆盖公共值；两处均省略且未配置
旧别名 `timeout_seconds` 时默认 3600 秒。profile 也可以覆盖
`preparation_timeout_seconds`、`agent_finalize_timeout_seconds`、
`evaluation_timeout_seconds`、`env_error_retries`、`result_processing`、
`token_prefix_reuse` 和旧版 `tokenizer` 选项，
省略时继承公共值。网关发布设置仍由 worker 池共用。
所有 profile 共用 actor、模型 tokenizer、rollout sampling 配置和 trainer。

profile 模式下，每条训练和验证任务都必须包含已定义的 `rollout_profile`。
单类数据源可在 [`data.train.sources`](data-config.md#mixing-rl-task-sources)
中配置，由加载器自动添加。使用 `data.train.files` 或验证文件时，在每行
数据中添加该字段。缺失或未知的 profile 名称直接报错，不回退到其他入口。

Controller 为每个 profile 独立准备执行代码和评测资源。Worker 根据任务的
`rollout_profile` 同时选择两者，再以原始 `task_id` 在对应评测根目录中查找
资源，因此不同任务类型可以使用相同 `task_id`。profile 不会继承其他
profile 的评测目录。只有选中的 `input_fields` 会作为任务输入上传到
sandbox；除非显式选中，`rollout_profile` 和 `task_id` 仅作为路由或评测
metadata。省略 `profiles` 时保留原有的单入口行为。

E2B 会在 sampler 预热之前准备资源，并在 INFO 日志中输出每个 profile
的开始、完成和耗时。打包评测资源时，在任务完成之间最多每五秒输出一次
进度，包含已完成/总任务数、压缩包字节数和耗时。

<a id="params"></a>

### params

**类型：** `dict`
**默认值：** `{}`

`rollout.params` 保存读取该字段的 custom/non-E2B worker 的专用设置，
与 `rollout.execution` 相互独立。E2B 要求 `params` 为空，其设置从 `execution` 读取。

<a id="length-constraints"></a>

## 长度约束

<a id="max_prompt_length"></a>

### max_prompt_length

**类型：** `int`
**默认值：** 继承自 `data.max_prompt_length`

提示词的最大 token 数。

**示例：**
```yaml
max_prompt_length: ${data.max_prompt_length}
```

<a id="max_response_length"></a>

### max_response_length

**类型：** `int`
**默认值：** 继承自 `data.max_response_length`

响应的最大 token 数。

**示例：**
```yaml
max_response_length: ${data.max_response_length}
```

<a id="common-patterns"></a>

## 常用模式

<a id="simple-llm-task"></a>

### 简单 LLM 任务


<a id="math-reasoning"></a>

### 数学推理

```yaml
rollout:
  num_workers: 128
  reward_fn: math_with_format
  sampling:
    temperature: 1.0
  type: single_turn
```

<a id="testing-configuration"></a>

### 测试配置


<a id="worker-scaling-guidelines"></a>

## Worker 扩展指南

<a id="development"></a>

### 开发

```yaml
rollout:
  num_workers: 8  # Small scale for debugging
```

<a id="moderate-concurrency"></a>

### 中等并发

```yaml
rollout:
  num_workers: 64  # Concurrent local rollout workers
```

<a id="higher-concurrency"></a>

### 更高并发

```yaml
rollout:
  num_workers: 256  # Requires sufficient driver and service capacity
```

<a id="temperature-guidelines"></a>

## 温度设置指南

<a id="task-based"></a>

### 按任务设置

```yaml
# Exact matching (code, math)
temperature: 0.0

# Reasoning tasks
temperature: 0.7

# Creative tasks
temperature: 1.0

# Exploration training
temperature: 1.0
```

<a id="troubleshooting"></a>

## 故障排查

<a id="workers-idle"></a>

### Worker 空闲

**现象：** 吞吐量低，worker 未处理任务

**解决方法：**
- 检查数据加载器是否有可用数据项
- 确认 LLM 服务可访问
- 查看 worker 日志中的错误
- 检查下一批数据是否正在等待权重同步或验证

<a id="high-error-rate"></a>

### 错误率高

**现象：** 大量轨迹失败

**解决方法：**
- 检查 LLM 服务状态
- 验证评估器实现
- 检查提示词格式
- 检查 max_response_length 是否足够

<a id="custom-worker-not-found"></a>

### 找不到自定义 Worker

**现象：** `ModuleNotFoundError` 或 `AttributeError`

**解决方法：**
- 确认 `custom_module_path` 正确
- 检查 `custom_class_name` 是否与类匹配
- 确保文件位于正确位置
- 确保本地依赖可在 worker 进程中导入

<a id="related-documentation"></a>

## 相关文档

- [Rollout Worker](../05-rollout-workers/overview.md) - Worker 架构
- [自定义 Worker](../05-rollout-workers/custom-workers.md) - 实现自定义 worker
- [评估器](../05-rollout-workers/evaluators.md) - 评估模式
