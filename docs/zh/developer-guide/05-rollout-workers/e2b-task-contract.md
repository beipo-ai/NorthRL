[English](../../../en/developer-guide/05-rollout-workers/e2b-task-contract.md) | 简体中文 | [文档目录](../../index.md)

# E2B 任务接口参考

本文说明任务函数、结果类型以及评测与失败语义。首次接入请先阅读[接入指南](e2b-task-integration.md)，完整示例见[最小任务示例](../../recipe/e2b_minimal/README.md)。

<a id="hooks"></a>

## 任务函数与输入

```python
def resolve_workspace(input_data: RolloutInput) -> Path: ...

async def prepare(input_data: RolloutInput, workspace: Path) -> None: ...

async def run_agent(
    input_data: RolloutInput,
    workspace: Path,
    model_environment: dict[str, str],
) -> AgentResult | None: ...

async def evaluate(
    input_data: RolloutInput,
    workspace: Path,
    agent_result: AgentResult,
    evaluation_assets: Path,
) -> EvaluationResult: ...
```

- `run_agent` 和 `evaluate` 必须提供；`prepare` 可省略。
- workspace 可由 `resolve_workspace` 返回，也可直接配置 `entrypoint.workspace`。配置的 workspace 优先于函数返回值，路径必须是 **sandbox 内已存在的绝对路径**。
- `input_data.rollout_id` 标识一次执行；worker 按 recipe 的 `rollout.execution.input_fields` 选取数据字段，放进 `input_data.task`，不自动上传完整数据行。某行中不存在的字段会跳过，由任务函数检查必需字段。`template` 用于宿主选择 sandbox，不自动进入 `task`。
- `input_data.metadata` 保存 `task_id`、`group_id`、`run_id`、`expected_rollouts`（数据中存在时）及 `weight_version`。其中 `task_id` 用于定位和校验评测材料，不需要列入 `input_fields`。metadata 不会自动加入 agent prompt。
- 任务函数自行验证 `input_data.task` 中的字段并构建 agent prompt；通用 worker 不要求固定字段名。SWE recipe 配置 `input_fields: [instruction, interface]`，任务函数要求非空 `instruction`，存在非空 `interface` 时以一个空行连接两段原文，不额外添加标题。SWE 的 Claude prompt 只使用这两个字段。
- `model_environment` 是模型 API 环境变量字典，包含 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 和 `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`。地址指向 sandbox 内 gateway，key `northrl` 是本地占位值。任务方不需要 Weaver 凭证或模型权重路径。
- `AgentResult` 是 `run_agent` 交给后续 `evaluate` 的任务输出，字段含义见下表。改好的 workspace 会直接保留，无须把代码内容塞进返回值。
- `evaluation_assets` 是**第二阶段才上传**的该题评测目录。`evaluate` 返回 `EvaluationResult(reward=..., status=..., metrics=...)`。reward 必须是有限浮点数，metrics 是数值字典。

所有这些类型均从 `northrl.remote_rollout.contracts` 导入。`input_data` 是 `RolloutInput` 对象，包含 `rollout_id`、`task` 和 `metadata`，不是单独的指令字符串。协议中的 `task` 仍是通用 JSON 字典，训练 worker 按上述字段范围构造它。任务脚本被平台加载，模块顶层不得自行运行 agent、启动 runtime 或安装隐藏测试。任务脚本本身在第一阶段可见，秘密评分规则应放进后上传的评测包。

未列入 `input_fields` 的数据字段不进入 `input_data.task`。SWE 不选择 `patch`、`test_patch`、`pr_description`、`install_config` 和 `meta`。其他任务也应只选择解题输入，隐藏答案和测试通过后续评测包上传。runtime 在加载请求时仍核验输入文件属主并设置 `0600` 权限；agent 应以任务配置的受限系统用户运行，不能读取该文件。

<a id="agent-result"></a>

## AgentResult

agent 结束并完成清理后，平台把 `AgentResult` 保存到阶段记录，再交给同一个 sandbox 中另一个进程执行的 `evaluate`。模型轨迹、token 和 log probabilities 由 runtime 单独记录。

| 字段 | 默认值 | 含义与使用方式 |
| --- | --- | --- |
| `exit_code` | `0` | agent 执行状态，仅用于诊断。无论退出码为 0、非零或超时返回的 124，清理完成后都会调用 `evaluate`，由评测决定 reward。平台不要求 Claude 输出最终 `success` 标记。 |
| `final_answer` | `None` | 任务自行提供的最终回答文本。例如问答任务返回生成的答案，由 `evaluate` 与隐藏答案比较。平台不会自动从模型对话中提取此字段，也不会自动给它评分。只需要检查 workspace 的任务可以省略。 |
| `artifacts` | `{}` | 产物名称到 workspace 相对路径的映射，例如 `{"patch": "fix.patch", "report": "output/report.json"}`。它只保存文件引用，不保存文件内容；评测函数按任务约定解析路径并读取文件。平台不会据此自动复制、上传或下载文件，也不校验路径是否为相对路径或文件是否存在。 |

例如，agent 已在 workspace 写好 `output/answer.txt` 后，可以返回：

```python
return AgentResult(
    exit_code=0,
    final_answer="42",
    artifacts={"answer": "output/answer.txt"},
)
```

评测函数可以读取 `agent_result.final_answer`，也可以读取 `workspace / agent_result.artifacts["answer"]`；由任务决定使用哪种产物。SWE 任务直接评测改过的代码仓库，`AgentResult` 只需携带实际退出码。`run_agent` 返回 `None` 等价于默认的 `AgentResult()`；最终 reward 由 `evaluate` 返回的 `EvaluationResult` 决定。

`model_config = _MODEL_CONFIG` 是共享的 Pydantic 校验设置，不是任务输出字段：使用严格类型并拒绝未声明字段。`artifacts` 的 `default_factory=dict` 会为每个结果创建独立的空字典。

<a id="evaluation-result"></a>

## EvaluationResult

由任务的 `evaluate` 返回，表示对 agent 产物的最终评分。runtime 把它与 agent 阶段的轨迹合并成 `RemoteRolloutResult`，再交给宿主 worker；任务无需自行构造训练轨迹。

| 字段 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `reward` | 必填 | 这次完整 rollout 的最终分数，必须为有限数值；不固定在 `[0, 1]`，分值范围由任务定义。多个 segment 共用这个父结果评分，不分别重复计算 reward。 |
| `metrics` | `{}` | 附加评测指标，键为字符串、值为有限数值，例如 `{"swebench_resolved": 1.0}` 或 `{"tests_passed": 8.0}`。供结果记录和观测使用，框架不会自动把这些值合成为 reward。 |
| `status` | `"success"` | 只能是 `success` 或 `task_failed`。`success` 允许任意有限 reward，包括 0，不强制表示题目答对；`task_failed` 表示任务未通过，强制要求 `reward=0.0`。两者均是有效评测结果，训练侧均使用成功父轨迹，按 main 的完整组规则参与后续处理。 |

例如，SWE 任务根据测试结果返回：

```python
return EvaluationResult(
    reward=1.0 if passed else 0.0,
    status="success" if passed else "task_failed",
    metrics={"swebench_resolved": float(passed)},
)
```

评测器崩溃、隐藏材料缺失等故障应抛出 `RolloutEnvironmentError`，由框架分类为 `env_error`；`EvaluationResult.status` 不接受 `env_error`，不能用正常的零分结果掩盖评测故障。`model_config` 复用严格类型、拒绝额外字段和拒绝 NaN/Infinity 的公共配置。

<a id="agent-stage-result"></a>

## AgentStageResult

这是框架生成的跨进程交接记录，任务代码不用构造或返回它。agent 清理完成、本地 relay 关闭后，平台写入 `agent-stage.json.gz`；executor 验证身份后才上传隐藏评测材料，再启动 evaluate 进程读取这份记录。它只说明 agent 阶段完成，尚不包含最终评分。

| 字段 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `rollout_id` | 必填 | 逻辑 rollout 的非空 ID，来自 `RolloutInput`；环境重试时沿用。executor 和 evaluate 进程都会核对它是否属于当前任务执行。 |
| `attempt_id` | 必填 | 本次执行尝试的非空 ID，由 executor 每次重新生成。即使 `rollout_id` 相同，重试的 `attempt_id` 也不同，用来排除旧尝试留下的阶段记录。 |
| `phase` | `"agent_completed"` | 唯一允许的值，标记 agent 阶段已经提交。它不是动态进度字段，也不表示评测已经完成。 |
| `workspace` | 必填 | sandbox 内 agent 使用的 workspace 绝对路径字符串。evaluate 沿用此目录；runtime 另外检查目录存在，并与 agent 阶段路径一致。 |
| `agent_result` | 必填 | 任务 `run_agent` 的输出，即上面的 `AgentResult`；hook 返回 `None` 时由框架填入默认结果，交给 evaluate 使用。 |
| `trace` | `None` | agent 阶段记录的 `RemoteRolloutTrace`，包含 segments、模型对话及来源信息；有 trace 时其 `rollout_id` 必须一致。没有录到任何模型轮次时为空，例如 agent 在首个模型请求完成前退出，或首个请求就超长。 |
| `metrics` | `{}` | 没有 trace 时保留 agent 阶段运行指标；有 trace 时指标放在 `trace.metrics`，此字段为空。不是任务评测分数。 |
| `termination_reason` | `None` | 普通完成为 `None`；另一个允许值是 `context_length_exceeded`，表示因上下文预算耗尽停止生成，但仍可评测 workspace。它不表示题目通过或失败。 |
| `length_details` | `{}` | 上下文停止时的 token 预算详情：`prompt_tokens` 是该请求的输入 token 数，`requested_max_tokens` 是请求的最大响应 token 数，`max_sequence_length` 是适用的总上下文上限。普通完成时为空。模型仅校验字符串键和整数值，不强制必须有这三个键。 |

阶段记录不限制 `agent_result.exit_code`，退出码只用于诊断（例如 agent 超时为 124）。`model_config` 沿用严格公共配置。最终任务分数来自下一阶段返回的 `EvaluationResult`，不会由 `phase`、退出码或上下文停止原因推导。

<a id="process-cleanup"></a>

## 进程清理与跨阶段数据

如果运行 Claude Code 或其他子进程，`run_agent` 必须在正常返回、异常和取消时都清理 agent 及其子进程，等待它们真正退出。不可遗留后台 agent 进程在评测材料上传后继续读取材料或修改产物。SWE 适配器已有专用 agent 用户及清理逻辑，可参考其处理方式。

任务方不写 `agent-stage.json.gz` 或最终 `result.json.gz`，这些都由平台统一生成和校验。`--phase agent` / `--phase evaluate` 是 executor 调用 `e2b_runtime.py` 的平台命令行参数；任务脚本不需要实现 CLI 或 `main()`，只实现上面的 hooks。agent 和 evaluate 在同一个 sandbox 的不同 Python 进程中运行，跨阶段数据通过 workspace 文件和 `AgentResult` 传递，不依赖模块全局变量。

`profile_block` 是可选计时工具，可以继续使用 `profile_block("task.stage", rollout_id=input_data.rollout_id)` 关联日志，不是任务必须提供的接口。

<a id="hidden-assets"></a>

## 隐藏评测材料与任务文件

`entrypoint.target` 必须是宿主上 `.py` 脚本的绝对路径。不接受相对路径或 `~`，也不会自动补全项目根目录。平台上传并导入该脚本后调用任务函数，不执行它的 `__main__`。`northrl` 包内的公共 Python 代码会随平台上传；外部脚本只自动上传该脚本，不自动打包它旁边的辅助文件。其他公开工具、数据和依赖需预装到模板中。

`task_files_root/<task_id>/` 会递归打包，保留目录结构。这里可以放任何该任务需要的评测文件，不限于 SWE 的固定四个文件；评测目录不能包含 symlink。`task_files_root` 配置后必须是宿主机上的绝对路径，可指向当前 checkout 以外的目录；不接受相对路径或 `~`。评测函数按自己的格式读取传入目录。

任务数据必须包含非空的 `template`，宿主用它创建 sandbox。`input_data.task` 仅包含 `input_fields` 选中的现有字段，具体要求由任务函数校验。配置 `task_files_root` 时还必须提供非空字符串 `task_id`，worker 将其保存在 `metadata.task_id`，以定位对应评测包；不需要外部评测材料时可以省略 `task_files_root`，平台仍会调用 `evaluate`，传入空评测目录。同类任务新增题目通常只增加数据、模板或逐题评测材料，无须再复制一份任务模块。

agent 可见的模板文件、任务模块、初始代码包和题目输入中不能预放隐藏测试、答案或秘密评分规则，压缩包中的副本也不例外。评测函数所需的隐藏材料应放在评测包中，不通过任务输入传递。公开测试可以提前提供。平台延后上传不能删除模板里已经存在的秘密副本，接入时应检查模板内容是否符合预期。

<a id="scoring-and-failures"></a>

## 评分与失败

- 解题未通过可正常返回 0 分。`status="task_failed"` 必须配 `reward=0.0`；`success` 允许连续分数，包括 0，具体含义由任务声明。两者在训练侧都是有效的 `Trajectory.status="success"`，原始任务状态保存在 `e2b_status`。
- 评测程序崩溃、资产缺失或环境不可用，应抛出 `RolloutEnvironmentError`，不能当成任务正确执行后的 0 分。评分脚本用非零退出码表示“不通过”还是“执行故障”，必须由任务的 `evaluate` 明确解析。
- `run_agent` 返回任何 `exit_code`，或抛出普通 agent 执行异常时，平台在清理完成后发布阶段记录并调用 `evaluate`，保留实际评测分数和已录制轨迹。退出码记录在 `agent_exit_code` 指标中，不预先把 reward 置零；没有模型轨迹也允许评测现有产物。
- 显式 `RolloutEnvironmentError`、模型服务故障、进程清理失败或 sandbox 总时限耗尽仍按环境／执行故障处理。平台必须确认 agent 阶段已结束后，才能上传隐藏评测材料。
- 平台保留有效的部分轨迹并分类处理失败；评测上传和运行不改变第一阶段的模型版本。重试由 executor 管理。`env_error_retries` 仅允许在 agent 命令下发前重建 sandbox；命令下发后优先恢复原尝试，不自动重新运行 agent。详见[执行时限与恢复](../11-configuration-reference/e2b-execution-budgets.md)。
- worker 每次只提交一个完整 `Trajectory`，其内部 `segments` 保存全部训练序列，reward 和 score 属于父结果。发生序列截断时会清零该 rollout 的训练 reward 和 `swebench_resolved`。
- 环境或评测故障最终成为 `Trajectory.status="failed"`、`reward=None` 的空序列结果，由 main 的完整组规则处理；不能伪装成有效的 0 分。有效的零序列结果仍保留最终 reward 并参与组内比较。
- remote trace/result 使用 `segments` 序列化。统计按完整 rollout、内部序列、token 分别计数，见[轨迹分组](../04-trajectory-pool/trajectory-pool.md)与[训练指标](../../metrics.md)。

Agent 退出时取消尚未结束的模型请求，不会单独导致评测失败；判断方式见[请求取消](e2b-runtime.md#request-cancellation)。

<a id="integration-checks"></a>

## 接入检查

接入验收至少覆盖：任务输入只包含 `input_fields` 选中的字段，任务脚本正确校验字段并组装 prompt；`task_id` 位于 metadata，SWE prompt 不包含它；第一阶段 agent 看不到隐藏资产和请求输入文件；agent 清理并保存 trace 后才上传评测包；评分正反例正确；评测器故障不伪装成 0 分；一次真实 rollout 产生有效训练轨迹；最后完成小批训练。
