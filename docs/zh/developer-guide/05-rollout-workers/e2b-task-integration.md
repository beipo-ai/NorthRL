[English](../../../en/developer-guide/05-rollout-workers/e2b-task-integration.md) | 简体中文 | [文档索引](../../index.md)

<a id="e2b-task-integration"></a>

# 接入 Agent 任务

通过 E2B 任务接口，你可以根据任务结果训练 Agent 使用的模型。
你提供任务输入、Agent 执行逻辑和评分规则；NorthRL 在沙箱中运行任务，
通过 NorthGate 记录模型交互，再将轨迹和奖励送入训练流程。

第一次接入可以先按照[最小任务示例](../../recipe/e2b_minimal/README.md)运行。
示例提供数据准备、完整训练配置和本地启动命令。
本指南介绍如何将这套流程用于自己的任务。

<a id="tasks-and-rewards"></a>

## 1. 准备任务输入和评测材料

先确定 Agent 接收什么输入、需要产出什么，以及如何评分。
任务可以要求返回答案、修改代码仓库或生成文件。
由你定义哪些结果对应哪些奖励。

每条数据都需要 `template` 字段，指定你有权限使用的 E2B 模板。
下面是最小任务使用的数据结构：

```json
{"instruction": "Reply with exactly: READY", "template": "YOUR_E2B_TEMPLATE_ID", "task_id": "task-001"}
```

- `rollout.execution.input_fields` 是必填列表，元素必须是非空且不重复的字段名字符串。
  它选择传入 `input_data.task` 的字段；`[]` 表示不传入任何字段。
  本例使用 `[instruction]`。
- `template` 用于选择沙箱模板。`task_id` 也会放入
  `input_data.metadata["task_id"]`，二者无需加入 `input_fields`。
- 数据中不存在的已选字段会被跳过。构造 Agent prompt 前，
  任务代码应检查所需字段是否齐全、有效。
- 将 `data.train.prompt_key` 设置为数据加载器使用的输入字段，本例为
  `instruction`。数据文件须采用[支持的格式](../03-data-loader/data-loader.md)。

如果评测需要隐藏答案或测试，将其保存在运行 NorthRL 的机器上，
目录结构为 `task_files_root/<task_id>/`。例如：

```text
/data/my_task/evaluation/
└── task-001/
    └── expected.txt
```

对于上面的样本，`expected.txt` 的内容为 `READY`。`task_id` 应是单个
目录名，例如 `task-001`；评测目录中不能包含符号链接。
隐藏材料不要放入题目输入、Agent 可见的模板或任务模块中。
NorthRL 只在 Agent 执行和清理结束后上传这些材料。
公开测试可以在执行阶段提供给 Agent。

如果评测不需要外部材料，可以省略 `task_files_root`；框架仍会调用
`evaluate`，并传入一个空评测目录。

<a id="task-hooks"></a>

## 2. 实现执行与评分接口

将任务函数放在一个 Python 模块中。必须实现 `run_agent` 和 `evaluate`，
需要准备任务环境时再添加 `prepare`。接口如下：

```python
from pathlib import Path

from northrl.remote_rollout.contracts import AgentResult, EvaluationResult, RolloutInput


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

以上仅展示接口签名。编写函数体时，可以参考
[最小任务实现](../../../../recipe/e2b_minimal/rollout.py)。

| 接口 | 你需要实现的逻辑 |
| --- | --- |
| `prepare` | 在 Agent 执行前准备任务文件或依赖。 |
| `run_agent` | 读取任务输入，使用 `model_environment` 运行 Agent，通过 `AgentResult` 返回答案或产物引用。 |
| `evaluate` | 读取答案或修改后的工作区，按需使用 `evaluation_assets`，返回 `EvaluationResult`。 |

`evaluate` 在同一个沙箱的另一个进程中执行。跨阶段数据应通过
`AgentResult` 或工作区文件传递，不要依赖模块全局变量。
框架不会自动从模型对话中提取 `final_answer`，也不会按进程退出码评分。
按工作区评测的任务可以只返回 `AgentResult(exit_code=...)`，不填写答案。

奖励必须是有限数值。`status="success"` 允许任意有限分数，包括零；
`status="task_failed"` 要求 `reward=0.0`。评测材料缺失、评测程序故障
或环境不可用时，应抛出 `RolloutEnvironmentError`，不要记为普通零分结果。

如果 Agent 以子进程运行，`run_agent` 必须在正常结束、异常和取消时，
停止并等待 Agent 及其子进程退出。评测材料会在清理之后上传，
评测期间不应还有后台 Agent 在运行。
平台导入任务模块后调用这些函数，不会执行任务的 `main()`。
完整字段和状态说明见[任务接口参考](e2b-task-contract.md)。

<a id="sandbox-template"></a>

## 3. 准备 E2B 模板和工作区

选择你的 E2B 凭据有权访问的 [E2B](https://e2b.dev/) 模板，
在其中准备 Agent 所需的工具、任务依赖和公开文件。
将已存在、可写的绝对路径（例如 `/workspace`）配置为
`entrypoint.workspace`。框架会先解析工作区，再调用 `prepare`，
因此应在模板中创建该目录，不要等到 hook 执行时再创建。

需要共享工具或数据时，可使用 [`execution.volume_mounts`](../11-configuration-reference/rollout-config.md#execution)。
挂载的数据共享且可写，是否复制到沙箱本地由任务决定。

注意区分这三类位置：

| 配置 | 位置 |
| --- | --- |
| `entrypoint.target` | 运行 NorthRL 的机器上的任务 `.py` 模块绝对路径。 |
| `entrypoint.task_files_root` | 同一台机器上的评测材料绝对路径。 |
| `entrypoint.workspace` | 沙箱内部已经存在的目录。 |

宿主机路径不接受 `~` 或相对路径。NorthRL 会上传任务模块和公共运行代码，
但不会自动打包外部任务模块旁边的其他文件。
辅助代码和 Agent 依赖应准备在模板中。
以子进程运行的 Agent 应使用任务配置的受限用户，
使其无法读取平台私有的请求文件和凭据文件。

<a id="model-services"></a>

## 4. 连接模型服务

选择你有权限训练的 Weaver 模型，在启动 NorthRL 的 shell 中
导出以下环境变量：

| 环境变量 | 用途 |
| --- | --- |
| `WEAVER_BASE_URL` | Weaver 服务地址。 |
| `WEAVER_API_KEY` | 模型训练与采样凭据。 |
| `NORTHGATE_API_KEY` | Agent 模型调用所用网关的访问凭据。 |
| `E2B_API_KEY` | 使用所选模板创建沙箱的凭据。 |

使用私有 E2B 服务时，按服务方提供的信息配置 SDK 地址。
Weaver 和 NorthGate 必须支持发布所选 sampler，并返回训练 token 数据；
开始运行前应确认服务已具备这些能力。

在 `run_agent` 中，使用传入的 `model_environment` 调用模型：

| API | 平台提供的环境变量 |
| --- | --- |
| OpenAI Chat Completions | `OPENAI_BASE_URL`、`OPENAI_API_KEY` |
| Anthropic Messages | `ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY` |

两种 API 均支持普通响应和流式响应；当前任务接口不支持 OpenAI Responses API。
将平台提供的值传给模型客户端或 Agent 子进程，并使用 `model: northrl`。
NorthRL 会将该名称路由到所选 sampler，记录实际 token 和对数概率；
任务代码不需要构造训练 token 数组，也不需要填写远端模型 URL。

E2B 不使用 `execution.tokenizer`、`execution.chat_template` 或 `rollout.parsers`，
分词、对话模板和响应解析由 NorthGate 处理。宿主侧的 `models.actor.tokenizer`
仍可用于数据准备和训练。

<a id="training-recipe"></a>

## 5. 配置配方并在本地启动

从[完整的最小训练配方](../../recipe/e2b_minimal/README.md#training-config)开始，
替换任务数据、模型和评测设置。下面的片段展示需要合并到完整配方中的任务接入字段：

```yaml
backend:
  type: weaver
data:
  train:
    prompt_key: instruction
training:
  weight_sync:
    mode: sync
rollout:
  type: e2b
  execution:
    input_fields: [instruction]
    agent_timeout_seconds: 3600
    entrypoint:
      target: /path/to/NorthRL/recipe/my_task/rollout.py
      workspace: /workspace
      task_files_root: /data/my_task/evaluation
```

将宿主机绝对路径替换为实际位置。这个片段省略了模型、数据文件、优化器
及其他训练参数，请保留完整配方中的相应设置，并按自己的任务调整。
如果启用了验证数据集，也要准备其任务接口所需的字段和评测材料。

安装 `.[core]` 依赖后，按示例的[本地启动步骤](../../recipe/e2b_minimal/README.md#run)
运行。`TRAIN_CONFIG` 和 Hydra 配置参数需要指向同一份完整配方。
直接本地运行不会自动加载 `.env`，请先导出环境变量或 source 环境脚本。
通用[本地运行指南](../01-getting-started/quick-start.md#run-locally)
介绍了程序入口和 console 日志设置。

<a id="verify-integration"></a>

## 6. 验证后再扩大规模

1. 用已知正确和错误的结果检查评分逻辑。隐藏的标准答案文件缺失时，
   应报告环境错误，而不是判为任务失败。
2. 确认只有选定字段进入 Agent 输入，隐藏评测材料只在 Agent 清理后出现。
3. 在沙箱中运行一个小任务，检查 Agent 产物、评测奖励和记录的模型轨迹。
   Agent 打印了文本，不等于已经产生可训练的轨迹。
4. 完成一次小批量训练，按[示例中的检查步骤](../../recipe/e2b_minimal/README.md#inspect)
   同时确认任务得分和模型更新，再提高并发规模。

<a id="troubleshooting"></a>

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| 缺少任务所需字段 | 数据字段、`input_fields`，以及任务 hook 中的输入校验。 |
| 无法加载任务模块或工作区 | 宿主机路径是否为绝对路径，沙箱工作区是否已经存在。 |
| 找不到评测材料 | `task_id`、`task_files_root` 下的对应目录，以及 `evaluate` 预期的文件。 |
| 模型调用失败或没有训练轨迹 | 平台提供的 API 环境、接口类型、服务凭据，以及 sampler 发布状态。 |
| 长任务超时 | `agent_timeout_seconds`、Agent 客户端自身的请求超时，以及任务的清理逻辑。 |

<a id="further-reference"></a>

## 进一步参考

- [任务接口参考](e2b-task-contract.md)：函数参数、返回值和评分语义。
- [运行时流程](e2b-runtime.md)：执行、评测与故障恢复。
- [E2B 高级配置](../11-configuration-reference/e2b-advanced.md)：可选设置与运行细节。
