[English](../../../en/recipe/advanced-recipes/README.md) | 简体中文 | [文档索引](../../index.md)

<a id="overview"></a>

# 进阶 Recipe 的资源准备

本页汇总进阶 recipe 的资源要求，说明共用的环境配置和本地启动方式。
各 recipe 已有的数据格式、模板构建要求和评测规则，链接到对应 README，
不在这里重复展开。安装方法见[快速开始](../../developer-guide/01-getting-started/quick-start.md)。

这些 recipe 提供训练配置和任务接入代码，不包含所需的数据集、评测包，
也不提供任务模板或资源卷的访问权限。首次验证训练流程，可以先使用
[最小 Agent 训练示例](../e2b_minimal/README.md)。

| Recipe | 需要准备的主要资源 | 配置与详细说明 |
| --- | --- | --- |
| GRPO R3 | 训练集、验证集、支持 R3 的 Weaver 模型 | [资源清单](#grpo-r3) |
| SWEbench | 任务数据、带代码仓库和 Claude Code 的模板、隐藏测试 | [资源清单](#swebench) |
| TMax | 任务数据、逐题模板、隐藏测试、Claude Code 卷 | [资源清单](#tmax) · [任务 README（中文）](../../../../recipe/tmax/README.md) |
| ATC v3 | 任务数据、带 ATC runtime 的模板、任务资源卷、隐藏评测包 | [资源清单](#atc-v3) · [任务 README（英文）](../../../../recipe/atc_v3/README.md) |
| JobBench | 任务数据、带工具及输入文件的模板、评测包、参考资料卷、独立评分模型 | [资源清单](#jobbench) · [任务 README（英文）](../../../../recipe/jobbench/README.md) |
| MiMo | 对应任务类型的数据、模板和评测资源；general 还需要独立评分模型 | [四类任务的资源清单](#mimo) |

<a id="shared-configuration"></a>

## 共用配置

| 变量 | 用途 |
| --- | --- |
| `WEAVER_BASE_URL`、`WEAVER_API_KEY` | Weaver 服务地址和访问凭据。 |
| `EXPERIMENT_PATH` | 本次运行的本地输出目录，每次独立实验使用新目录。 |
| `NORTHRL_PATH` | NorthRL 仓库的绝对路径；E2B recipe 用它定位任务入口。 |
| `E2B_API_KEY` | E2B 凭据，需要有任务模板及所用资源卷的访问权限。 |
| `NORTHGATE_API_KEY` | E2B 训练采样使用的 NorthGate 凭据。 |

后三项用于本页的 E2B recipe，GRPO R3 不需要 E2B 资源。使用自建 E2B 服务时，
还需按该服务的接入说明设置 `E2B_DOMAIN` / `E2B_API_URL`。
E2B 沙箱必须能访问 Weaver 发布的 NorthGate 采样地址。

根目录的 `run.sh` 会在加载环境文件前，将 `NORTHRL_PATH` 默认设为所在仓库根目录，
环境文件可以用它拼接路径。已提供的值会保留，环境文件也可以重新赋值。
直接使用 `python -m northrl.main` 时，需自行导出 `NORTHRL_PATH`。

模型名必须对应当前 Weaver 账户可用的模型。tokenizer 必须与训练模型一致，
可以填写 Hugging Face ID 或 NorthRL 进程能读取的本地目录。服务中的模型别名
不一定是有效的 tokenizer ID。切换模型时，还需检查上下文长度、采样参数、
微调能力和 NorthGate codec 配置。

将凭据和资源路径放在本地私有环境文件中，通过 `--env-file` 选择它。
[`run.sh`](../../../../run.sh) 会先加载并导出这些变量，再启动 Python；未显式指定文件时，
沿用当前终端中已导出的变量，不会自动加载 `.env` 或 `recipe/<名称>/env.sh`。
已有的 `NORTHRL_ENV_FILE` 写法仍然支持，`--env-file` 的优先级更高。
脚本使用当前 `PATH` 中的 `python`，或通过 `PYTHON` 指定的解释器，不会自动选择
仓库中的 `.venv`。若某 recipe 需要辅助环境脚本，在私有环境文件中显式 `source`，
例如下文的 JobBench。
具体加载规则见[环境文件](../../developer-guide/01-getting-started/configuration-setup.md#environment-files)，
完整示例见[首次小规模运行](#first-run)。凭据不要提交到 Git。

<a id="resource-locations"></a>

## 文件、模板和卷分别在哪里

| 配置内容 | 所在位置 | 示例 |
| --- | --- | --- |
| 训练数据、本地 tokenizer | 启动 NorthRL 的机器 | `/data/tmax/train.jsonl` |
| `entrypoint.target` | 同一台机器上的任务 Python 文件，使用绝对路径 | `/path/to/NorthRL/northrl/rollout_entrypoints/tmax_claude_code_rollout.py` |
| `entrypoint.task_files_root` | 同一台机器上的隐藏评测包根目录，使用绝对路径 | `/data/tmax/evaluation` |
| 数据行的 `template` | E2B 服务中的任务模板 ID 或别名 | `your-task-template` |
| `entrypoint.workspace`、卷挂载点 | E2B 沙箱内部 | `/home/user`、`/mnt/ccbin` |
| `volume_mounts` 中挂载点对应的值 | E2B 卷名或 ID，不是本地文件路径 | `your-claude-code-volume` |

本地路径必须能由 NorthRL 进程实际读取；填写路径不会自动挂载另一台机器的文件系统。
隐藏评测文件按 `<task_files_root>/<task_id>/` 存放，在 Agent 阶段结束后才上传。
任务评测目录及其内容不能是软链接，也不要把隐藏测试放进 Agent 可见的模板或资源卷。
各 recipe 的具体文件布局见其 README。

E2B rollout 从每行数据的 `template` 选择模板。设置 `E2B_TEMPLATE_ID` 不会覆盖
已有数据中的模板；最小示例只在生成数据时使用该变量。任务接入的路径和数据规则见
[接入指南](../../developer-guide/05-rollout-workers/e2b-task-integration.md)。

<a id="sandbox-environment"></a>

## 向沙箱传递环境变量

本地环境变量不会全部传入沙箱。需要额外工具或评分服务时，先在私有环境文件中
设置变量值，再用以下选择器列出变量名，以逗号分隔：

| 选择器 | 作用范围 |
| --- | --- |
| `NORTHRL_E2B_AGENT_ENV` | Agent 阶段命令，包括 `prepare` / `run_agent`；不传给评测阶段。任务入口决定哪些变量继续传给 Agent 子进程。 |
| `NORTHRL_E2B_EVALUATOR_ENV` | 仅评测阶段，例如独立评分模型的地址、凭据和模型名。 |

JobBench 和 MiMo general 的选择器示例见各自资源清单。Agent 调用训练中的模型时，
应使用任务接口提供的 `model_environment`；不要用独立模型服务的地址和 key 替换它，
否则调用不会经过训练轨迹采集路径。

<a id="grpo-r3"></a>

## 使用 R3 路由回放的 GRPO

配置文件：[`grpo_r3_northrl.yaml`](../../../../recipe/grpo_r3_northrl/grpo_r3_northrl.yaml)。

| 变量 | 所需资源 |
| --- | --- |
| `NORTHRL_TRAIN_DATA_PATH` | 必填，训练 JSONL 路径。 |
| `NORTHRL_R3_MATH500_DATA` | 必填，本地 Math500 验证集路径。 |
| `NORTHRL_R3_AIME2024_DATA` | 必填，本地 AIME 2024 验证集路径。 |
| `NORTHRL_R3_AIME2025_DATA` | 必填，本地 AIME 2025 验证集路径。 |
| `NORTHRL_R3_AIME2026_DATA` | 必填，本地 AIME 2026 验证集路径。 |
| `NORTHRL_R3_MODEL` | Weaver 模型名，默认 `Qwen/Qwen3-30B-A3B-Base`。 |
| `NORTHRL_R3_TOKENIZER` | 对应的 tokenizer，默认 `Qwen/Qwen3-30B-A3B-Base`。 |

该配置默认开启验证。需要准备四份验证集，或者在自己的配置副本中只保留
已有的数据集。如果不需要验证，将 `validate.enabled` 设为 `false`，并将
`validate.data` 设为 `null`，避免继续解析未使用的数据路径环境变量。数据行需符合
配置中的 prompt 和奖励格式，详见[数据配置](../../developer-guide/11-configuration-reference/data-config.md)
和[单轮任务指南](../math/README.md)。

Weaver 服务必须支持所选模型的 R3 路由回放；仅修改模型名不能启用这项能力。
附带的 [`env.sh`](../../../../recipe/grpo_r3_northrl/env.sh) 中是空的本地配置项，
加载前需在私有副本中填写，凭证不要提交到 Git。

<a id="swebench"></a>

## 使用 Claude Code 的 SWEbench

配置文件：[`e2b_claude_grpo.yaml`](../../../../recipe/swebench/e2b_claude_grpo.yaml)。

| 必填变量 | 所需资源 |
| --- | --- |
| `NORTHRL_SWEBENCH_MODEL` | Weaver 服务中可训练的模型名。 |
| `NORTHRL_SWEBENCH_TOKENIZER` | 与该模型匹配的 tokenizer ID 或本地目录。 |
| `NORTHRL_SWEBENCH_TRAIN_DATA` | 包含 `task_id`、`template` 和 `instruction` 的训练数据。 |
| `NORTHRL_SWEBENCH_TASK_FILES_ROOT` | 各任务隐藏评测包的根目录。 |

每行数据的 `template` 必须指向当前 E2B API key 可访问的模板。
[任务入口](../../../../northrl/rollout_entrypoints/swerebench_claude_code_rollout.py)
要求沙箱内恰好有一个位于根目录下的 Git 工作目录，并在 `/usr/local/bin/claude`
提供 Claude Code。如果使用依赖 Node 的 Claude 安装方式，还需要对应的 Node
可执行文件。运行时文件必须满足入口对 root 所有权和写权限的检查。
默认的域名访问限制还需要模板提供 `nft`、`setpriv`，以及设置相应网络规则的能力。

在 `<NORTHRL_SWEBENCH_TASK_FILES_ROOT>/<task_id>/` 中准备该任务的
`run-tests.sh`、`tests/config.json`、`log_parsers.py` 和 `swe_constants.py`。
这些文件在 Agent 执行结束后才用于评测。本仓库不包含模板构建源码、任务代码仓库、
数据集或评测包；准备好这些资源后才能运行此 recipe。

<a id="tmax"></a>

## TMax 终端任务

配置文件：[`train.yaml`](../../../../recipe/tmax/train.yaml)。

| 必填变量 | 所需资源 |
| --- | --- |
| `NORTHRL_TMAX_MODEL` | Weaver 服务中可训练的模型名。 |
| `NORTHRL_TMAX_TOKENIZER` | 与该模型匹配的 tokenizer ID 或本地目录。 |
| `NORTHRL_TMAX_TRAIN_DATA` | 包含 `task_id`、`template`、`instruction` 的 JSONL；可带 `band` 元数据。 |
| `NORTHRL_TMAX_TASK_FILES_ROOT` | 评测包根目录，文件位于 `<task_id>/run-tests.sh`。 |
| `NORTHRL_TMAX_CC_VOLUME` | 自己可访问的 E2B 卷，根目录含可执行的 `claude`，不提供默认卷。 |

模板中的用户、工作目录和依赖，Claude Code 卷的准备方式，数据样例、评测目录布局及
结果块格式，均见 [TMax README](../../../../recipe/tmax/README.md)。所用 E2B 部署需要支持
此 recipe 的卷挂载和卸载。模型仅支持文本时，应排除依赖图片或视频输入的任务。

默认配置为每步 32 题 × 16 次 rollout、512 个 worker、109 步。
首次运行建议使用下文的[小规模参数](#first-run)，再按数据量和服务容量扩大规模。

<a id="atc-v3"></a>

## ATC v3

配置文件：[`e2b_claude_grpo.yaml`](../../../../recipe/atc_v3/e2b_claude_grpo.yaml)。

| 必填变量 | 所需资源 |
| --- | --- |
| `NORTHRL_ATC_MODEL` | Weaver 服务中可训练的模型名。 |
| `NORTHRL_ATC_TOKENIZER` | 与该模型匹配的 tokenizer ID 或本地目录。 |
| `NORTHRL_ATC_TRAIN_DATA` | 准备好的 ATC 数据集，包含各任务的模板和资源元数据。 |
| `NORTHRL_ATC_EVALUATION_ROOT` | 各 `<task_id>/` 下包含隐藏的 `manifest.json` 和 `verifier.py`。 |
| `NORTHRL_ATC_VOLUME` | 自己可访问的任务资源卷，挂载到 `/mnt/data`。 |

数据布局、`asset_bundles` 与卷内文件的对应关系、模板内 ATC runtime 的接口及
权限要求，见 [ATC README](../../../../recipe/atc_v3/README.md)。模板中的 runtime、
MCP 实现及评测所需资源需要自行提供，通用 E2B 模板不能直接替代。

<a id="jobbench"></a>

## JobBench

配置文件：[`grpo.yaml`](../../../../recipe/jobbench/grpo.yaml)。

| 变量 | 所需资源 |
| --- | --- |
| `NORTHRL_JOBBENCH_ACTOR` | Weaver 服务中可训练的模型名。 |
| `NORTHRL_JOBBENCH_TOKENIZER` | 对应 tokenizer；未设置时使用模型名，只有该名称同时是有效 tokenizer ID 时才可省略。 |
| `NORTHRL_JOBBENCH_DATA_DIR` | 已准备好的 `jobbench.jsonl` 和 `evaluation/` 的共同父目录。 |
| `NORTHRL_JOBBENCH_REFERENCE_VOLUME` | 当前 YAML 挂载的参考资料卷。 |
| `JOBBENCH_REFERENCES_ROOT`、`JOBBENCH_REFERENCES_MANIFEST_SHA256` | 沙箱内参考资料发布目录及其 manifest 的 SHA256。 |
| `JOBBENCH_JUDGE_BASE_URL`、`JOBBENCH_JUDGE_API_KEY`、`JOBBENCH_JUDGE_MODEL` | 独立评测模型的 Anthropic 兼容接口、凭据和模型名。 |
| `SERPER_API_KEY` | 可选，为 Agent 启用搜索；不提供时不启用该搜索能力。 |

在私有环境文件中填写以上值后，补上：

```bash
NORTHRL_E2B_EVALUATOR_ENV=JOBBENCH_JUDGE_BASE_URL,JOBBENCH_JUDGE_API_KEY,JOBBENCH_JUDGE_MODEL
source "$NORTHRL_PATH/recipe/jobbench/env.sh"
```

辅助脚本保留已有值，设置默认搜索地址，并把参考资料和搜索变量加入 Agent 阶段选择器。
若已有评测变量选择器，应合并所需变量名。评分模型的请求不属于策略模型的训练轨迹。
有意不使用参考资料时，需要同时移除配置中的卷映射并取消参考资料变量，
不能仅将必需的卷名留空。

数据和评测包格式、OpenCode 模板要求、参考资料布局及评分规则见
[JobBench README](../../../../recipe/jobbench/README.md)。其中提到的具体模板名称和 ID
不代表公开访问权限；实际使用的是数据行中当前账号可访问的兼容模板。

<a id="mimo"></a>

## MiMo

四份配置共用 `NORTHRL_MIMO_ACTOR`（Weaver 模型名）、`NORTHRL_MIMO_TOKENIZER`
（匹配的 tokenizer）和 `NORTHRL_MIMO_DATA_DIR`（本地数据根目录）。tokenizer 未设置时
使用模型名；服务别名不能直接作为 tokenizer ID 时，必须显式填写。

下面列出四类任务的资源布局和对应入口。只需准备所选任务类型的文件；
数据、模板构建源码及隐藏评测包均不随仓库提供。

| 配置 | 数据根目录下的 JSONL | 同一根目录下的评测包 | 模板需具备的主要内容 |
| --- | --- | --- | --- |
| [code](../../../../recipe/mimo/code_grpo.yaml) | `code.jsonl` | `evaluation/code/<task_id>/` 中的 `evaluation.json`、`test.patch` | 任务 Git 仓库和测试依赖；[任务入口](../../../../northrl/rollout_entrypoints/mimo/code.py)。 |
| [terminal](../../../../recipe/mimo/terminal_grpo.yaml) | `terminal.jsonl` | `evaluation/terminal/<task_id>/` 中的 `evaluation.json`、`tests/test.sh` | 对应终端任务环境和工具；[任务入口](../../../../northrl/rollout_entrypoints/mimo/terminal.py)。 |
| [general](../../../../recipe/mimo/general_grpo.yaml) | `general.jsonl` | `evaluation/general/<task_id>/` 中的 `evaluation.json`、`verify.py`、`verifier_meta.json`、`_helpers.py`、`run_verify.py` | `/work/workspace`、任务 MCP 服务及其运行依赖；[任务入口](../../../../northrl/rollout_entrypoints/mimo/general.py)。 |
| [arvo](../../../../recipe/mimo/arvo_grpo.yaml) | `arvo.jsonl` | 当前配置不使用本地 `task_files_root` | 任务二进制、Agent / verifier 用户及 `/root/binary/run.sh`；[任务入口](../../../../northrl/rollout_entrypoints/mimo/arvo.py)。 |

数据行需要 `template`、`task_id`、`instruction`、`cwd` 和 `agent_user`；其他字段及
评测配置遵循对应入口。`cwd` 必须是模板内已有的绝对路径，`agent_user` 必须对应
模板中的非 root 用户，Claude Code 位于 `/opt/mimo-agents/claude`。
这些是任务专用模板，空白 Linux 模板不足以运行。

general 另外需要独立评分模型，在私有环境文件中配置：

```bash
GA_JUDGE_URL="https://your-judge-endpoint"
GA_JUDGE_KEY="your-judge-key"
GA_JUDGE_MODEL="your-judge-model"
GA_JUDGE_API="chat"  # 支持 chat、responses、gemini；需匹配评分服务
NORTHRL_E2B_EVALUATOR_ENV=GA_JUDGE_URL,GA_JUDGE_KEY,GA_JUDGE_MODEL,GA_JUDGE_API
```

上述变量仅传给评测阶段；若已有选择器，应合并变量名。E2B 的训练采样由服务端
NorthGate codec 处理，`rollout.parsers`、`execution.tokenizer` 不用于配置该采样路径，
参见[任务接入指南](../../developer-guide/05-rollout-workers/e2b-task-integration.md)。

<a id="first-run"></a>

## 首次小规模运行

以下以 TMax 为例，前提是已安装 NorthRL 运行依赖，并准备好任务数据、模板和卷。
将下面内容保存到私有文件 `/absolute/path/tmax.env`，替换所有示例值：

```bash
EXPERIMENT_PATH="/absolute/path/to/new-tmax-run"
WEAVER_BASE_URL="https://your-weaver-host"
WEAVER_API_KEY="your-weaver-key"
NORTHGATE_API_KEY="your-northgate-key"
E2B_API_KEY="your-e2b-key"
# 自建 E2B 服务：按接入说明补充 E2B_DOMAIN / E2B_API_URL。

NORTHRL_TMAX_MODEL="your-weaver-model"
NORTHRL_TMAX_TOKENIZER="/absolute/path/to/matching-tokenizer"
NORTHRL_TMAX_TRAIN_DATA="/data/tmax/train.jsonl"
NORTHRL_TMAX_TASK_FILES_ROOT="/data/tmax/evaluation"
NORTHRL_TMAX_CC_VOLUME="your-claude-code-volume"
```

从 NorthRL 仓库根目录运行，每步 1 题、每题 4 次 rollout、4 个 worker，只运行 1 步：

```bash
bash run.sh --env-file /absolute/path/tmax.env recipe/tmax/train.yaml \
  data.train.batch_size=1 \
  data.train.rollout_repeat_n=4 \
  rollout.num_workers=4 \
  rollout.max_concurrent_requests=4 \
  training.total_steps=1 \
  training.save.every_n_steps=1 \
  algorithm.optim.mini_batch_size=4 \
  'runtime.logger.backends=[console]'
```

如果这些变量已经在终端导出，可以省略 `--env-file` 及其路径。
若之前导出过 `NORTHRL_ENV_FILE`，请先取消设置，避免继续加载那个文件。

保留原配置的 `algorithm.optim.batch_size=null`，使完整 rollout batch 在同一次
optimizer update 中处理；`mini_batch_size=4` 只限制单次训练请求的 Datum 数量。
多轮轨迹可能产生多个 Datum，不能将它直接等同于 rollout 数量。
缩小并发和步数不会缩短单任务超时预算；TMax 的 Agent 默认仍可运行 1 小时。

此示例只启用终端日志，无需 W&B 配置。`run.sh` 不保存终端输出，需要保存时参照
[日志说明](../../developer-guide/01-getting-started/quick-start.md#monitoring)。该配置的调试输出位于
`EXPERIMENT_PATH`，检查点元数据位于其 `ckpt/` 目录；模型权重由 Weaver 管理。

运行后同时检查命令退出状态、`NorthRL training completed` 日志和训练指标。
`training/skipped_flag=1` 表示本步跳过了训练更新；结合
`training/forward_backward_call_count` 和 rollout 错误定位原因。
即使执行了训练请求，同一题的采样全部同分时，GRPO 优势值也可能全为零。
这次小规模运行用于验证流程，不能据此判断训练效果。

<a id="troubleshooting"></a>

## 常见配置问题

| 现象 | 优先检查 |
| --- | --- |
| 缺少环境变量或本地文件 | `--env-file` 路径或已导出的终端变量是否正确；变量是否已填写；NorthRL 进程是否能读取数据、tokenizer 和评测文件。 |
| 模板不存在或无权限 | 数据行的 `template`、E2B 服务地址及当前 key 的权限。 |
| 卷挂载失败 | 填的是卷名 / ID 而非本地路径；卷与模板属于同一 E2B 服务；该部署支持所需挂载能力。 |
| 找不到评测包或提示软链接 | `task_id` 与评测目录名称是否一致，所需文件是否齐全且为真实文件。 |
| 评分模型缺少配置 | 对应 judge 变量是否已填写，并列在 `NORTHRL_E2B_EVALUATOR_ENV` 中。 |
| 采样地址不可达或模型不匹配 | Weaver 发布的 NorthGate 地址能否从沙箱访问，以及模型权限、tokenizer 和 codec 是否匹配。 |
| 运行结束但没有训练更新 | rollout 是否有环境错误或缺少训练片段；查看训练跳过标志和训练请求计数。 |

任务未通过评测可以是正常的零分样本；模板、文件或服务配置错误需要按具体 recipe 的
错误处理规则排查，不能仅凭 reward 为零判断原因。内部真实路径、资源名和实验配置
可保存在私有 recipes 仓库，公开文档使用可替换的示例值。
