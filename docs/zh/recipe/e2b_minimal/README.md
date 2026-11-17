[English](../../../en/recipe/e2b_minimal/README.md) | 简体中文 | [文档目录](../../index.md)

<a id="overview"></a>

# 最小 Agent 任务训练示例

在 E2B 沙箱中执行一条任务，为回答评分，再将轨迹交给 Weaver 完成一个训练步。NorthRL 在本地运行，模型推理和训练由服务完成。

共用的[任务代码](../../../../recipe/e2b_minimal/rollout.py)实现两个接口：

| 接口 | 这个示例做什么 |
| --- | --- |
| `run_agent` | 将 `instruction` 发送给平台提供的模型接口，把模型回复放入 `AgentResult.final_answer` 返回。 |
| `evaluate` | 将回答与隐藏的 `expected.txt` 比较，忽略首尾空白。相同得 `1` 分，否则得 `0` 分；评测文件缺失属于环境错误。 |

NorthRL 在沙箱中加载并调用这两个接口，执行 `python rollout.py` 不会启动训练。本页会带你准备数据和完整训练配置；示例目录中并没有预先放置这些文件。

<a id="prerequisites"></a>

## 1. 准备环境

- 按照[本地安装与运行说明](../../developer-guide/01-getting-started/quick-start.md#installation)，在 Python 3.12 环境中安装 NorthRL 的 `.[core]` 依赖。后续所有 `python` 命令都使用这个环境，不需要本地 GPU。
- 准备 [Weaver](https://weaver.nex-agi.cn/) 服务地址、API key，以及有权微调的模型。本配置使用全参数微调，需要选择支持该模式的模型和匹配的 tokenizer。
- 准备 [NorthGate](https://northgate.xiaobei.top/) API key。Weaver 服务需要开启 sampler 自动发布到 NorthGate，并配置模型对应的 tokenizer、chat template 和响应 parser。
- 准备 [E2B](https://e2b.dev/) API key，以及该 key 有权使用的模板。模板需要可写的 `/workspace`、供平台初始化使用的 root 权限，以及 x86-64 或 ARM64 的 glibc Linux 环境。平台初始化会准备 Python 3.12 和公共运行依赖；沙箱需要能够下载这些依赖并访问 NorthGate。不要在模板中预放隐藏答案。

在仓库根目录执行以下命令，将占位值替换为实际配置，后续步骤继续使用同一个 shell：

```bash
export NORTHRL_PATH="$PWD"
export NORTHRL_E2B_EXAMPLE="$HOME/northrl-e2b-example"
export WEAVER_BASE_URL="https://your-weaver-host"
export WEAVER_API_KEY="your-weaver-api-key"
export NORTHGATE_API_KEY="your-northgate-api-key"
export E2B_API_KEY="your-e2b-api-key"
export E2B_TEMPLATE_ID="your-e2b-template-id"
export NORTHRL_MODEL="your-weaver-model"
export NORTHRL_TOKENIZER="your-matching-tokenizer"
```

模型名以 Weaver 服务支持的名称为准；tokenizer 可以是匹配的 Hugging Face ID 或本地目录。自建 E2B 部署可能还需要服务方提供的 `E2B_DOMAIN` 或 `E2B_API_URL`。NorthRL 会从 Weaver 获取 NorthGate 地址和 sampler 模型名，无需在任务代码中写死这些值。

<a id="task-data"></a>

## 2. 准备任务和隐藏答案

以下命令在仓库外创建示例文件：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["NORTHRL_E2B_EXAMPLE"])
assets = root / "evaluation" / "task-001"
assets.mkdir(parents=True, exist_ok=True)
task = {
    "instruction": "Reply with exactly: READY",
    "template": os.environ["E2B_TEMPLATE_ID"],
    "task_id": "task-001",
}
(root / "train.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
(assets / "expected.txt").write_text("READY\n", encoding="utf-8")
PY
```

`instruction` 是 Agent 收到的任务；`template` 选择沙箱模板；`task_id` 对应 `evaluation/task-001/`。只有在 `run_agent` 结束、Agent 阶段完成收尾之后，NorthRL 才会上传隐藏答案。

工作文件组织如下：

```text
$NORTHRL_E2B_EXAMPLE/
├── train.jsonl
├── evaluation/task-001/expected.txt
├── train.yaml                         # Create in the next step
└── logs/                              # Created when training starts
```

<a id="training-config"></a>

## 3. 保存训练配置

将下面的完整配置保存为 `$NORTHRL_E2B_EXAMPLE/train.yaml`：

```yaml
experiment:
  name: e2b-minimal
  project: NorthRL-examples
models:
  actor:
    name: ${oc.env:NORTHRL_MODEL}
    tokenizer: ${oc.env:NORTHRL_TOKENIZER}
    lora:
      enabled: false
data:
  train:
    files: ["${oc.env:NORTHRL_E2B_EXAMPLE}/train.jsonl"]
    batch_size: 1
    rollout_repeat_n: 2
    prompt_key: instruction
    shuffle: false
    drop_last: false
  max_prompt_length: 1024
  max_sequence_length: 4096
validate:
  enabled: false
algorithm:
  type: grpo
  sampling_logprob_source: inference
  optim:
    lr: 1.0e-6
    batch_size: null
  entropy_coeff: 0.0
rollout:
  type: e2b
  num_workers: 2
  max_concurrent_requests: 2
  sampling:
    temperature: 0.8
    max_tokens: 1024
  execution:
    input_fields: [instruction]
    agent_timeout_seconds: 120
    env_error_retries: 0
    entrypoint:
      target: ${oc.env:NORTHRL_PATH}/recipe/e2b_minimal/rollout.py
      workspace: /workspace
      task_files_root: ${oc.env:NORTHRL_E2B_EXAMPLE}/evaluation
training:
  total_steps: 1
  save:
    final: false
    dir: ${oc.env:EXPERIMENT_PATH}/ckpt
  resume:
    max_retries: 0
  weight_sync:
    mode: sync
backend:
  type: weaver
runtime:
  logger:
    backends: [console]
  debug:
    enable_data_dump: true
    dump_dir: ${oc.env:EXPERIMENT_PATH}/debug_dump
    remote_api_dump_options:
      prepared_trajectories: true
      training_metrics: true
```

同一条任务采样两次，组成一个 GRPO 组。这是在检查流程能否跑通，不用于判断模型质量：如果组内两次奖励相同，相对优势就为零。首次运行关闭自动重试和最终 checkpoint 保存。正式训练时，应增加任务多样性和训练步数，并开启 `training.save.final`。

`target` 和 `task_files_root` 解析为本机绝对路径，`/workspace` 则是沙箱内的目录。NorthRL 上传指定的任务模块和公共运行代码，不会自动打包任务模块旁边的其他文件。评测文件放在本例中单独的评测目录；Agent 需要的额外依赖预装在模板中。

<a id="run"></a>

## 4. 检查配置并启动训练

指定配置文件，并创建本次运行的输出目录：

```bash
export TRAIN_CONFIG="$NORTHRL_E2B_EXAMPLE/train.yaml"
export EXPERIMENT_PATH="$NORTHRL_E2B_EXAMPLE/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPERIMENT_PATH"
```

先在本地校验配置，不会创建沙箱或调用模型服务：

```bash
python - <<'PY'
import os
from omegaconf import OmegaConf
from northrl.utils.config_schema import NorthRLConfig

config = OmegaConf.load(os.environ["TRAIN_CONFIG"])
NorthRLConfig.model_validate(OmegaConf.to_container(config, resolve=True))
print("Configuration is valid")
PY
```

然后启动一次训练。以下命令会使用你的 E2B 和 Weaver 服务：

```bash
set -o pipefail
python -m northrl.main \
  --config-path="$NORTHRL_E2B_EXAMPLE" \
  --config-name=train 2>&1 | tee "$EXPERIMENT_PATH/northrl.log"
```

直接启动读取的是已经导出的环境变量，不会自动加载 `.env` 文件。

<a id="inspect"></a>

## 5. 检查回答、奖励和训练结果

- **是否完成：** 命令成功退出，日志末尾出现 `NorthRL training completed`。还要确认 `training/optimizer_step_count` 为 `1`、`training/skipped_flag` 为 `0`；运行结束并不代表每个批次都执行了参数更新。
- **奖励：** 查看 `debug_dump/rollout_rounds/step_000000.json`。每条 rollout 包含 `task_id`、`rollout_id`、`reward`、`e2b_status` 和 `response_tokens`。仅回答 `READY` 得 `1` 分，其他回答得 `0` 分；`env_error` 表示执行环境故障，不是答错题。
- **回答和模型调用：** 按照对应的 ID 打开 `debug_dump/rollout_traces/e2b_rollout_<rollout_id>.json.gz`，其中 `messages` 包含模型回复。trace 用于排查执行过程，最终奖励以训练批次记录为准。
- **训练数据：** `debug_dump/prepared_trajectories/step_000000.pt` 保存轨迹及计算后的优势值；实际执行更新时，`debug_dump/training_metrics/step_000000.pt` 保存 Weaver 返回的训练指标。

以上路径均相对于 `$EXPERIMENT_PATH`。下面的命令打印批次记录和可读的模型消息：

```bash
python - <<'PY'
import gzip
import json
import os
from pathlib import Path

root = Path(os.environ["EXPERIMENT_PATH"]) / "debug_dump"
round_data = json.loads((root / "rollout_rounds/step_000000.json").read_text())
for record in round_data["rollouts"]:
    print(json.dumps(record, indent=2, ensure_ascii=False))
    trace = root / "rollout_traces" / f"e2b_rollout_{record['rollout_id']}.json.gz"
    if trace.exists():
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            print(json.dumps(json.load(stream).get("messages"), indent=2, ensure_ascii=False))
PY
```

启动失败时，先检查服务凭据和 sampler 发布状态、模板能否访问依赖源和 NorthGate，以及任务/评测文件的绝对路径。[运行机制参考](../../developer-guide/05-rollout-workers/e2b-runtime.md)说明各阶段日志和环境错误的处理方式。`northrl.debug_rollout` 不支持 E2B 任务，检查完整接入流程请使用这里的小批训练。

<a id="next"></a>

## 下一步：训练自己的 Agent

按照 [Agent 接入指南](../../developer-guide/05-rollout-workers/e2b-task-integration.md)，将 `run_agent` 换成你的 Agent，根据任务结果实现 `evaluate`，再接入自己的数据。[接口约定](../../developer-guide/05-rollout-workers/e2b-task-contract.md)介绍参数和返回类型，[执行时限](../../developer-guide/11-configuration-reference/e2b-execution-budgets.md)介绍超时与恢复，[进阶配置](../../developer-guide/11-configuration-reference/e2b-advanced.md)介绍历史 token 复用和网络策略。
