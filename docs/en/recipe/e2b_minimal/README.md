English | [简体中文](../../../zh/recipe/e2b_minimal/README.md) | [Documentation index](../../index.md)

<a id="overview"></a>

# Minimal Agent task training example

Run one task in an E2B sandbox, score its answer, and send the resulting trajectories to Weaver for one training step. NorthRL runs locally; model inference and training run on the services.

The shared [task module](../../../../recipe/e2b_minimal/rollout.py) contains two hooks:

| Hook | What this example does |
| --- | --- |
| `run_agent` | Sends `instruction` to the provided model API and returns the model's text as `AgentResult.final_answer`. |
| `evaluate` | Compares that text with the hidden `expected.txt`, ignoring surrounding whitespace. A match scores `1`; otherwise it scores `0`. Missing evaluation files are environment errors. |

NorthRL loads these hooks inside the sandbox. Running `python rollout.py` does not start training. This walkthrough supplies the data and complete training configuration; they are not pre-existing files in the example directory.

<a id="prerequisites"></a>

## 1. Prepare the environment

- Install NorthRL with `.[core]` in a Python 3.12 environment, following [local installation and execution](../../developer-guide/01-getting-started/quick-start.md#installation). Use that environment for every `python` command below. No local GPU is required.
- Have a [Weaver](https://weaver.nex-agi.cn/) endpoint, API key, and a model you can fine-tune. This configuration uses full fine-tuning; choose a compatible model and its matching tokenizer.
- Have a [NorthGate](https://northgate.xiaobei.top/) API key. Your Weaver service must automatically publish samplers to NorthGate, with the model's tokenizer, chat template, and response parser configured.
- Have an [E2B](https://e2b.dev/) API key and a template accessible to that key. The template needs a writable `/workspace`, root access for platform setup, and glibc-based Linux on x86-64 or ARM64. Bootstrap prepares Python 3.12 and the shared runtime; the sandbox must be able to download those dependencies and reach NorthGate. Keep hidden answers out of the template.

From the repository root, replace the placeholders and export these values in the same shell:

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

The model name must match your Weaver service; the tokenizer can be a matching Hugging Face ID or local directory. A custom E2B deployment may also require `E2B_DOMAIN` or `E2B_API_URL` from its operator. NorthRL obtains the NorthGate URL and sampler model name from Weaver; you do not hardcode them in the task module.

<a id="task-data"></a>

## 2. Create a task and its hidden answer

This command creates the example files outside the repository:

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

`instruction` goes to the Agent. `template` selects the sandbox, and `task_id` selects `evaluation/task-001/`. NorthRL uploads the hidden answer only after `run_agent` finishes and the Agent phase is finalized.

The working files will be organized as follows:

```text
$NORTHRL_E2B_EXAMPLE/
├── train.jsonl
├── evaluation/task-001/expected.txt
├── train.yaml                         # Create in the next step
└── logs/                              # Created when training starts
```

<a id="training-config"></a>

## 3. Save the training configuration

Save the complete configuration below as `$NORTHRL_E2B_EXAMPLE/train.yaml`:

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

One prompt is sampled twice to form one GRPO group. This is a workflow check, not a model-quality benchmark: equal rewards within the group produce zero relative advantage. Automatic retries and final checkpoint saving are disabled for this first run. Increase task diversity and training length, and enable `training.save.final`, when you move to real training.

`target` and `task_files_root` resolve to absolute paths on your machine; `/workspace` is a directory inside the sandbox. NorthRL uploads the selected task module and its shared runtime, not sibling files beside the task module. Keep evaluation files in the separate evaluation directory shown here; install any extra Agent dependencies in the template.

<a id="run"></a>

## 4. Check the configuration and start training

Set the recipe and a new output directory:

```bash
export TRAIN_CONFIG="$NORTHRL_E2B_EXAMPLE/train.yaml"
export EXPERIMENT_PATH="$NORTHRL_E2B_EXAMPLE/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPERIMENT_PATH"
```

Validate the configuration locally, without creating a sandbox or contacting the model services:

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

Then start the one-step run. This command uses your E2B and Weaver services:

```bash
set -o pipefail
python -m northrl.main \
  --config-path="$NORTHRL_E2B_EXAMPLE" \
  --config-name=train 2>&1 | tee "$EXPERIMENT_PATH/northrl.log"
```

Direct execution reads exported environment variables; it does not automatically load a `.env` file.

<a id="inspect"></a>

## 5. Check the answer, reward, and training step

- **Completion:** the command exits successfully and the log ends with `NorthRL training completed`. Also check `training/optimizer_step_count` is `1` and `training/skipped_flag` is `0`; a completed run can contain skipped training batches.
- **Reward:** inspect `debug_dump/rollout_rounds/step_000000.json`. Each rollout includes its `task_id`, `rollout_id`, `reward`, `e2b_status`, and `response_tokens`. An exact `READY` answer receives `1`; other answers receive `0`. `env_error` is an execution failure, not an incorrect answer.
- **Answer and model calls:** open `debug_dump/rollout_traces/e2b_rollout_<rollout_id>.json.gz` using the matching ID. Its `messages` include the model response. The trace is diagnostic data; use the round record for the final reward.
- **Training data:** `debug_dump/prepared_trajectories/step_000000.pt` contains the trajectories and computed advantages. `debug_dump/training_metrics/step_000000.pt` contains Weaver's returned training metrics when an update runs.

All paths above are relative to `$EXPERIMENT_PATH`. To print the round records and readable model messages:

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

If startup fails, check service credentials and sampler publication, the template's access to dependencies and NorthGate, and the absolute task/evaluation paths. The [runtime reference](../../developer-guide/05-rollout-workers/e2b-runtime.md) covers phase logs and environment failures. `northrl.debug_rollout` does not support E2B tasks; use this small training run to check the complete integration.

<a id="next"></a>

## Next: train your own Agent

Follow the [Agent integration guide](../../developer-guide/05-rollout-workers/e2b-task-integration.md) to replace `run_agent` with your Agent, adapt `evaluate` to its task outcomes, and add your own data. Refer to the [hook contract](../../developer-guide/05-rollout-workers/e2b-task-contract.md) for argument and return types, [execution budgets](../../developer-guide/11-configuration-reference/e2b-execution-budgets.md) for timeouts and recovery, and [advanced configuration](../../developer-guide/11-configuration-reference/e2b-advanced.md) for token history and network policies.
