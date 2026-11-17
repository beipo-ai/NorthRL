English | [简体中文](../../../zh/recipe/math/README.md) | [Documentation index](../../index.md)

# Single Turn Math Task

This folder contains recipes for training models on single-turn math problems on the Weaver backend.

## Task Description

Single-turn math is a task where the model receives a math problem and must generate the correct solution in a single response. The reward is calculated using rule-based evaluation comparing the generated answer to the ground truth.

## Configuration Files

- **`weaver.yaml`** - Weaver training, LoRA fine-tuning (rank 32)
- **`weaver_full_ft.yaml`** - Weaver training, full fine-tuning (no LoRA, lower lr)

Both files are self-contained in the current 9-block config schema (`experiment` / `models` / `data` / `validate` / `algorithm` / `rollout` / `training` / `backend` / `runtime`).

They use `rollout.type: single_turn` with a math `reward_fn`.

## Configuration Requirements

These recipes read environment variables via `${oc.env:...}`. For direct execution, set them in the environment running the controller:

```bash
export NORTHRL_DATA_PATH=/path/to/your/data        # training/test parquet root
export EXPERIMENT_PATH=/path/to/output           # checkpoint output dir
export WEAVER_API_KEY=sk-your-weaver-api-key      # or set in the env, not the yaml
export WEAVER_BASE_URL=https://your-weaver-host   # weaver service endpoint
```

**Optional:**
- `export FEISHU_WEBHOOK_URL=https://your-webhook-url` (for notifications)

Install the runtime dependencies and prepare data access in the same local environment. If you keep exports in a private shell script, source it before starting Python.

## Data Format

The supplied recipes point to Parquet files. JSONL is also supported by changing
`data.train.files`; use the same fields in either format:

- `prompt`: A list of chat messages with `role` and `content`, not a plain string.
- `label`: The reference answer, preferably a string. Legacy `ground_truth` and
  `reward_model.ground_truth` are also supported.
- `reward_fn`: Optional short reward name for this row. It overrides the recipe's
  `rollout.reward_fn`, which defaults to `math_with_format` in these recipes.
- `metadata`: Optional task information passed through to the reward function.

For example, a JSONL row selecting binary boxed-answer scoring is:

```json
{"prompt": [{"role": "user", "content": "What is 40 + 2? Put the final answer in \\boxed{}."}], "label": "42", "reward_fn": "math_answer_only"}
```

To use this reward for every row that omits `reward_fn` or sets it to `null`, set:

```yaml
rollout:
  reward_fn: math_answer_only
```

Different rows may select different registered rewards in the same file. Unknown
names and explicit empty strings raise an error. See the
[data format and mixed-reward examples](../../developer-guide/03-data-loader/data-loader.md#single-turn-reward-data)
and the [supported reward names](../../developer-guide/11-configuration-reference/rollout-config.md#single-turn-reward-selection).
The bp-rl-judge `math` reward has different behavior and is not registered here.

The supplied recipes expect these files:
```
${NORTHRL_DATA_PATH}/math/
├── level1_train.parquet
├── level2_train.parquet
├── level3_train.parquet
├── level4_train.parquet
├── level5_train.parquet
├── level1_test.parquet
├── level2_test.parquet
├── level3_test.parquet
├── level4_test.parquet
└── level5_test.parquet
```

## Usage

Follow the [local quick start](../../developer-guide/01-getting-started/quick-start.md#run-locally) to install the runtime and prepare service access. After setting the variables above, run from the repository root:

```bash
export TRAIN_CONFIG="$(pwd)/recipe/math/weaver.yaml"
# For full fine-tuning, select recipe/math/weaver_full_ft.yaml instead.
mkdir -p "$EXPERIMENT_PATH"
python -m northrl.main \
  --config-path="$(dirname "$TRAIN_CONFIG")" \
  --config-name="$(basename "${TRAIN_CONFIG%.*}")" \
  'runtime.logger.backends=[console]'
```

## Key Configuration Parameters

- `data.train.batch_size`: Prompts per batch (default: 32)
- `data.train.rollout_repeat_n`: Responses per prompt (default: 8)
- `training.total_steps`: Total training steps (default: 100)
- `rollout.num_workers`: Parallel rollout workers (default: 256)
- `rollout.sampling.temperature`: Sampling temperature (default: 1.0)
- `algorithm.optim.lr`: Learning rate (weaver: 1e-4 / full_ft: 2e-6)
- `models.actor.lora.enabled`: LoRA vs full fine-tuning (weaver: true / full_ft: false)
- `data.max_prompt_length` / `data.max_sequence_length`: 4096 / 12288 (generation budget derives to 8192)
