English | [简体中文](../../../zh/recipe/sft_quickstart/README.md) | [Documentation index](../../index.md)

<a id="sft-快速上手"></a>

# SFT Quick Start

Start with a checkout of NorthRL containing this recipe. You need a Linux or macOS machine
with Python 3.12+, internet access to PyPI and Hugging Face, and a Weaver account
with permission to train the chosen model. The driver runs locally; Weaver runs
the GPU trainer. No local GPU or preinstalled NorthRL environment is required.

<a id="setup"></a>

## 1. Set up your own environment

From the repository root, install once:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  .venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -e ".[core]"
cp recipe/sft_quickstart/.env.example .env
```

On Linux the CPU PyTorch wheel avoids local CUDA dependencies; macOS uses its
standard PyPI wheel. The launcher
selects `.venv/bin/python` automatically; you do not need to activate it.
Edit just these two lines in `.env` to try the bundled dataset:

```dotenv
WEAVER_BASE_URL=https://your-weaver-host
WEAVER_API_KEY=your-real-weaver-key
```

The defaults use the public `Qwen/Qwen3.5-0.8B:262144` Weaver entry and download
only its tokenizer from `Qwen/Qwen3.5-0.8B` on Hugging Face. The model weights stay
on Weaver. Your account must have access to this entry. The bundled eight-row
sample is for checking the workflow, not for improving model quality.
`.env` is git-ignored and uses shell assignment syntax; quote values with spaces.

<a id="1-配置训练数据"></a>

## 2. Use the bundled sample or supply your own data

The default `.env` is ready to use:

```dotenv
SFT_DATA_PATH=recipe/sft_quickstart/train.example.jsonl
SFT_MODEL=Qwen/Qwen3.5-0.8B:262144
SFT_TOKENIZER=Qwen/Qwen3.5-0.8B
SFT_BATCH_SIZE=4
SFT_MAX_LENGTH=4096
SFT_EPOCHS=1
```

To use your own data, put it at `data/train.jsonl` under the repository root and
change `SFT_DATA_PATH=data/train.jsonl`. Each line is a conversation:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

Only assistant completions and their end-of-turn tokens contribute to the loss.
You can also supply matching-tokenizer arrays of equal length:

```json
{"input_ids":[1,2,3,4],"loss_mask":[0,0,1,1]}
```

`SFT_BATCH_SIZE` is the number of samples per optimizer step. Use at least that
many rows, or reduce it. An incomplete final batch is dropped each epoch;
`SFT_MAX_LENGTH` truncates each sample. Data and output paths in `.env` are relative
to the repository root unless absolute. The local driver reads and prepares the data.

<a id="2-在-yaml-中配置-weaver"></a>

## Choose Qwen3.8-Flash-Next (optional)

Change these lines in `.env` if your Weaver account can access this model:

```dotenv
SFT_MODEL=Qwen/Qwen3.8-Flash-Next:262144
SFT_TOKENIZER=Qwen/Qwen3.8-Flash-Next
SFT_MAX_LENGTH=262144
SFT_LR=5.0e-6
```

The tokenizer is downloaded from Hugging Face automatically. No internal model
path or model weights are needed locally. Service-side startup can take several minutes.
This recipe uses full fine-tuning (`lora.enabled: false`) for either model.

<a id="3-启动训练"></a>

## 3. Check and launch

```bash
bash recipe/sft_quickstart/run.sh --check
bash recipe/sft_quickstart/run.sh
```

`--check` renders/tokenizes all rows and counts the optimizer windows in one
epoch. It downloads the tokenizer when needed but does not contact Weaver or
start training. Invalid rows and an undersized dataset fail before training.
The normal command repeats that check, starts training, and prints the run
directory. Leave the driver running; use `tmux` for a long job.
Service access and model permissions are checked when training starts; Weaver
manages the resources needed for training.

For a two-step smoke without checkpoint saving:

```bash
bash recipe/sft_quickstart/run.sh   training.epochs=null +training.total_steps=2 training.save.final=false
```

This starts a real training run on Weaver. The default eight-row dataset and batch size
four yield two steps per epoch.

<a id="monitoring"></a>

## 4. Monitor and collect the result

Set `RUN` to the actual directory printed by the launcher:

```bash
RUN=logs/sft-THE-PRINTED-RUN-ID
tail -f "$RUN/northrl.log"
grep 'Training step .* metrics:' "$RUN/northrl.log" | tail -3
cat "$RUN/exit_code"
```

Look for `weaver/loss`, `weaver/grad_norm`, and `weaver/update_successful` on each
completed step. The log also prints the Weaver model ID. Startup can take several
minutes. `exit_code` is written after the command exits; `0` and
`NorthRL training completed` mean the driver completed. A missing exit file or
an error needs investigation; service availability alone does not mean training succeeded.

Detailed observations are under `$RUN/weaver-metrics/metrics-*/`; effective
configuration is in `$RUN/train_config_resolved.yaml`.

### Optional W&B loss curves

Add your W&B credentials to `.env`, then enable the backend:

```dotenv
WANDB_HOST=https://api.wandb.ai
WANDB_KEY=your-wandb-api-key
```

```bash
bash recipe/sft_quickstart/run.sh 'runtime.logger.backends=[console,wandb]'
```

The log prints the W&B run URL. Training loss is the numbered W&B metric for
`weaver/loss`, plotted against `training/train_step`. The installed W&B SDK
supports both legacy and new long API keys.

<a id="4-训练步数与-checkpoint"></a>

## Checkpoints and repeat runs

Normal training saves a final resumable checkpoint. Its server-managed path is
recorded in `$RUN/ckpt/weaver_checkpoints.jsonl`; weights are not downloaded to
this directory. The smoke command above disables the save. For periodic saves,
append `training.save.every_n_steps=100`. Set exactly one of epochs and total steps.

Every launch gets a new `logs/sft-...` directory. Override with `EXPERIMENT_PATH`
in `.env` if needed; use a new directory each time. The script refuses to overwrite
a prior run's log. Rerunning this quickstart creates a new model; explicit resume
is described in the [data guide](../../developer-guide/03-data-loader/sft-data.md#prefetch-and-exact-resume).

<a id="5-sft-配置约束"></a>

## More configuration

`SFT_BATCH_SIZE` also sets the token-budget multiplier: each step stays within
`SFT_MAX_LENGTH × SFT_BATCH_SIZE` shifted input tokens. Keep `drop_last: true`.
Use `SFT_SHUFFLE=false` only for an intentionally ordered stream.
SFT uses cross-entropy with token-mean loss and no weight sync; omit `rollout`
and `training.weight_sync`. See [SFT data](../../developer-guide/03-data-loader/sft-data.md)
for custom renderers, masking, and additional data files.
