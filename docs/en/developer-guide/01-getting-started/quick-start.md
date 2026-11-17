English | [简体中文](../../../zh/developer-guide/01-getting-started/quick-start.md) | [Documentation index](../../index.md)

<a id="quick-start-guide"></a>

# Quick Start Guide

Run the NorthRL driver on a local CPU machine. Weaver performs model training and sampling remotely. This guide uses the [math recipe](../../../../recipe/math/weaver.yaml); for a small bundled dataset and a ready-to-use local shell script, start with [SFT Quick Start](../../recipe/sft_quickstart/README.md).

<a id="prerequisites"></a>

## Prerequisites

- Python 3.12+, a writable output directory, and the dependencies required by your recipe.
- A Weaver endpoint, API key, and permission to train your selected model.
- Local access to the recipe's data and tokenizer, or network access to download the tokenizer.

<a id="installation"></a>

## Installation

From a new checkout on Linux or macOS:

```bash
git clone https://github.com/beipo-ai/NorthRL.git
cd NorthRL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname -s)" = Linux ]; then
  python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -e ".[core]" pyarrow
```

The `core` extra includes the training runtime and Weaver SDK. `pyarrow` reads the math recipe's Parquet files. Install any additional dependencies your task requires in this same environment.

<a id="run-locally"></a>

## Run Locally

The math recipe expects `math/level{1..5}_train.parquet` and matching `level{1..5}_test.parquet` files below `NORTHRL_DATA_PATH`. Choose a model available to your Weaver account in `models.actor.name`. Replace the placeholders and run from the repository root in Bash:

```bash
export EXPERIMENT_PATH="$(pwd)/logs/math-$(date +%Y%m%d-%H%M%S)"
export NORTHRL_DATA_PATH="/path/to/your/data"
export WEAVER_BASE_URL="https://your-weaver-host"
export WEAVER_API_KEY="your-weaver-api-key"
mkdir -p "$EXPERIMENT_PATH"
bash run.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]'
```

`run.sh` uses the exported variables above and starts `northrl.main` with the selected YAML. It does not automatically load `.env`. To load an environment file, use `bash run.sh --env-file /path/to/env.sh recipe/math/weaver.yaml`; see [Environment Files](configuration-setup.md#environment-files). The script uses `python` from your active environment and defaults `NORTHRL_PATH` to the repository root if it is unset or empty. Install dependencies beforehand. The console-only override avoids requiring W&B credentials.

<a id="minimal-setup-for-a-new-task"></a>

## Use Your Own Task

Start from a complete recipe that matches your workload. Update its model, data, reward, and training settings, then run `bash run.sh /path/to/recipe.yaml`. If using `--env-file`, place it before the recipe path. For E2B tasks, prepare the template, task hooks, NorthGate route, and service credentials described in [task integration](../05-rollout-workers/e2b-task-integration.md). The [minimal E2B walkthrough](../../recipe/e2b_minimal/README.md) provides task data, a complete training configuration, and local launch commands.

<a id="verify-installation"></a>
<a id="monitoring"></a>

## Logs and Results

Training output appears in the terminal. To also save it, replace the launch command above with:

```bash
set -o pipefail
bash run.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]' 2>&1 | tee northrl.log
```

This saves console output to `northrl.log` in the current directory. `set -o pipefail` makes the pipeline fail when Python fails. Check the command's exit status and `NorthRL training completed`; `run.sh` does not write an `exit_code` file. The separate SFT script does write one.

NorthRL saves the effective configuration as `train_config_resolved.yaml` in the experiment directory. Checkpoint metadata, when saving is enabled, records service-side paths; model weights remain on Weaver. Keep the driver running for the training session, for example in `tmux`.

<a id="next-steps"></a>

## Next Steps

- Configure service access, data, and logging in [Local Configuration](configuration-setup.md).
- Read the [math recipe guide](../../recipe/math/README.md) or [Core Architecture](../02-core-architecture/overview.md).
