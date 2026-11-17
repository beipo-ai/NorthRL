English | [简体中文](../../../zh/developer-guide/01-getting-started/configuration-setup.md) | [Documentation index](../../index.md)

<a id="configuration-setup"></a>

# Local Configuration

NorthRL runs its driver in your local Python environment and calls remote services for model execution. Follow [Quick Start](quick-start.md#installation) to install the runtime and [Run Locally](quick-start.md#run-locally) to launch a complete recipe.

<a id="service-access"></a>

## Service Access

Set credentials in the shell that runs NorthRL:

```bash
export WEAVER_BASE_URL="https://your-weaver-host"
export WEAVER_API_KEY="your-weaver-api-key"
```

The selected `models.actor.name` must be available to your account with the requested LoRA or full fine-tuning capability. `models.actor.tokenizer` selects a local tokenizer directory or a downloadable model ID; check that it matches the model. Some recipes use the service-provided tokenizer when this field is null.

For E2B agent tasks, also obtain `E2B_API_KEY`, `NORTHGATE_API_KEY`, and access to the required sandbox template. Weaver and NorthGate must expose the selected sampler and its training token data. Task hooks, data, and template dependencies still need adaptation; see [task integration](../05-rollout-workers/e2b-task-integration.md).

<a id="data-and-output"></a>

## Data and Output

```bash
export NORTHRL_DATA_PATH="/path/to/your/data"
export EXPERIMENT_PATH="/path/to/a/new-experiment"
mkdir -p "$EXPERIMENT_PATH"
```

`NORTHRL_DATA_PATH` is used by the math recipes; other recipes may require different variables. Check every `${oc.env:...}` expression in your YAML. The driver must be able to read the data and write the output directory. Use a new experiment directory for each independent run.

Use `bash run.sh /path/to/recipe.yaml` to select the recipe. The script sets `TRAIN_CONFIG` and Hydra's configuration path to the same source YAML. Custom task modules and task-specific assets must be accessible locally.

<a id="environment-files"></a>

## Environment Files

Choose one of two ways to prepare the environment before starting a recipe.

**Load an environment file.** Put service credentials and local paths in a Bash file such as `/path/to/env.sh`:

```bash
WEAVER_BASE_URL="https://your-weaver-host"
WEAVER_API_KEY="your-weaver-api-key"
NORTHRL_DATA_PATH="/path/to/your/data"
EXPERIMENT_PATH="/path/to/a/new-experiment"
```

From the repository root, pass `--env-file` before the recipe path:

```bash
bash run.sh --env-file /path/to/env.sh recipe/math/weaver.yaml \
  'runtime.logger.backends=[console]'
```

The script sources this file and automatically exports its variables before starting Python. Its assignments override inherited values. The file must exist and be readable. Keep credentials out of Git. `--env-file=/path/to/env.sh` is also supported.

**Use variables already exported in your shell.** If your shell is already configured, launch directly:

```bash
bash run.sh recipe/math/weaver.yaml 'runtime.logger.backends=[console]'
```

Use `export NAME=value` for these variables. If you manually `source /path/to/env.sh`, its assignments must use `export`, or you can run `set -a; source /path/to/env.sh; set +a` to export them. Without an explicit file selection, `run.sh` reads no environment file, including the repository-root `.env`.

For existing commands, `NORTHRL_ENV_FILE=/path/to/env.sh bash run.sh recipe/math/weaver.yaml` remains supported as an explicit file selection. `--env-file` takes precedence over `NORTHRL_ENV_FILE`. Unset `NORTHRL_ENV_FILE` when switching to an already prepared shell environment.

Relative recipe and environment-file paths are resolved from the directory where you run the command. Before loading the selected file, the script defaults `NORTHRL_PATH` to the repository root containing `run.sh`, preserving a supplied value. The file can use this variable to construct paths or override it; if the final value is unset or empty, the repository root is used. The training YAML keeps its `${oc.env:...}` references. Place Hydra overrides after the recipe path, as shown above.

The script uses `python` on your current `PATH`, including an activated virtual environment; set `PYTHON=/path/to/python` to choose another interpreter. It does not automatically select the repository's `.venv`, install dependencies, or save console output. The separate [SFT quickstart script](../../recipe/sft_quickstart/README.md) retains its own `.env` loading, interpreter selection, preflight checks, and logging; run it directly.

<a id="logging"></a>

## Logging

Pass `'runtime.logger.backends=[console]'` after the recipe path for terminal logging. To use W&B, configure the account and enable its backend:

```bash
export WANDB_KEY="your-wandb-api-key"
export WANDB_HOST="https://api.wandb.ai"
```

Then pass `'runtime.logger.backends=[console,wandb]'`. The [Quick Start logging example](quick-start.md#monitoring) uses `tee` to save console output. Effective recipe configuration and checkpoint metadata are stored under `EXPERIMENT_PATH`; Weaver manages the model weights remotely.

<a id="publishing-images"></a>

## Optional Image Publishing

To publish a runtime image to Docker Hub from this repository, set the GitHub Actions repository variable `NORTHRL_IMAGE_REPOSITORY` to the destination repository without a tag. Configure the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets. The release workflow uses a GitHub-hosted runner and fails its destination check if `NORTHRL_IMAGE_REPOSITORY` is unset.

<a id="optional-issue-automation"></a>

## Optional Issue Automation

The issue-linking and milestone workflows use the repository's `GITHUB_TOKEN` by default and limit writes to this repository. Cross-repository automation is optional: configure `ISSUE_AUTOMATION_TOKEN` with access to the intended repositories to enable it. The existing `PROJECT_TOKEN` secret remains supported as a fallback. Neither an organization Project nor a separate feedback or knowledge-base repository is required.

<a id="troubleshooting"></a>

## Troubleshooting

- **Missing environment variable:** export it in your shell or set it in a file passed with `--env-file`; check the recipe's `${oc.env:...}` references. A nearby `.env` or `env.sh` is not loaded automatically.
- **Missing Python module:** install the `core` extra and task-specific dependencies in the same environment used to run the driver.
- **Unavailable model or unauthorized request:** check the Weaver endpoint, key, model identifier, and training permission.
- **Tokenizer or data not found:** check local paths and download access; a remote training service does not make local files available automatically.
- **No saved console log:** the direct entry point prints to the terminal; use the `tee` command in [Quick Start](quick-start.md#monitoring).
