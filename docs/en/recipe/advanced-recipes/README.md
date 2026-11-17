English | [简体中文](../../../zh/recipe/advanced-recipes/README.md) | [Documentation index](../../index.md)

<a id="overview"></a>

# Resource requirements for advanced recipes

This page collects resource requirements, shared environment settings, and local
launch instructions for advanced recipes. Dataset formats, template requirements,
and evaluation rules already covered by a recipe README are linked rather than
repeated here. For installation, use the
[Quick Start](../../developer-guide/01-getting-started/quick-start.md).

These recipes provide training configurations and task integrations. They do not
include the required datasets, evaluation packages, or access to task templates
and volumes. For a first end-to-end run, you can start with the
[minimal agent example](../e2b_minimal/README.md).

| Recipe | Main resources to prepare | Configuration and details |
| --- | --- | --- |
| GRPO R3 | Training and validation datasets; a Weaver model with R3 support | [Resource list](#grpo-r3) |
| SWEbench | Task data, templates with task repositories and Claude Code, hidden tests | [Resource list](#swebench) |
| TMax | Task data, per-task templates, hidden tests, a Claude Code volume | [Resource list](#tmax) · [Task README (Chinese)](../../../../recipe/tmax/README.md) |
| ATC v3 | Task data, templates with the ATC runtime, an asset volume, hidden evaluation packages | [Resource list](#atc-v3) · [Task README](../../../../recipe/atc_v3/README.md) |
| JobBench | Task data, templates with tools and inputs, evaluation packages, a reference volume, an independent judge | [Resource list](#jobbench) · [Task README](../../../../recipe/jobbench/README.md) |
| MiMo | Data, templates, and evaluation assets for the selected task type; an independent judge for general tasks | [Resources for all four task types](#mimo) |

<a id="shared-configuration"></a>

## Shared configuration

| Variable | Purpose |
| --- | --- |
| `WEAVER_BASE_URL`, `WEAVER_API_KEY` | Weaver service address and credentials. |
| `EXPERIMENT_PATH` | Local output directory; use a new directory for each independent experiment. |
| `NORTHRL_PATH` | Absolute NorthRL checkout path; E2B recipes use it to locate task entrypoints. |
| `E2B_API_KEY` | E2B credentials with access to the task templates and any required volumes. |
| `NORTHGATE_API_KEY` | NorthGate credentials for E2B training rollouts. |

The last three entries apply to the E2B recipes on this page; GRPO R3 does not
require E2B resources. For a custom E2B service, also configure
`E2B_DOMAIN` / `E2B_API_URL` according to that service's connection instructions.
E2B sandboxes must be able to reach the NorthGate sampling address published by Weaver.

The root `run.sh` defaults `NORTHRL_PATH` to its repository root before loading
an environment file, so the file can use it to construct paths. A supplied value
is preserved unless the file overrides it. When using `python -m northrl.main`
directly, export `NORTHRL_PATH` yourself.

Model names must identify models available to your Weaver account. Tokenizers
must match the trained model and can be Hugging Face IDs or local directories
readable by NorthRL. A service model alias is not necessarily a tokenizer ID.
When changing models, check the configured context length, sampling options,
fine-tuning support, and NorthGate codec as well.

Keep credentials and resource paths in a private local environment file selected
with `--env-file`. [`run.sh`](../../../../run.sh) sources and exports those
variables before starting Python. Without an explicit file selection, it uses
the current exported shell variables; it does not automatically load `.env` or
`recipe/<name>/env.sh`. The existing `NORTHRL_ENV_FILE` selector is also supported,
with `--env-file` taking precedence. The script uses `python` from your current
`PATH`, or the interpreter selected with `PYTHON`; it does not automatically
select the repository's `.venv`. If a recipe needs an environment helper,
explicitly source it from your private file, as shown for JobBench below.
See [environment files](../../developer-guide/01-getting-started/configuration-setup.md#environment-files)
for loading rules and [the first-run example](#first-run) for a complete setup.
Do not commit credentials to Git.

<a id="resource-locations"></a>

## Where files, templates, and volumes live

| Configuration | Location | Example |
| --- | --- | --- |
| Training data, local tokenizer | The machine running NorthRL | `/data/tmax/train.jsonl` |
| `entrypoint.target` | An absolute path to the task Python file on that machine | `/path/to/NorthRL/northrl/rollout_entrypoints/tmax_claude_code_rollout.py` |
| `entrypoint.task_files_root` | An absolute path to the hidden evaluation root on that machine | `/data/tmax/evaluation` |
| Dataset row's `template` | A task template ID or alias on the E2B service | `your-task-template` |
| `entrypoint.workspace`, volume mount points | Inside the E2B sandbox | `/home/user`, `/mnt/ccbin` |
| Value mapped to each mount point in `volume_mounts` | An E2B volume name or ID, not a local path | `your-claude-code-volume` |

Local paths must be readable by the NorthRL process; setting a path does not
mount another machine's filesystem. Hidden evaluation assets live under
`<task_files_root>/<task_id>/` and are uploaded only after the agent stage ends.
Task evaluation directories and their contents must not be symlinks. Keep hidden
tests out of agent-visible templates and volumes. Consult each recipe README
for its exact file layout.

E2B rollouts select a template from each dataset row's `template` field. Setting
`E2B_TEMPLATE_ID` does not override templates in existing data; the minimal example
uses that variable only when generating its dataset. See the
[integration guide](../../developer-guide/05-rollout-workers/e2b-task-integration.md)
for path and dataset rules.

<a id="sandbox-environment"></a>

## Forwarding environment variables to sandboxes

The entire local environment is not forwarded to the sandbox. For additional
tools or judge services, set their values in your private environment file, then
list the variable names in these comma-separated selectors:

| Selector | Scope |
| --- | --- |
| `NORTHRL_E2B_AGENT_ENV` | The agent-stage command, including `prepare` / `run_agent`, but not evaluation. The task entrypoint decides which values reach agent subprocesses. |
| `NORTHRL_E2B_EVALUATOR_ENV` | Evaluation only, such as an independent judge's endpoint, credentials, and model name. |

The JobBench and MiMo general sections include selector examples. For calls to
the model being trained, agents should use the task interface's
`model_environment`. Replacing it with a separate model endpoint and key bypasses
the training trajectory collection path.

<a id="grpo-r3"></a>

## GRPO with R3 router replay

Configuration: [`grpo_r3_northrl.yaml`](../../../../recipe/grpo_r3_northrl/grpo_r3_northrl.yaml).

| Variable | Resource |
| --- | --- |
| `NORTHRL_TRAIN_DATA_PATH` | Required training JSONL path. |
| `NORTHRL_R3_MATH500_DATA` | Required local Math500 validation dataset path. |
| `NORTHRL_R3_AIME2024_DATA` | Required local AIME 2024 validation dataset path. |
| `NORTHRL_R3_AIME2025_DATA` | Required local AIME 2025 validation dataset path. |
| `NORTHRL_R3_AIME2026_DATA` | Required local AIME 2026 validation dataset path. |
| `NORTHRL_R3_MODEL` | Weaver model name; defaults to `Qwen/Qwen3-30B-A3B-Base`. |
| `NORTHRL_R3_TOKENIZER` | Matching tokenizer; defaults to `Qwen/Qwen3-30B-A3B-Base`. |

Validation is enabled by default. Supply all four validation datasets or edit
your configuration copy to select the datasets you have. To disable validation,
set `validate.enabled: false` and `validate.data: null` so unused
environment-variable interpolations are not resolved. Dataset rows must follow
the configured prompt and reward format; see [data configuration](../../developer-guide/11-configuration-reference/data-config.md)
and the [single-turn task guide](../math/README.md).

The service must support R3 router replay for the selected model. Changing the
model name alone does not enable that capability. The supplied
[`env.sh`](../../../../recipe/grpo_r3_northrl/env.sh) contains empty local settings;
populate your private copy before loading it, and keep credentials out of Git.

<a id="swebench"></a>

## SWEbench with Claude Code

Configuration: [`e2b_claude_grpo.yaml`](../../../../recipe/swebench/e2b_claude_grpo.yaml).

| Required variable | Resource |
| --- | --- |
| `NORTHRL_SWEBENCH_MODEL` | A model registered on your Weaver service. |
| `NORTHRL_SWEBENCH_TOKENIZER` | The matching tokenizer ID or local directory. |
| `NORTHRL_SWEBENCH_TRAIN_DATA` | Training dataset with `task_id`, `template`, and `instruction` fields. |
| `NORTHRL_SWEBENCH_TASK_FILES_ROOT` | Root of the per-task hidden evaluation packages. |

Each row's `template` must select an E2B template accessible to your API key.
The [task entrypoint](../../../../northrl/rollout_entrypoints/swerebench_claude_code_rollout.py)
expects exactly one top-level Git checkout in the sandbox and Claude Code at
`/usr/local/bin/claude`. A Node-based Claude installation must also provide its
Node executable. Runtime files must satisfy the entrypoint's root ownership and
write-protection checks.
The default domain restrictions also require `nft`, `setpriv`, and the ability
to install the corresponding network rules inside the sandbox.

Prepare `<NORTHRL_SWEBENCH_TASK_FILES_ROOT>/<task_id>/` with `run-tests.sh`,
`tests/config.json`, `log_parsers.py`, and `swe_constants.py` for that task.
These files are supplied to evaluation after the agent finishes. Template build
sources, task checkouts, datasets, and evaluation packages are not bundled here;
the recipe is not ready to run until you supply them.

<a id="tmax"></a>

## TMax terminal tasks

Configuration: [`train.yaml`](../../../../recipe/tmax/train.yaml).

| Required variable | Resource |
| --- | --- |
| `NORTHRL_TMAX_MODEL` | A model registered on your Weaver service. |
| `NORTHRL_TMAX_TOKENIZER` | The matching tokenizer ID or local directory. |
| `NORTHRL_TMAX_TRAIN_DATA` | JSONL with `task_id`, `template`, and `instruction`; optional `band` metadata. |
| `NORTHRL_TMAX_TASK_FILES_ROOT` | Evaluation packages at `<task_id>/run-tests.sh`. |
| `NORTHRL_TMAX_CC_VOLUME` | Your accessible E2B volume containing an executable `claude` at its root. No default volume is supplied. |

The [TMax README](../../../../recipe/tmax/README.md) covers the template's user,
workspace and dependencies, Claude Code volume setup, dataset examples,
evaluation directory layout, and result-block format. The E2B deployment must
support this recipe's volume mounts and unmounts. For a text-only model, exclude
tasks requiring image or video input.

Defaults are 32 tasks × 16 rollouts per step, 512 workers, and 109 steps. Start
with the [small-run settings below](#first-run), then increase scale according
to your dataset and service capacity.

<a id="atc-v3"></a>

## ATC v3

Configuration: [`e2b_claude_grpo.yaml`](../../../../recipe/atc_v3/e2b_claude_grpo.yaml).

| Required variable | Resource |
| --- | --- |
| `NORTHRL_ATC_MODEL` | A model registered on your Weaver service. |
| `NORTHRL_ATC_TOKENIZER` | The matching tokenizer ID or local directory. |
| `NORTHRL_ATC_TRAIN_DATA` | Prepared ATC dataset with per-task template and asset metadata. |
| `NORTHRL_ATC_EVALUATION_ROOT` | Hidden `manifest.json` and `verifier.py` under each `<task_id>/`. |
| `NORTHRL_ATC_VOLUME` | Your accessible task-asset volume, mounted at `/mnt/data`. |

See the [ATC README](../../../../recipe/atc_v3/README.md) for the dataset layout,
how `asset_bundles` maps to files in the volume, and the template's ATC runtime
interface and permissions. Supply that runtime, its MCP implementations, and
the evaluation resources; a generic E2B template is not sufficient.

<a id="jobbench"></a>

## JobBench

Configuration: [`grpo.yaml`](../../../../recipe/jobbench/grpo.yaml).

| Variable | Resource |
| --- | --- |
| `NORTHRL_JOBBENCH_ACTOR` | A trainable model on your Weaver service. |
| `NORTHRL_JOBBENCH_TOKENIZER` | Matching tokenizer; defaults to the model name, which must also be a valid tokenizer ID if this variable is omitted. |
| `NORTHRL_JOBBENCH_DATA_DIR` | Parent directory containing the prepared `jobbench.jsonl` and `evaluation/`. |
| `NORTHRL_JOBBENCH_REFERENCE_VOLUME` | Reference volume mounted by the current YAML. |
| `JOBBENCH_REFERENCES_ROOT`, `JOBBENCH_REFERENCES_MANIFEST_SHA256` | Reference release directory inside the sandbox and its manifest SHA256. |
| `JOBBENCH_JUDGE_BASE_URL`, `JOBBENCH_JUDGE_API_KEY`, `JOBBENCH_JUDGE_MODEL` | Independent judge's Anthropic-compatible endpoint, credentials, and model name. |
| `SERPER_API_KEY` | Optional; enables search for the agent. Without it, this search capability is disabled. |

After setting these values in your private environment file, add:

```bash
NORTHRL_E2B_EVALUATOR_ENV=JOBBENCH_JUDGE_BASE_URL,JOBBENCH_JUDGE_API_KEY,JOBBENCH_JUDGE_MODEL
source "$NORTHRL_PATH/recipe/jobbench/env.sh"
```

The helper preserves existing values, sets the default search URL, and adds
reference and search variables to the agent-stage selector. Merge the required
names if you already have an evaluator selector. Judge requests are separate
from the policy model's training trace. To intentionally run without references,
remove the volume mapping and unset reference variables together; leaving the
required volume name empty is not sufficient.

See the [JobBench README](../../../../recipe/jobbench/README.md) for dataset and
evaluation formats, OpenCode template requirements, reference layout, and scoring
rules. Specific template names and IDs mentioned there do not grant public
access; dataset rows must select compatible templates accessible to your account.

<a id="mimo"></a>

## MiMo

All four configurations use `NORTHRL_MIMO_ACTOR` (Weaver model name),
`NORTHRL_MIMO_TOKENIZER` (matching tokenizer), and `NORTHRL_MIMO_DATA_DIR` (local
data root). The tokenizer defaults to the model name; set it explicitly if the
service alias is not a valid tokenizer ID.

The table below lists resource layouts and entrypoints for all four task types.
Prepare only the files for your selected task type. Datasets, template build
sources, and hidden evaluation packages are not bundled with the repository.

| Configuration | JSONL under the data root | Evaluation package under the same root | Main template requirements |
| --- | --- | --- | --- |
| [code](../../../../recipe/mimo/code_grpo.yaml) | `code.jsonl` | `evaluation.json` and `test.patch` in `evaluation/code/<task_id>/` | Task Git checkout and test dependencies; [entrypoint](../../../../northrl/rollout_entrypoints/mimo/code.py). |
| [terminal](../../../../recipe/mimo/terminal_grpo.yaml) | `terminal.jsonl` | `evaluation.json` and `tests/test.sh` in `evaluation/terminal/<task_id>/` | Task-specific terminal environment and tools; [entrypoint](../../../../northrl/rollout_entrypoints/mimo/terminal.py). |
| [general](../../../../recipe/mimo/general_grpo.yaml) | `general.jsonl` | `evaluation.json`, `verify.py`, `verifier_meta.json`, `_helpers.py`, and `run_verify.py` in `evaluation/general/<task_id>/` | `/work/workspace`, task MCP services and their runtime dependencies; [entrypoint](../../../../northrl/rollout_entrypoints/mimo/general.py). |
| [arvo](../../../../recipe/mimo/arvo_grpo.yaml) | `arvo.jsonl` | The current configuration does not use a local `task_files_root` | Task binaries, agent / verifier users, and `/root/binary/run.sh`; [entrypoint](../../../../northrl/rollout_entrypoints/mimo/arvo.py). |

Dataset rows need `template`, `task_id`, `instruction`, `cwd`, and `agent_user`;
additional fields and evaluation settings follow the corresponding entrypoint.
`cwd` must be an existing absolute path inside the template, `agent_user` must
identify its non-root account, and Claude Code must be available at
`/opt/mimo-agents/claude`. These are task-specific templates; a bare Linux template
is not sufficient.

General tasks also require an independent judge. Add these settings to your
private environment file:

```bash
GA_JUDGE_URL="https://your-judge-endpoint"
GA_JUDGE_KEY="your-judge-key"
GA_JUDGE_MODEL="your-judge-model"
GA_JUDGE_API="chat"  # chat, responses, or gemini; must match the judge service
NORTHRL_E2B_EVALUATOR_ENV=GA_JUDGE_URL,GA_JUDGE_KEY,GA_JUDGE_MODEL,GA_JUDGE_API
```

These variables go only to evaluation; merge names into any existing selector.
E2B training sampling uses the server-side NorthGate codec, not
`rollout.parsers` or `execution.tokenizer`. See the
[integration guide](../../developer-guide/05-rollout-workers/e2b-task-integration.md).

<a id="first-run"></a>

## First small run

This TMax example assumes NorthRL's runtime dependencies are installed and the
task data, templates, and volume are ready. Save the following as a private file
at `/absolute/path/tmax.env`, replacing all example values:

```bash
EXPERIMENT_PATH="/absolute/path/to/new-tmax-run"
WEAVER_BASE_URL="https://your-weaver-host"
WEAVER_API_KEY="your-weaver-key"
NORTHGATE_API_KEY="your-northgate-key"
E2B_API_KEY="your-e2b-key"
# Custom E2B: add E2B_DOMAIN / E2B_API_URL per the service's connection instructions.

NORTHRL_TMAX_MODEL="your-weaver-model"
NORTHRL_TMAX_TOKENIZER="/absolute/path/to/matching-tokenizer"
NORTHRL_TMAX_TRAIN_DATA="/data/tmax/train.jsonl"
NORTHRL_TMAX_TASK_FILES_ROOT="/data/tmax/evaluation"
NORTHRL_TMAX_CC_VOLUME="your-claude-code-volume"
```

From the NorthRL repository root, run one step with one task, four rollouts per
task, and four workers:

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

If these variables are already exported in your shell, omit `--env-file` and
its path. Unset any previously exported `NORTHRL_ENV_FILE` to avoid loading that file.

Keep `algorithm.optim.batch_size=null` from the original recipe so the complete
rollout batch is processed in one optimizer update. `mini_batch_size=4` only
limits the number of Datums per training request. A multi-turn trajectory can
produce multiple Datums, so this is not a rollout count. Reducing concurrency and
steps does not reduce per-task timeouts; TMax's agent budget is still one hour.

This example uses console logging and requires no W&B configuration. `run.sh`
does not save terminal output; see the [logging instructions](../../developer-guide/01-getting-started/quick-start.md#monitoring)
to retain it. This configuration writes debug output to `EXPERIMENT_PATH` and
checkpoint metadata to its `ckpt/` directory; Weaver manages model weights.

Check the command's exit status, the `NorthRL training completed` log, and
training metrics. `training/skipped_flag=1` means the training update was skipped;
inspect `training/forward_backward_call_count` and rollout errors to identify why.
Even when training requests run, GRPO advantages can all be zero if every sample
for a task receives the same reward. This small run verifies the workflow; it
does not establish training effectiveness.

<a id="troubleshooting"></a>

## Common configuration problems

| Symptom | Check first |
| --- | --- |
| Missing environment variable or local file | Correct `--env-file` path or exported shell variables; populated values and readable data, tokenizer, and evaluation files. |
| Template not found or access denied | Dataset row's `template`, E2B service address, and current key's permissions. |
| Volume mount fails | A volume name / ID rather than a local path; volume and templates on the same E2B service; deployment support for the required mounts. |
| Missing evaluation package or symlink error | Matching `task_id` directory names and complete, regular evaluation files. |
| Missing judge configuration | Populated judge variables listed in `NORTHRL_E2B_EVALUATOR_ENV`. |
| Unreachable sampling endpoint or model mismatch | Sandbox access to Weaver's published NorthGate address; matching model permissions, tokenizer, and codec. |
| Run finishes without a training update | Rollout environment errors or missing training segments; skipped-update flag and training request count. |

A task that fails evaluation can be a valid zero-reward sample. Investigate
template, file, or service failures using the recipe's error rules rather than
inferring the cause from a zero reward alone. Actual internal paths, resource
names, and experiment settings can live in a private recipes repository; public
documentation uses replaceable example values.
