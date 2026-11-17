# JobBench on E2B

This recipe runs the 499 JobBench document and office-work tasks in the
`jobbench-opencode-20261028` template (`hnon9xcowgs6fhshzol4`), derived from
`jobbench-claude-codex-202610` (`1a0jj5frcq30uj7pdng7`). It preserves the
591,780,875 bytes of required task inputs, document-processing tools,
Claude Code 2.1.285 and Codex 0.160.0, and preinstalls **OpenCode V2 2.2.0** at
`/usr/local/bin/opencode`. The policy uses NorthRL's OpenAI-compatible relay. The independent
rubric judge continues to use Claude Code and its own Anthropic-compatible endpoint.

Only task metadata and instructions are uploaded for each rollout. Inputs are
copied from the template cache into `/workspace/task`; new deliverables belong
in `/workspace/output`. Tasks that require updating a supplied database or other
input modify its working copy in `/workspace/task`. Both directories are graded.
The template cache and original source dataset are preserved.

`northrl/rollout_entrypoints/jobbench/entrypoint.py` implements `prepare`, `run_agent`
and `evaluate`. The same package contains `opencode.py` for CLI configuration and
output parsing, `references.py` for reference files, and `search.py` for Serper.
It loads and validates only the selected task; dataset conversion and
whole-dataset checks belong to offline tooling outside NorthRL.

## Use an offline-prepared dataset

Prepare the dataset once, offline, using local tooling outside this repository.
Point `NORTHRL_JOBBENCH_DATA_DIR` at the resulting directory on the driver host:

```text
jobbench-northrl/
  jobbench.jsonl
  evaluation/<package-id>/evaluation.json
  evaluation/<package-id>/RUBRICS.json
```

Training reads these prepared files directly. There is no conversion step at
training startup or inside a rollout, and the driver does not need the original
`packages` directory. The template must contain the same input bytes recorded
in the prepared dataset. Regenerate the affected records offline when inputs,
rubrics, template selection or the agent turn limit change. Existing prepared
rows must select the OpenCode template in their `template` field; selecting the
older Claude/Codex template fails preparation because OpenCode is not installed.

The JSONL contains `template`, `task_id`, `data_source`, `instruction`, `task_key`,
`task_sha256`, `rubrics_sha256` and `max_turns`. It contains no rubric text,
`task_card.md`, sample answers or optional search-reference files. Input and
rubric digests bind each public row to its template inputs and private grader
assets. No task attachments need to be committed to NorthRL or uploaded per rollout.

The private `evaluation` directory remains on the host during agent execution.
NorthRL uploads only the selected task's assets after the agent stage has ended
and its processes have been stopped.

## Configure and launch

Run the driver directly in a prepared NorthRL Python environment with the recipe's
runtime dependencies. Use the normal
NorthRL Weaver/NorthGate and E2B host credentials and a model sampler that supports
the OpenAI-compatible interface. Configure the **independent evaluator** with an
Anthropic-compatible endpoint and a model that can inspect documents, images
and structured files:

```bash
export NORTHRL_PATH=/absolute/path/to/NorthRL
export NORTHRL_JOBBENCH_DATA_DIR=/data/jobbench-northrl
export NORTHRL_JOBBENCH_ACTOR=your-weaver-model
export NORTHRL_JOBBENCH_TOKENIZER=/absolute/path/to/tokenizer

export NORTHRL_JOBBENCH_REFERENCE_VOLUME=your-reference-volume
export JOBBENCH_REFERENCES_ROOT=/mnt/data/jobbench/references/your-release
export JOBBENCH_REFERENCES_MANIFEST_SHA256=your-64-character-manifest-sha256
export SERPER_API_KEY=your-serper-key

export JOBBENCH_JUDGE_BASE_URL=https://your-judge-endpoint
export JOBBENCH_JUDGE_API_KEY=your-judge-key
export JOBBENCH_JUDGE_MODEL=your-judge-model
export NORTHRL_E2B_EVALUATOR_ENV=JOBBENCH_JUDGE_BASE_URL,JOBBENCH_JUDGE_API_KEY,JOBBENCH_JUDGE_MODEL

# Source after private exports; existing values are preserved.
source "$NORTHRL_PATH/recipe/jobbench/env.sh"

export TRAIN_CONFIG="$NORTHRL_PATH/recipe/jobbench/grpo.yaml"
export EXPERIMENT_PATH=/absolute/path/to/jobbench-experiment
mkdir -p "$EXPERIMENT_PATH"
cd "$NORTHRL_PATH"
python -m northrl.main --config-path "$NORTHRL_PATH/recipe/jobbench" --config-name grpo
```

Provide the normal `E2B_API_KEY`, `WEAVER_API_KEY`, `NORTHGATE_API_KEY` and any
site-specific E2B settings through the host launch environment. The template and
reference volume must be accessible to that account on the configured E2B service;
a volume on another service endpoint is not interchangeable. The recipe runs one
training step with four rollouts per sample. Server-side NorthGate owns tokenizer/
template/parser settings; this recipe does not configure deprecated E2B parsers.

`env.sh` sets the default `SERPER_API_URL=https://google.serper.dev/search` and
adds the reference and search variable names to `NORTHRL_E2B_AGENT_ENV`.
The driver forwards their values only to the root agent-stage command, which
prepares the sandbox and starts the non-root policy. They are not serialized
into recipe/task payloads or passed to bootstrap or evaluation. The policy itself
receives the local relay credentials and optional Serper settings.

Judge variables are forwarded only to the evaluation phase using
`NORTHRL_E2B_EVALUATOR_ENV`. They are not dataset fields and are not passed to the
policy agent. OpenCode selects `localrelay/northrl`, using the local relay's `/v1`
address and credentials from `model_environment`; the judge uses its own model
and endpoint. Judge requests are not part of the policy's training trace.

For installation and local logging, see the [Quick Start](../../docs/en/developer-guide/01-getting-started/quick-start.md#run-locally). Keep host credentials in your private launch environment.

Execution budgets are configured under `rollout.execution`:

| Setting | Default in this recipe |
| --- | ---: |
| `preparation_timeout_seconds` | 600 |
| `agent_timeout_seconds` | 3600 |
| `evaluation_timeout_seconds` | 1800 |

These are separate phase budgets. `max_turns` maps to OpenCode's `steps` limit;
OpenCode reserves the last step for a summary without tools, so this is not
identical to Claude Code's turn limit. The judge has a 60-turn cap. The framework
handles wall-clock timeouts and cleanup before uploading private assets. It still
evaluates available files after an agent failure or timeout. Do not add an
`agent_timeout_sec` dataset field.

## References, search and the OpenCode runtime

The recipe mounts `NORTHRL_JOBBENCH_REFERENCE_VOLUME` at `/mnt/data`. Set
`JOBBENCH_REFERENCES_ROOT` to an uploaded release containing `manifest.json` and
`references/<task_key>/files_required_to_search`, and pin the manifest's SHA256
with `JOBBENCH_REFERENCES_MANIFEST_SHA256`. Required task inputs remain in the
template; only external reference files come from this release.

Before starting the policy, preparation makes the **entire `/mnt/data` mount**
read-only using a bind remount and verifies the mount flags. File permissions
alone do not enforce this on the shared NFS service. It verifies the manifest
and only the selected task's file hashes, then includes the actual reference
directory in the prompt. The 61 tasks recorded without external files are valid
and receive no reference directory. A configured missing release, checksum
mismatch or failed read-only remount stops preparation instead of silently
continuing. This protects the mount in each sandbox; it does not change the
volume service's global access policy. Leave both reference variables unset and
remove the recipe's volume mapping only when intentionally running without
references.

When `SERPER_API_KEY` is set, the policy prompt explains how to use the shell tool:

```bash
python3 /opt/northrl-jobbench/search.py 'search query' --num 5
```

The helper returns source URLs and snippets; the agent must fetch source content
when the task requires it. Native OpenCode web search is disabled. A missing key
is explicitly reported in the prompt. `SERPER_API_URL` may select a trusted
Serper-compatible HTTPS endpoint; requests send the key in `X-API-KEY`, refuse
redirects and use a 30-second timeout. No search credentials are provided to the
judge.

OpenCode is installed once while building the template, using the fixed Linux
x86_64 baseline **2.2.0** package after verifying its pinned SHA512. Rollouts
use the template's fixed `/usr/local/bin/opencode` path; they do not download or
extract an installation archive. The binary must report version `2.2.0`, and it and its path
must be root-owned, non-writable by other users and free of symlinks. A missing,
non-executable or wrong-version binary fails preparation with an actionable
error. The small OpenCode configuration is generated per rollout because the
step limit comes from the current task.

## Evaluation contract

The source distribution supplies `RUBRICS.json`, not an executable official
JobBench grader. This integration defines a **NorthRL rubric-judge reward**;
it should not be presented as an official JobBench score.

A separate Claude Code process inspects the original inputs, an immutable
snapshot of both candidate directories, and the original task rubric. It can
use the template's document libraries, query SQLite, inspect spreadsheet cells,
and render/read visual output. It must cite evidence and mark a rubric passed
only when all that rubric's criteria are satisfied. The evaluator validates
exact coverage of rubric IDs, boolean decisions and evidence fields, then
computes the reward itself:

```text
reward = sum(rubric.weight * rubric.passed) / sum(rubric.weight)
```

Missing or incorrect deliverables can earn zero. Missing judge credentials,
model/CLI errors, mismatched assets or malformed assessments are environment
errors, not fabricated zero rewards. Scores depend on the chosen judge model
and rubric interpretation; pin the judge configuration for comparable runs.

The agent and judge run under separate non-root accounts. The template's
sudo-enabled default `user` account is not used. Preparation also removes group/other
write permissions from the template's root-owned Python runtime before starting
the agent: the published image has writable files under `/usr/local`.
Judge input snapshots reject
symlinks and special files; skipped paths are reported to the judge. Candidate
content is treated as untrusted evidence, not evaluator instructions. As with
other LLM judges, prompt-injection robustness is not an absolute guarantee.

The sandbox retains the detailed report at
`/opt/northrl-jobbench/private/evaluation.json`; `EvaluationResult` returns reward,
passed/rubric counts, total weight and agent exit code. `AgentResult.artifacts`
references the two working directories. These references do not automatically
download files before sandbox teardown; configure external artifact collection
if you need to retain the generated documents.

## Tests

```bash
python -m pytest tests/unittests/remote_rollout/test_jobbench*.py
```
