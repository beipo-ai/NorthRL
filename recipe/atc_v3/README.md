# ATC v3 on E2B

Resource setup: [English](../../docs/en/recipe/advanced-recipes/README.md#atc-v3) | [简体中文](../../docs/zh/recipe/advanced-recipes/README.md#atc-v3)

This recipe requires an ATC dataset, task asset archives, and an E2B template with
Claude Code and a compatible ATC runtime already installed. You must supply these
resources in your own deployment; this repository does not publish a prepared
dataset, volume, or template.

NorthRL only provides the task hooks and training configuration. Dataset conversion,
archive publication, template builds, MCP implementations, the verifier context,
and standalone acceptance tools are maintained outside this repository.

## Prepared artifacts

The process running NorthRL needs the following external files:

```text
atc-v3-delivery/
  dataset.jsonl
  private_eval/<task_id>/manifest.json
  private_eval/<task_id>/verifier.py
```

Each dataset row contains `task_id`, `template`, `data_source`, `instruction`,
`asset_path`, `asset_bundles`, `task_sha256`, and `agent=claude_code`.
`asset_bundles` identifies immutable vePFS archives and the exact members needed
for the task, with sizes and SHA256 digests. `asset_path` carries task/release
identity; it does not require an expanded directory on vePFS. A Parquet index
can encode `asset_bundles` as JSON text.

The template must install the trusted runtime package at
`/opt/atc-v3-runtime/atc_v3_runtime`, expose `API_VERSION = 1`, and provide the
`prepare`, `run_agent`, and `evaluate` functions used by the entrypoint. The
package and its parent directories must be root-owned and protected against
writes by other users. Task databases, MCP source and schemas are materialized
from vePFS into each sandbox's private local storage. Codex CLI may also be
installed, but this recipe uses Claude Code with the existing model gateway.

The runtime stops the agent and all MCP processes and snapshots their state
before returning from `run_agent`. NorthRL then uploads only the selected task's
private evaluation files and invokes `evaluate`. Private verifiers, reference
trajectories and final databases must not be published as public task assets.

## Run

Run the driver directly in a prepared NorthRL Python environment with the recipe's
runtime dependencies. Load the usual E2B, Weaver and NorthGate credentials,
then set the artifact paths in the same shell:

```bash
export NORTHRL_PATH=/absolute/path/to/NorthRL
export NORTHRL_ATC_TRAIN_DATA=/data/atc-v3-delivery/dataset.jsonl
export NORTHRL_ATC_EVALUATION_ROOT=/data/atc-v3-delivery/private_eval
export NORTHRL_ATC_VOLUME=your-atc-assets-volume
export NORTHRL_ATC_MODEL=your-weaver-model
export NORTHRL_ATC_TOKENIZER=/absolute/path/to/tokenizer
export TRAIN_CONFIG="$NORTHRL_PATH/recipe/atc_v3/e2b_claude_grpo.yaml"
export EXPERIMENT_PATH=/absolute/path/to/new-experiment
mkdir -p "$EXPERIMENT_PATH"
cd "$NORTHRL_PATH"
python -m northrl.main --config-path "$NORTHRL_PATH/recipe/atc_v3" --config-name e2b_claude_grpo
```

The dataset selects a compatible template that your E2B API key can access. The
recipe mounts your asset volume at `/mnt/data`; the deployment must support these
volume mounts, and the mounted archive layout must match `asset_bundles`.
`NORTHRL_ATC_MODEL` is a model registered on your Weaver service, and
`NORTHRL_ATC_TOKENIZER` must match that model. Configure proxies for your network;
add service addresses to `NO_PROXY`/`no_proxy` only when they need a direct
connection. Remote training and E2B sandboxes still use their configured services.

For installation and local logging, see the [Quick Start](../../docs/en/developer-guide/01-getting-started/quick-start.md#run-locally).

## Outcomes

Valid verifier scores, including zero, remain scored rollouts. Verifier exceptions,
unsupported contracts, timeouts and malformed scores are logged and receive
reward `0.0` with the existing `task_failed` status. These rollouts retain their
recorded training segments and participate as normal zero-reward samples.
Sandbox, asset, MCP and evidence failures produce `env_error` and are excluded
from reward training.

The external verifier context is reconstructed from the dataset, not the original
SDK. Validate the verifier contract and scores against your own prepared tasks
before training; this repository does not certify externally supplied task assets.
