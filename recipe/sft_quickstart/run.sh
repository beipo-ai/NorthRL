#!/usr/bin/env bash
# Run the SFT driver locally; Weaver provisions and runs the GPU trainer.
set -euo pipefail
umask 077

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash recipe/sft_quickstart/run.sh [--check] [Hydra overrides...]
Reads .env from the NorthRL root (override with NORTHRL_ENV_FILE).
--check validates the dataset and tokenizer without contacting Weaver.
Uses .venv/bin/python if present, otherwise python (override with PYTHON).
HELP
    exit 0
fi
CHECK=false
if [[ ${1:-} == --check ]]; then CHECK=true; shift; fi
ENV_FILE=${NORTHRL_ENV_FILE:-$ROOT/.env}
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # This is the user's own shell environment file.
    source "$ENV_FILE"
    set +a
fi
: "${WEAVER_BASE_URL:?Set WEAVER_BASE_URL in .env}"
: "${WEAVER_API_KEY:?Set WEAVER_API_KEY in .env}"
: "${SFT_DATA_PATH:?Set SFT_DATA_PATH in .env}"
if [[ "$WEAVER_API_KEY" == replace-with-your-weaver-api-key ]]; then
    echo "Replace the placeholder WEAVER_API_KEY in .env" >&2
    exit 1
fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
else
    PYTHON=${PYTHON:-python}
fi
if [[ "$SFT_DATA_PATH" != /* ]]; then SFT_DATA_PATH="$ROOT/$SFT_DATA_PATH"; fi
export WEAVER_BASE_URL WEAVER_API_KEY SFT_DATA_PATH
export TRAIN_CONFIG="$ROOT/recipe/sft_quickstart/weaver.yaml"
export EXPERIMENT_PATH=${EXPERIMENT_PATH:-$ROOT/logs/sft-$(date -u +%Y%m%dT%H%M%SZ)-$$}
if [[ "$EXPERIMENT_PATH" != /* ]]; then export EXPERIMENT_PATH="$ROOT/$EXPERIMENT_PATH"; fi
export PYTHONUNBUFFERED=1
cd "$ROOT"

# Compose overrides just as the Hydra entry point does, then validate its schema.
# Credentials are intentionally absent from the recipe and resolved config.
"$PYTHON" - "$@" <<'PY'
import asyncio
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from northrl.data_loader import SFTStreamingDataLoader
from northrl.utils.config_schema import NorthRLConfig

recipe = Path(os.environ["TRAIN_CONFIG"])
with initialize_config_dir(config_dir=str(recipe.parent), version_base=None):
    config = compose(config_name=recipe.stem, overrides=sys.argv[1:])
validated = NorthRLConfig.model_validate(OmegaConf.to_container(config, resolve=True))
data = validated.data.train.model_dump()
data.update(
    max_sequence_length=validated.data.max_sequence_length,
    tokenizer=validated.models.actor.tokenizer,
    epochs=1,
    seed=validated.experiment.seed,
    config_file_path=str(recipe),
    keep_batch_order=True,
)
loader = SFTStreamingDataLoader(OmegaConf.create(data))
try:
    windows = asyncio.run(loader.count_optimizer_windows())
finally:
    loader.close()
print(f"Preflight passed: dataset and tokenizer; {windows} optimizer windows per epoch.")
print("Weaver credentials and model access are checked when training starts.")
PY
if "$CHECK"; then exit 0; fi
mkdir -p "$EXPERIMENT_PATH"
if [[ -e "$EXPERIMENT_PATH/northrl.log" || -e "$EXPERIMENT_PATH/exit_code" ]]; then
    echo "Run directory already used; set a new EXPERIMENT_PATH: $EXPERIMENT_PATH" >&2
    exit 1
fi
cp "$TRAIN_CONFIG" "$EXPERIMENT_PATH/recipe.yaml"
printf 'Run directory: %s\nMonitor: tail -f "%s/northrl.log"\n' "$EXPERIMENT_PATH" "$EXPERIMENT_PATH"
set +e
"$PYTHON" -m northrl.main \
    --config-path="$(dirname "$TRAIN_CONFIG")" \
    --config-name=weaver "$@" 2>&1 | tee "$EXPERIMENT_PATH/northrl.log"
STATUS=("${PIPESTATUS[@]}")
set -e
RC=${STATUS[0]}
if [[ $RC == 0 ]]; then RC=${STATUS[1]}; fi
printf '%s\n' "$RC" > "$EXPERIMENT_PATH/exit_code"
printf 'NorthRL exit code: %s; run directory: %s\n' "$RC" "$EXPERIMENT_PATH"
exit "$RC"
