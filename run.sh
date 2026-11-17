# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

usage() {
    cat <<'HELP'
Usage: bash run.sh [--env-file PATH] RECIPE.yaml [Hydra overrides...]
Uses the current shell environment; .env is not loaded automatically.
--env-file selects a Bash file to source before Python and takes precedence
over the optional NORTHRL_ENV_FILE environment variable.
Defaults NORTHRL_PATH to this script's directory and PYTHON to python on PATH.
HELP
}

if [[ ${1:-} == --help || ${1:-} == -h || $# == 0 ]]; then
    usage
    if [[ $# == 0 ]]; then exit 2; fi
    exit 0
fi

ENV_FILE=${NORTHRL_ENV_FILE:-}
case "$1" in
    --env-file)
        if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
            printf '%s\n' '--env-file requires a file path.' >&2
            exit 2
        fi
        ENV_FILE=$2
        shift 2
        ;;
    --env-file=*)
        ENV_FILE=${1#*=}
        if [[ -z "$ENV_FILE" ]]; then
            printf '%s\n' '--env-file requires a file path.' >&2
            exit 2
        fi
        shift
        ;;
esac
if [[ $# == 0 ]]; then
    printf '%s\n' 'A recipe YAML is required after --env-file.' >&2
    usage >&2
    exit 2
fi
if [[ "$1" == -* ]]; then
    printf 'Unknown startup option: %s\n' "$1" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "$1" || ! -r "$1" || "$1" != *.yaml ]]; then
    printf 'Recipe must be a readable .yaml file: %s\n' "$1" >&2
    exit 2
fi

# Keep the selected recipe and arguments independent of the sourced script.
readonly NORTHRL_RUN_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly NORTHRL_RUN_CWD=$PWD
readonly NORTHRL_RUN_CONFIG="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
shift
readonly -a NORTHRL_RUN_OVERRIDES=("$@")

# Environment files can use the checkout path when defining their own paths.
export NORTHRL_PATH="${NORTHRL_PATH:-$NORTHRL_RUN_ROOT}"
if [[ -n "$ENV_FILE" ]]; then
    if [[ "$ENV_FILE" != /* ]]; then ENV_FILE="$NORTHRL_RUN_CWD/$ENV_FILE"; fi
    if [[ ! -f "$ENV_FILE" || ! -r "$ENV_FILE" ]]; then
        printf 'Environment file is missing or unreadable: %s\n' "$ENV_FILE" >&2
        exit 2
    fi
    set -a
    source "$ENV_FILE"
    set +a
fi

export NORTHRL_PATH="${NORTHRL_PATH:-$NORTHRL_RUN_ROOT}"
PYTHON=${PYTHON:-python}
cd -- "$NORTHRL_RUN_CWD"
export TRAIN_CONFIG="$NORTHRL_RUN_CONFIG"
CONFIG_NAME=${TRAIN_CONFIG##*/}
exec "$PYTHON" -m northrl.main \
    --config-path="${TRAIN_CONFIG%/*}" \
    --config-name="${CONFIG_NAME%.yaml}" "${NORTHRL_RUN_OVERRIDES[@]}"
