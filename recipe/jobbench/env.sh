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

export SERPER_API_KEY="${SERPER_API_KEY:-}"
export SERPER_API_URL="${SERPER_API_URL:-https://google.serper.dev/search}"
export NORTHRL_JOBBENCH_REFERENCE_VOLUME="${NORTHRL_JOBBENCH_REFERENCE_VOLUME:-}"
export JOBBENCH_REFERENCES_ROOT="${JOBBENCH_REFERENCES_ROOT:-}"
export JOBBENCH_REFERENCES_MANIFEST_SHA256="${JOBBENCH_REFERENCES_MANIFEST_SHA256:-}"

# Forward names only; secret values are never stored in the recipe/task payload.
# Ignore spacing while checking existing names. The runtime still validates the
# original selector, so invalid names and empty comma-separated entries fail.
jobbench_selected_env="$(printf '%s' "${NORTHRL_E2B_AGENT_ENV:-}" | tr -d '[:space:]')"
if [ -z "$jobbench_selected_env" ]; then
    NORTHRL_E2B_AGENT_ENV=""
fi
for jobbench_env_name in SERPER_API_KEY SERPER_API_URL JOBBENCH_REFERENCES_ROOT \
    JOBBENCH_REFERENCES_MANIFEST_SHA256; do
    case ",${jobbench_selected_env}," in
        *,"${jobbench_env_name}",*) ;;
        *) NORTHRL_E2B_AGENT_ENV="${NORTHRL_E2B_AGENT_ENV:+${NORTHRL_E2B_AGENT_ENV},}${jobbench_env_name}" ;;
    esac
done
export NORTHRL_E2B_AGENT_ENV
unset jobbench_env_name jobbench_selected_env
