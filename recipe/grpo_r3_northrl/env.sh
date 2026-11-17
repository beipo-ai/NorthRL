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

# Fill in local values and source this file before running the recipe.
# Do not commit credentials.
export FEISHU_WEBHOOK_URL=""
export WEAVER_API_KEY=""
export WEAVER_BASE_URL=""

# Absolute paths readable by the process running NorthRL.
export NORTHRL_TRAIN_DATA_PATH=""
export NORTHRL_R3_MATH500_DATA=""
export NORTHRL_R3_AIME2024_DATA=""
export NORTHRL_R3_AIME2025_DATA=""
export NORTHRL_R3_AIME2026_DATA=""

# Optional: override NORTHRL_R3_MODEL and NORTHRL_R3_TOKENIZER together if the
# Weaver model differs from Qwen/Qwen3-30B-A3B-Base. The tokenizer must match it.
# Resource requirements: docs/en/recipe/advanced-recipes/README.md#grpo-r3
