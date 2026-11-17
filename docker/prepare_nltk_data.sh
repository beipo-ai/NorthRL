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

set -euo pipefail

# Download fixed IFEval resources without fetching the NLTK package index.
# NLTK_DATA_BASE_URL can point to an internal mirror of the same archives.
export NLTK_DATA="${NLTK_DATA:-/usr/local/share/nltk_data}"
cache_dir="${NLTK_DOWNLOAD_CACHE:-/root/.cache/northrl-nltk}"
base_url="${NLTK_DATA_BASE_URL:-https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers}"
mkdir -p "$cache_dir"
download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT

for resource in \
    punkt:51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec \
    punkt_tab:e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106; do
    name="${resource%%:*}"
    checksum="${resource#*:}"
    archive="$cache_dir/$name.zip"
    if [[ -f "$archive" ]] && printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status; then
        printf 'Using cached NLTK resource: %s\n' "$name"
        continue
    fi

    printf 'Downloading NLTK resource: %s\n' "$name"
    downloaded="$download_dir/$name.zip"
    timeout --kill-after=5s 300s curl \
        --fail --location --show-error \
        --connect-timeout 15 --max-time 120 \
        --retry 2 --retry-all-errors --retry-max-time 240 \
        --output "$downloaded" "${base_url%/}/$name.zip"
    printf '%s  %s\n' "$checksum" "$downloaded" | sha256sum --check
    mv "$downloaded" "$archive"
done

python - "$cache_dir" <<'PY'
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import nltk

data_dir = Path(os.environ["NLTK_DATA"])
tokenizers_dir = data_dir / "tokenizers"
tokenizers_dir.mkdir(parents=True, exist_ok=True)
for name in ("punkt", "punkt_tab"):
    with ZipFile(Path(sys.argv[1]) / f"{name}.zip") as archive:
        archive.extractall(tokenizers_dir)

nltk.data.path = [str(data_dir)]
nltk.data.find("tokenizers/punkt")
nltk.data.find("tokenizers/punkt_tab/english")
nltk.data.load("nltk:tokenizers/punkt/english.pickle")
print(f"Verified NLTK tokenizer data in {data_dir}")
PY
chmod -R a+rX "$NLTK_DATA"
