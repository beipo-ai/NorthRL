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

set -eu
umask 022
# Ignore template/user PATH additions while selecting trusted system tools.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

_BOOTSTRAP_ROOT=/opt/northrl-bootstrap
_BOOTSTRAP_PYTHON=${_BOOTSTRAP_ROOT}/python
_BOOTSTRAP_RUNTIME=${_BOOTSTRAP_ROOT}/runtime
_UV_ROOT=/opt/northrl-uv
_UV_BIN=${_UV_ROOT}/bin/uv
_PYTHON_INSTALL_ROOT=/opt/northrl-python
_PYTHON_VERSION=3.12
_UV_VERSION=0.12.17

_log() {
    printf '[bootstrap] %s\n' "$*" >&2
}

_fail() {
    _log "ERROR: $*"
    exit 1
}

_python_is_usable() {
    candidate=$1
    "$candidate" -I -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' \
        >/dev/null 2>&1
}

_use_python() {
    candidate=$1
    mkdir -p "$_BOOTSTRAP_ROOT"
    rm -f "$_BOOTSTRAP_PYTHON"
    ln -s "$candidate" "$_BOOTSTRAP_PYTHON"
    _log "using system Python: $candidate"
}

_find_uv() {
    for candidate in \
        "$_UV_BIN" \
        /usr/local/bin/uv \
        /usr/bin/uv \
        /root/.local/bin/uv \
        /workspace/.local/bin/uv
    do
        if [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

_download_uv() {
    mkdir -p "$_UV_ROOT/bin"
    temporary_directory=$(mktemp -d /tmp/northrl-uv-bootstrap.XXXXXX)
    trap 'rm -rf "$temporary_directory"' EXIT HUP INT TERM
    installer="$temporary_directory/install.sh"
    installer_url="https://astral.sh/uv/${_UV_VERSION}/install.sh"

    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --silent --show-error \
            --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time 120 \
            "$installer_url" >"$installer" || _fail "failed to download uv installer"
    elif command -v wget >/dev/null 2>&1; then
        wget --https-only --timeout=15 --tries=3 --output-document="$installer" \
            "$installer_url" || _fail "failed to download uv installer"
    else
        _fail "Python and uv are absent, and neither curl nor wget is available"
    fi

    UV_UNMANAGED_INSTALL="$_UV_ROOT/bin" \
        UV_INSTALL_DIR="$_UV_ROOT/bin" \
        sh "$installer" || _fail "uv installer failed"
    [ -x "$_UV_BIN" ] || _fail "uv installer did not create $_UV_BIN"
    _log "installed uv $_UV_VERSION at $_UV_BIN"
}

_prepare_with_uv() {
    uv=$1
    mkdir -p "$_PYTHON_INSTALL_ROOT" "$_BOOTSTRAP_ROOT"
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export UV_NO_CONFIG=1
    export UV_PYTHON_DOWNLOADS=automatic
    export UV_PYTHON_INSTALL_DIR="$_PYTHON_INSTALL_ROOT"
    unset UV_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_FIND_LINKS \
        UV_CONFIG_FILE UV_PYTHON UV_PYTHON_INSTALL_MIRROR UV_PYTHON_DOWNLOADS_JSON_URL \
        UV_PYTHON_PREFERENCE UV_MANAGED_PYTHON UV_NO_MANAGED_PYTHON || true

    rm -rf "$_BOOTSTRAP_RUNTIME"
    "$uv" venv --clear --managed-python --python "$_PYTHON_VERSION" \
        "$_BOOTSTRAP_RUNTIME" || _fail "uv could not prepare Python $_PYTHON_VERSION"
    [ -x "$_BOOTSTRAP_RUNTIME/bin/python" ] || \
        _fail "uv did not create $_BOOTSTRAP_RUNTIME/bin/python"
    rm -f "$_BOOTSTRAP_PYTHON"
    ln -s "$_BOOTSTRAP_RUNTIME/bin/python" "$_BOOTSTRAP_PYTHON"
    _log "using uv-managed Python $_PYTHON_VERSION: $_BOOTSTRAP_PYTHON"
}

_system_python=$(command -v python3 2>/dev/null || true)
if [ -n "$_system_python" ] && _python_is_usable "$_system_python"; then
    _use_python "$_system_python"
else
    _log "no usable system python3 found; preparing uv-managed Python $_PYTHON_VERSION"
    _uv=$( _find_uv || true )
    if [ -z "$_uv" ]; then
        _download_uv
        _uv=$_UV_BIN
    fi
    _prepare_with_uv "$_uv"
fi

_bootstrap_script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$_BOOTSTRAP_PYTHON" -I "$_bootstrap_script_directory/e2b_bootstrap.py"
