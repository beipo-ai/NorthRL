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

"""Path utilities for NorthRL framework."""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_path_from_config(path: str | None, config_file_path: str | None = None) -> str | None:
    """
    Resolve a path relative to the config file's location.

    This is the preferred method for resolving paths in recipe configs,
    as it makes recipes self-contained and independent of working directory.

    Priority:
    1. Absolute path - use as-is
    2. Relative path - resolve relative to config file's directory

    Args:
        path: Path to resolve (can be None, absolute, or relative)
        config_file_path: Path to the config file (to determine base directory)

    Returns:
        Resolved absolute path, or None if input is None

    Raises:
        ValueError: If path cannot be resolved
    """
    if path is None:
        return None

    # Convert to string if Path object
    path_str = str(path)

    # If absolute path, return as-is
    if os.path.isabs(path_str):
        if not os.path.exists(path_str):
            logger.warning(f"Absolute path does not exist: {path_str}")
        return path_str

    # Relative path - resolve relative to config file's directory
    if config_file_path is None:
        raise ValueError(
            f"Cannot resolve relative path '{path_str}': " "config_file_path not provided"
        )

    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    resolved = os.path.join(config_dir, path_str)
    resolved = os.path.abspath(resolved)

    if not os.path.exists(resolved):
        logger.warning(
            f"Resolved path does not exist: {resolved} (from config: {config_file_path})"
        )

    return resolved
