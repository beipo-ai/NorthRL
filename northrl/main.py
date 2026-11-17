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

import logging
import os
import signal
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from .controller import NorthRLController
from .utils.config_schema import NorthRLConfig
from .utils.config_utils import dump_resolved_config
from .utils.logging_utils import set_logging_basic_config

logger = logging.getLogger(__name__)


def _make_retry_config(config: DictConfig) -> DictConfig:
    """Prefer the latest checkpoint on retry, retaining the requested fallback."""
    config_data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_data, dict):
        raise TypeError("NorthRL config must resolve to a mapping")

    retry_config = OmegaConf.create(config_data)
    # An explicit resume path remains the fallback until this run has saved a
    # full checkpoint. A path ignored by disabled resume must stay ignored.
    if retry_config.training.resume.mode == "disable":
        retry_config.training.resume.path = ""
    retry_config.training.resume.mode = "auto"
    OmegaConf.set_struct(retry_config, True)
    return retry_config


def main_task(config: DictConfig) -> None:
    """Run NorthRL locally, rebuilding once from the latest checkpoint on failure."""
    # Print initial config for debugging
    from pprint import pprint

    OmegaConf.resolve(config)
    # NOTE: legacy migrate_legacy_config() call removed in Phase 0; old recipes
    # are no longer auto-migrated. Use the new schema (see CONFIG_REFACTOR.md).

    logger.info("Starting NorthRL with configuration:")
    pprint(OmegaConf.to_container(config, resolve=True))

    # Dump resolved config to experiment path for reproducibility
    dump_resolved_config(config)

    max_retries = int(config.training.resume.max_retries)
    attempt_config = config
    for retry in range(max_retries + 1):
        if retry > 0:
            logger.info(
                "Starting NorthRL retry %d/%d with training.resume.mode=auto",
                retry,
                max_retries,
            )
        # A failed controller owns thread pools, queues, clients, and worker
        # state that must not be reused. Controller.run() drains and closes it;
        # every retry starts from a newly constructed object graph.
        controller = NorthRLController(attempt_config)
        try:
            controller.run()
            return
        except Exception:  # pylint: disable=broad-exception-caught
            if retry >= max_retries:
                raise
            logger.exception(
                "NorthRL training failed; rebuilding the controller and retrying "
                "from the latest checkpoint (retry %d/%d)",
                retry + 1,
                max_retries,
            )
            attempt_config = _make_retry_config(config)


def _sigterm_handler(signum, _frame):
    """Convert SIGTERM to SystemExit so controller cleanup and atexit handlers run."""
    sys.exit(128 + signum)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    """Run the recipe selected with --config-path, --config-name, and TRAIN_CONFIG."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    set_logging_basic_config()

    # TRAIN_CONFIG identifies the recipe file for downstream relative paths.
    try:
        config_file_path = os.environ["TRAIN_CONFIG"]
        logger.info(f"Using config file path from TRAIN_CONFIG: {config_file_path}")
    # Fail hard if we couldn't determine the config file path
    except Exception as exc:
        raise RuntimeError(
            "Cannot determine config file path. "
            "Set TRAIN_CONFIG to the recipe file and pass "
            "--config-path=<recipe-directory> --config-name=<recipe-name>."
        ) from exc

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Validate against the pydantic schema (see CONFIG_REFACTOR.md). This must run
    # BEFORE injecting _config_file_path, because NorthRLConfig forbids extra fields.
    OmegaConf.resolve(config)
    config_as_dict = OmegaConf.to_container(config, resolve=True)
    validated = NorthRLConfig.model_validate(config_as_dict)  # raises on schema failure

    # Rebuild the downstream config from the validated model so it carries every
    # schema default. Recipes omit optional fields (e.g. algorithm.kl.mode when
    # kl.enabled=false); the raw OmegaConf wouldn't have them, and consumers that
    # access fields by attribute would hit "Key 'x' is not in struct". Dumping the
    # pydantic model (by_alias so `validate` keeps its yaml key) gives a fully
    # defaulted dict that we wrap back into OmegaConf for the rest of the run.
    config = OmegaConf.create(validated.model_dump(by_alias=True))

    # Store config_file_path on the config object for downstream use (custom
    # module and data path resolution). This is an internal
    # "ghost field" — not part of the user-facing schema; see CONFIG_REFACTOR.md
    # §7.2 #1 (kept as-is in this PR; deletion deferred to a follow-up PR).
    OmegaConf.set_struct(config, False)
    config._config_file_path = config_file_path  # pylint: disable=protected-access
    OmegaConf.set_struct(config, True)
    logger.info("Starting NorthRL in local mode")
    main_task(config)
    logger.info("NorthRL training completed")
    logger.info(f"Experiment log path: {os.environ.get('EXPERIMENT_PATH', 'not found')}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
