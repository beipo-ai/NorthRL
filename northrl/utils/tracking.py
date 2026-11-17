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

"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from northrl.utils.metric_catalog import (
    STEP_METRIC,
    define_wandb_metrics,
    normalize_step_metrics,
    prepare_wandb_metrics,
)
from northrl.utils.wandb_workspace import configure_wandb_workspaces

from .git import capture_git_change  # pylint: disable=relative-beyond-top-level
from .logger.feishu_logger import LarkReporter

logger = logging.getLogger(__name__)


class LoggerProtocol(Protocol):
    """Protocol for logger objects that implement the log method."""

    def log(self, data: dict[str, Any], step: int) -> None: ...

    def finish(self, exit_code: int = 0) -> None: ...


class Tracking:
    supported_backend = ["wandb", "mlflow", "swanlab", "console"]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: str | list[str] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn(
                    "`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning
                )
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger: dict[str, LoggerProtocol] = {}
        # State changes are short; backend I/O never holds this lock.
        self._log_lock = threading.RLock()
        self._backend_publish_locks: dict[str, Any] = {}
        self._wandb_pending_step: int | None = None
        self._wandb_committed_step: int | None = None
        self._weaver_metrics: list[tuple[str, Any]] = []
        self._weaver_metric_view: Any = None
        self._closed = False
        self.wandb_url = ""
        self.feishu_logger: LarkReporter | None = None
        logger_config = config.get("runtime", {}).get("logger", {}) if config else {}
        feishu_config = logger_config.get("feishu", {})
        if feishu_config.get("enabled", False):
            self.feishu_logger = LarkReporter(
                url=feishu_config["url"],
                user_name=os.environ.get("NORTHRL_USER", os.environ.get("USER", "")),
            )

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            wandb.login(host=os.environ["WANDB_HOST"], key=os.environ["WANDB_KEY"])  # type: ignore[attr-defined]  # pylint: disable=c-extension-no-member
            # Temporarily disable struct mode to add git info
            config["git"] = capture_git_change()
            # Each tracker owns its run, including when a controller is rebuilt
            # in the same process. A delayed destructor must never finish a
            # later attempt's run via the process-global wandb module.
            wandb_run = wandb.init(  # type: ignore[attr-defined]  # pylint: disable=c-extension-no-member
                project=project_name, name=experiment_name, config=config, reinit=True
            )
            self.logger["wandb"] = wandb_run
            # Presentation definitions do not change pending/commit ownership.
            define_wandb_metrics(wandb_run)
            configure_wandb_workspaces(wandb_run)
            # get_url may return None; fall back to empty string for consistency
            self.wandb_url = wandb_run.get_url() or ""

        if "mlflow" in default_backend:
            import mlflow

            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(
                    SWANLAB_API_KEY
                )  # NOTE: previous login information will be overwritten
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab  # type: ignore[assignment]

        if "console" in default_backend:
            from .logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

    def log(self, data, step, backend=None, commit: bool = False):
        # Preserve explicit missing numbered observations until W&B alias selection.
        source_data = dict(data)
        data = normalize_step_metrics(source_data)
        with self._log_lock:
            if self._closed:
                return
            backends = [
                (name, instance, self._backend_publish_locks.setdefault(name, threading.RLock()))
                for name, instance in self.logger.items()
            ]
        for default_backend, logger_instance, publish_lock in backends:
            if backend is None or default_backend in backend:
                with publish_lock:
                    if self._closed:
                        continue
                    if default_backend == "wandb":
                        if (
                            self._wandb_committed_step is not None
                            and step <= self._wandb_committed_step
                        ) or (
                            self._wandb_pending_step is not None and step < self._wandb_pending_step
                        ):
                            logger.warning("Ignoring late W&B metrics for training step %s", step)
                            continue
                        previous_pending = self._wandb_pending_step
                        cast(Any, logger_instance).log(
                            data={**prepare_wandb_metrics(source_data), STEP_METRIC: step},
                            # W&B history also contains Weaver operation rows.
                            # NorthRL's chart axis and late-result checks remain the
                            # original training step, not this transport cursor.
                            step=max(
                                step,
                                cast(Any, logger_instance).step
                                + int(previous_pending is not None and previous_pending < step),
                            ),
                            commit=commit,
                        )
                        if commit:
                            self._wandb_committed_step = step
                            self._wandb_pending_step = None
                            self._publish_weaver_metrics(logger_instance)
                        else:
                            if previous_pending is not None and previous_pending < step:
                                self._wandb_committed_step = previous_pending
                            self._wandb_pending_step = step
                    else:
                        logger_instance.log(data=data, step=step)

    def flush(self, step: int) -> None:
        """Commit this round once, without advancing an unrelated pending round."""
        with self._log_lock:
            if self._closed or "wandb" not in self.logger:
                return
            wandb_run = self.logger["wandb"]
            publish_lock = self._backend_publish_locks.setdefault("wandb", threading.RLock())
        with publish_lock:
            if self._closed or self._wandb_pending_step != step:
                return
            cast(Any, wandb_run).log(
                data={}, step=max(step, cast(Any, wandb_run).step), commit=True
            )
            self._wandb_committed_step = step
            self._wandb_pending_step = None
            self._publish_weaver_metrics(wandb_run)

    def queue_weaver_metrics(self, operation_id: str, observations: Any) -> None:
        """SDK callback: retain every operation until the next round is committed.

        No W&B I/O runs on the SDK completion thread. Local persistence is owned
        independently by the SDK. There is no metric-name/default-only filter.
        """
        with self._log_lock:
            if not self._closed and "wandb" in self.logger and observations.points:
                self._weaver_metrics.append((operation_id, observations))

    def _publish_weaver_metrics(self, wandb_run: Any) -> None:
        """Publish separate operation rows under the existing W&B publish lock.

        Never merge multiple optimizer windows into one value (in particular,
        averaging top-10% means would be incorrect). Wait for NorthRL's pending
        row to commit so these writes cannot prematurely flush partial RL data.
        """
        with self._log_lock:
            pending, self._weaver_metrics = self._weaver_metrics, []
        for operation_id, observations in pending:
            try:
                if self._weaver_metric_view is None:
                    from weaver._wandb_metrics import WandbMetricView

                    self._weaver_metric_view = WandbMetricView(
                        wandb_run, namespace="weaver_metrics_v1"
                    )
                self._weaver_metric_view.log(operation_id, observations, commit=True)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Could not publish Weaver metrics for %s", operation_id, exc_info=True
                )

    def log_live(self, data: dict[str, Any], elapsed_seconds: float) -> None:
        """Write runtime monitoring to ordinary logs without a metric-backend row."""
        with self._log_lock:
            if self._closed:
                return
        logger.info(
            "[rollout-monitor] elapsed_seconds=%.3f metrics=%s",
            elapsed_seconds,
            json.dumps(data, sort_keys=True),
        )

    def close(self, exit_code: int = 0) -> None:
        """Finish owned runs, deferring finish if a backend write is in flight."""
        if getattr(self, "_log_lock", None) is None:
            return
        with self._log_lock:
            if self._closed:
                return
            self._closed = True
            publish_locks = [
                self._backend_publish_locks.setdefault(name, threading.RLock())
                for name in ("wandb", "swanlab")
                if name in self.logger
            ]
        with ExitStack() as acquired:
            for publish_lock in publish_locks:
                if not publish_lock.acquire(blocking=False):
                    threading.Thread(
                        target=self._close_after_publish,
                        args=(exit_code, publish_locks),
                        name="northrl-logger-close",
                        daemon=True,
                    ).start()
                    return
                acquired.callback(publish_lock.release)
            self._close_backends(exit_code)

    def _close_after_publish(self, exit_code: int, publish_locks: list[Any]) -> None:
        with ExitStack() as acquired:
            for publish_lock in publish_locks:
                acquired.enter_context(publish_lock)
            self._close_backends(exit_code)

    def _close_backends(self, exit_code: int) -> None:
        for backend in ("wandb", "swanlab"):
            logger_instance = getattr(self, "logger", {}).get(backend)
            if logger_instance is None:
                continue
            try:
                if backend == "wandb":
                    import wandb

                    # W&B 0.19.8 finish() also clears global bindings, even
                    # for an old or disabled run already ended by reinit.
                    active_run = getattr(wandb, "run", None)
                    if active_run is logger_instance:
                        if self._weaver_metrics and self._wandb_pending_step is not None:
                            cast(Any, logger_instance).log(data={}, commit=True)
                            self._wandb_pending_step = None
                        self._publish_weaver_metrics(logger_instance)
                        logger_instance.finish(exit_code=exit_code)
                else:
                    logger_instance.finish()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to finish %s experiment logger", backend, exc_info=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Globals may already be unavailable during interpreter shutdown.
            pass


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow

        mlflow.log_metrics(metrics=data, step=step)

    def finish(self, exit_code: int = 0) -> None:
        """Finish method for compatibility with LoggerProtocol."""


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(
        _transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/"
    )


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(
        _transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict
    )

    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return {str(k): v for k, v in ans.items()}
