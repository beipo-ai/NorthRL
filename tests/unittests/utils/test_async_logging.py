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

"""Background-loop logging must not block networking or alter synchronous callers."""

import asyncio
import logging
import threading

import pytest

from northrl.utils import async_utils
from northrl.utils.async_logging import AsyncLogDispatcher
from northrl.utils.async_utils import (
    flush_sdk_logs,
    run_coro,
    shutdown_event_loop,
    submit_coro,
    submit_rollout_coro,
)


@pytest.fixture(autouse=True)
def clean_sdk_loop():
    shutdown_event_loop()
    yield
    shutdown_event_loop()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.delivery_threads = []

    def emit(self, record):
        self.records.append(record)
        self.delivery_threads.append(threading.current_thread().name)


@pytest.fixture
def log():
    instance = logging.getLogger("northrl.test.sdk-logging")
    previous = instance.level, instance.propagate, instance.handlers[:]
    instance.setLevel(logging.DEBUG)
    instance.propagate = False
    instance.handlers.clear()
    yield instance
    instance.level, instance.propagate, instance.handlers = previous


@pytest.mark.parametrize("handler_target", ["custom", "root"])
@pytest.mark.parametrize(
    ("submit", "loop_name"),
    [(submit_coro, "northrl-async-loop"), (submit_rollout_coro, "northrl-rollout-loop")],
)
def test_slow_handler_does_not_block_background_coroutines(log, handler_target, submit, loop_name):
    entered, release = threading.Event(), threading.Event()
    records = _Records()

    class SlowHandler(logging.Handler):
        def emit(self, record):
            if record.name != log.name:
                return
            entered.set()
            assert release.wait(5), "test did not release slow handler"
            records.handle(record)

    target = logging.getLogger() if handler_target == "root" else log
    log.propagate = handler_target == "root"
    handler = SlowHandler()
    target.addHandler(handler)

    async def emit():
        log.info("slow background-loop log")
        return threading.current_thread().name

    try:
        assert submit(emit()).result(timeout=2) == loop_name
        assert entered.wait(2)

        async def unrelated_network_work():
            await asyncio.sleep(0)
            return "network still progresses"

        assert submit(unrelated_network_work()).result(timeout=2) == "network still progresses"
    finally:
        release.set()
        shutdown_event_loop()
        target.removeHandler(handler)

    assert records.delivery_threads == ["northrl-sdk-logs"]
    assert records.records[0].threadName == loop_name


@pytest.mark.parametrize("stop_first", ["sdk", "rollout"])
def test_sdk_and_rollout_dispatchers_drain_and_share_hook_until_both_stop(log, stop_first):
    original = logging.Logger.callHandlers
    records = _Records()
    log.addHandler(records)
    started = {name: threading.Event() for name in ("sdk", "rollout")}
    bridges = {"sdk": async_utils._ASYNC_BRIDGE, "rollout": async_utils._ROLLOUT_BRIDGE}

    async def pending(name):
        started[name].set()
        try:
            await asyncio.Event().wait()
        finally:
            log.warning("%s cancelled during shutdown", name)

    results = {name: bridge.submit(pending(name)) for name, bridge in bridges.items()}
    assert all(event.wait(2) for event in started.values())
    hook = logging.Logger.callHandlers
    assert hook is not original
    assert bridges["sdk"]._log_dispatcher is not bridges["rollout"]._log_dispatcher

    bridges[stop_first].shutdown()
    assert results[stop_first].cancelled()
    assert records.records[0].getMessage() == f"{stop_first} cancelled during shutdown"
    assert logging.Logger.callHandlers is hook

    still_running = "rollout" if stop_first == "sdk" else "sdk"

    async def emit():
        log.info("%s still running", still_running)

    bridges[still_running].submit(emit()).result(timeout=2)
    bridges[still_running].flush_logs()
    assert records.records[-1].getMessage() == f"{still_running} still running"
    assert records.delivery_threads == ["northrl-sdk-logs", "northrl-sdk-logs"]

    bridges[still_running].shutdown()
    assert results[still_running].cancelled()
    assert records.records[-1].getMessage() == f"{still_running} cancelled during shutdown"
    assert logging.Logger.callHandlers is original


def test_formatting_and_original_exception_are_delivered_by_worker(log):
    records = _Records()
    formatting_threads = []

    class Message:
        def __str__(self):
            formatting_threads.append(threading.current_thread().name)
            return "expensive message"

    class FormatHandler(logging.Handler):
        def emit(self, record):
            self.format(record)
            records.handle(record)

    log.addHandler(FormatHandler())

    async def emit():
        try:
            raise ValueError("original traceback")
        except ValueError:
            log.exception("failed: %s", Message())

    run_coro(emit())
    shutdown_event_loop()
    record = records.records[0]
    assert formatting_threads == ["northrl-sdk-logs"]
    assert record.exc_info[0] is ValueError
    assert "ValueError: original traceback" in record.exc_text
    assert record.getMessage() == "failed: expensive message"


def test_non_sdk_logging_stays_synchronous_and_keeps_caplog(log, caplog):
    log.propagate = True
    records = _Records()
    log.addHandler(records)

    async def start_sdk():
        return None

    run_coro(start_sdk())
    handler_list = list(logging.getLogger().handlers)
    with caplog.at_level(logging.INFO):
        log.info("synchronous caller")
        assert records.records[0].getMessage() == "synchronous caller"
        assert records.delivery_threads == [threading.current_thread().name]
        assert caplog.records[-1].getMessage() == "synchronous caller"
    assert logging.getLogger().handlers == handler_list


def test_sdk_handler_added_after_loop_start_and_propagation_are_preserved(log, caplog):
    log.propagate = True

    async def emit():
        # Third-party SDKs can attach their own handlers after loop startup.
        records = _Records()
        records.setLevel(logging.WARNING)
        log.addHandler(records)
        log.debug("filtered from custom handler")
        log.warning("SDK warning")
        return records

    with caplog.at_level(logging.DEBUG):
        records = run_coro(emit())
        shutdown_event_loop()
    assert [record.getMessage() for record in records.records] == ["SDK warning"]
    assert records.delivery_threads == ["northrl-sdk-logs"]
    sdk_records = [record for record in caplog.records if record.name == log.name]
    assert [record.getMessage() for record in sdk_records] == [
        "filtered from custom handler",
        "SDK warning",
    ]


def test_full_queue_drops_records_without_blocking_and_reports_count(log, caplog):
    entered, release = threading.Event(), threading.Event()
    records = _Records()

    class SlowFirstHandler(logging.Handler):
        def emit(self, record):
            if record.msg == "first":
                entered.set()
                assert release.wait(5)
            records.handle(record)

    log.addHandler(SlowFirstHandler())
    dispatcher = AsyncLogDispatcher(capacity=2)
    try:
        with dispatcher.activate():
            log.info("first")
        assert entered.wait(2)
        with dispatcher.activate():
            for index in range(20):
                log.info("queued %s", index)
        assert dispatcher._queue.qsize() == 2
    finally:
        release.set()
        dispatcher.close()
    assert [record.getMessage() for record in records.records] == ["first", "queued 0", "queued 1"]
    assert any("dropped 18 records" in record.message for record in caplog.records)


def test_broken_handler_does_not_kill_worker(log, capsys):
    records = _Records()

    class FailsOnce(logging.Handler):
        def emit(self, record):
            if record.msg == "bad":
                raise RuntimeError("broken handler")
            records.handle(record)

    log.addHandler(FailsOnce())

    async def emit():
        log.info("bad")
        log.info("good")

    run_coro(emit())
    shutdown_event_loop()
    assert [record.getMessage() for record in records.records] == ["good"]
    assert "RuntimeError: broken handler" in capsys.readouterr().err


def test_shutdown_drains_cancelled_task_logs_and_restores_logging(log):
    original = logging.Logger.callHandlers
    records = _Records()
    log.addHandler(records)
    started = threading.Event()

    async def pending():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            log.warning("cancelled during shutdown")

    pending_result = submit_coro(pending())
    assert started.wait(2)
    shutdown_event_loop()
    assert pending_result.cancelled()
    assert [record.getMessage() for record in records.records] == ["cancelled during shutdown"]
    assert logging.Logger.callHandlers is original

    async def restarted():
        log.info("restarted")

    run_coro(restarted())
    shutdown_event_loop()
    assert records.records[-1].getMessage() == "restarted"
    assert logging.Logger.callHandlers is original


def test_waiting_for_log_worker_is_forbidden_on_sdk_thread():
    dispatcher = AsyncLogDispatcher()
    try:
        with dispatcher.activate():
            with pytest.raises(RuntimeError, match="SDK loop"):
                dispatcher.flush()
            with pytest.raises(RuntimeError, match="SDK loop"):
                dispatcher.close()
        dispatcher.flush()
    finally:
        dispatcher.close()
    dispatcher.close()


def test_existing_logger_and_handler_filters_keep_their_semantics(log):
    records = _Records()
    records.addFilter(lambda record: "handler rejected" not in record.msg)
    log.addHandler(records)
    reject_logger = lambda record: "logger rejected" not in record.msg
    log.addFilter(reject_logger)
    try:

        async def emit():
            log.info("logger rejected")
            log.info("handler rejected")
            log.info("accepted")

        run_coro(emit())
        flush_sdk_logs()
        assert [record.getMessage() for record in records.records] == ["accepted"]
    finally:
        log.removeFilter(reject_logger)


def test_later_external_hook_survives_shutdown_and_sdk_restart(log):
    original = logging.Logger.callHandlers
    records = _Records()
    log.addHandler(records)

    async def emit(message):
        log.info(message)

    run_coro(emit("first"))
    scoped_hook = logging.Logger.callHandlers
    external_calls = []

    def external_hook(logger, record):
        external_calls.append(record.getMessage())
        scoped_hook(logger, record)

    logging.Logger.callHandlers = external_hook
    try:
        shutdown_event_loop()
        assert logging.Logger.callHandlers is external_hook
        log.info("synchronous after shutdown")
        run_coro(emit("restarted through external hook"))
        shutdown_event_loop()
        assert logging.Logger.callHandlers is external_hook
    finally:
        shutdown_event_loop()
        logging.Logger.callHandlers = original
    assert [record.getMessage() for record in records.records] == [
        "first",
        "synchronous after shutdown",
        "restarted through external hook",
    ]
    assert external_calls == ["synchronous after shutdown", "restarted through external hook"]


def test_shutdown_waits_for_running_executor_then_drains_sdk_logs(log):
    worker_entered, release_worker = threading.Event(), threading.Event()
    worker_finished = threading.Event()
    log_entered, release_log = threading.Event(), threading.Event()
    shutdown_finished = threading.Event()
    records = _Records()

    class SlowShutdownHandler(logging.Handler):
        def emit(self, record):
            if record.msg == "SDK task cancelled":
                log_entered.set()
                assert release_log.wait(5), "test did not release SDK log handler"
            records.handle(record)

    log.addHandler(SlowShutdownHandler())

    def executor_work():
        worker_entered.set()
        assert release_worker.wait(5), "test did not release executor worker"
        worker_finished.set()

    async def pending():
        try:
            await asyncio.to_thread(executor_work)
        finally:
            log.warning("SDK task cancelled")

    result = submit_coro(pending())
    assert worker_entered.wait(2)

    def shutdown():
        shutdown_event_loop()
        shutdown_finished.set()

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    try:
        assert log_entered.wait(2), "SDK task was not cancelled during shutdown"
        assert not worker_finished.is_set()
        assert not shutdown_finished.wait(0.05)
        release_worker.set()
        assert worker_finished.wait(2)
        assert not shutdown_finished.wait(0.05), "SDK logs must drain before shutdown returns"
        release_log.set()
        assert shutdown_finished.wait(2)
    finally:
        release_worker.set()
        release_log.set()
        shutdown_thread.join(timeout=5)
    assert not shutdown_thread.is_alive()
    assert result.cancelled()
    assert [record.getMessage() for record in records.records] == ["SDK task cancelled"]
