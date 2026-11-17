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

"""Profiling respects log levels without hiding failed or cancelled stages."""

import asyncio
import logging

import pytest

from northrl.utils import profiling


def test_successful_profiles_are_quiet_at_info(caplog, capsys):
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        with profiling.profile_block("adapter.generate", request_seq=1):
            pass
        profiling.profile_event("adapter.generate", "skipped", reason="already_done")

    assert not caplog.records
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_debug_profiles_preserve_timing_and_context(caplog, monkeypatch):
    ticks = iter((10.0, 12.3456))
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    with caplog.at_level(logging.DEBUG, logger=profiling.__name__):
        with profiling.profile_block("adapter.generate", rollout_id="rollout-1"):
            pass

    assert caplog.messages == [
        "[PROFILE] stage=adapter.generate status=started rollout_id='rollout-1'",
        "[PROFILE] stage=adapter.generate status=completed rollout_id='rollout-1' "
        "duration_seconds=2.346",
    ]
    assert all(record.levelno == logging.DEBUG for record in caplog.records)


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_failed_profiles_remain_visible_and_propagate(caplog, monkeypatch, failure):
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(ticks))

    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        with pytest.raises(type(failure)) as exc_info:
            with profiling.profile_block("adapter.generate", request_seq=2):
                raise failure

    assert exc_info.value is failure
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert caplog.messages[0] == (
        "[PROFILE] stage=adapter.generate status=failed request_seq=2 "
        f"duration_seconds=2.000 error_type='{type(failure).__name__}'"
    )


def test_nonroutine_status_stays_visible_at_info(caplog):
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        profiling.profile_event("runtime.context-limit", "stopped", rollout_id="rollout-2")

    assert caplog.messages == [
        "[PROFILE] stage=runtime.context-limit status=stopped rollout_id='rollout-2'"
    ]
    assert caplog.records[0].levelno == logging.INFO


def test_explicit_level_overrides_status_default(caplog):
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        profiling.profile_event("driver.file-download", "completed", level=logging.INFO)
        profiling.profile_event("driver.file-download", "failed", level=logging.DEBUG)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.messages == ["[PROFILE] stage=driver.file-download status=completed"]


def test_result_diagnostics_are_opt_in_and_do_not_read_cpu_clock_when_disabled(monkeypatch, caplog):
    monkeypatch.delenv("NORTHRL_PROFILE_RESULTS", raising=False)
    monkeypatch.setattr(
        profiling.time, "thread_time", lambda: pytest.fail("disabled diagnostics read CPU clock")
    )
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        with profiling.result_profile_block("driver.result-postprocess", rollout_id="r"):
            pass
    assert not caplog.records


@pytest.mark.parametrize("fails", [False, True])
def test_result_diagnostics_measure_thread_cpu_and_preserve_errors(monkeypatch, caplog, fails):
    monkeypatch.setenv("NORTHRL_PROFILE_RESULTS", "1")
    wall, cpu = iter([10.0, 14.0]), iter([2.0, 2.25])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(profiling.time, "thread_time", lambda: next(cpu))
    original = ValueError("original failure")

    def run():
        with profiling.result_profile_block("driver.result-pool-insert", rollout_id="r"):
            if fails:
                raise original

    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        if fails:
            with pytest.raises(ValueError) as error:
                run()
            assert error.value is original
        else:
            run()
    assert len(caplog.records) == 1
    assert "parent_wall_seconds=4.0 parent_cpu_seconds=0.25" in caplog.messages[0]
    assert f"status={'failed' if fails else 'completed'}" in caplog.messages[0]


def test_result_diagnostic_sink_failure_does_not_fail_work(monkeypatch):
    monkeypatch.setenv("NORTHRL_PROFILE_RESULTS", "1")

    def unavailable(*args, **kwargs):
        raise RuntimeError("diagnostic sink unavailable")

    monkeypatch.setattr(profiling, "profile_event", unavailable)
    with profiling.result_profile_block("driver.result-postprocess"):
        pass
    original = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError) as error:
        with profiling.result_profile_block("driver.result-postprocess"):
            raise original
    assert error.value is original
