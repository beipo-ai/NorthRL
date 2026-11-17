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

"""CPU-only async protocol checks; load the real methods without ML imports."""

import ast
import asyncio
import importlib.util
import logging
import time
import unittest
from array import array
from concurrent.futures import CancelledError as FutureCancelledError
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def load_file(name):
    path = ROOT / "northrl/utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_file("request_pipeline").run_request_pipeline
timings = load_file("training_timing").training_stage_metrics


async def direct(coro):
    return await coro


def trainer_method(name):
    path = ROOT / "northrl/trainer/remote_api_trainer.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if getattr(node, "name", None) == name)
    # Postponed annotations keep this harness independent of Torch/NorthRL imports.
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    namespace = {
        "asyncio": asyncio,
        "array": array,
        "time": time,
        "logger": logging.getLogger(__name__),
        "await_coro": direct,
        "run_request_pipeline": pipeline,
        "training_stage_metrics": timings,
        "iter_training_sequences": iter,
        "FutureCancelledError": FutureCancelledError,
        "align_target_logprobs_to_tokens": lambda **kw: [0.0, *kw["target_logprobs"]],
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_pipeline_and_invalid_depth_do_not_submit(self):
        async def unexpected(*args):
            self.fail("No request should be submitted or collected")

        self.assertEqual(await pipeline([], unexpected, unexpected, submit_ahead=1), [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            await pipeline([0], unexpected, unexpected, submit_ahead=-1)

    async def test_bounded_overlap_and_order(self):
        for ahead in (0, 1, 2):
            events, live, peak = [], 0, 0

            async def submit(item):
                nonlocal live, peak
                live += 1
                peak = max(peak, live)
                events.append(("submit", item))
                return item

            async def collect(item, handle):
                nonlocal live
                await asyncio.sleep(0.002 * (4 - item))
                live -= 1
                events.append(("done", item))
                return handle * 2

            result = await pipeline(range(4), submit, collect, submit_ahead=ahead)
            self.assertEqual(result, [0, 2, 4, 6])
            self.assertEqual([i for e, i in events if e == "submit"], list(range(4)))
            self.assertEqual(peak, ahead + 1)
            self.assertEqual(live, 0)
            if ahead:
                self.assertLess(events.index(("submit", 1)), events.index(("done", 0)))

    async def test_submit_or_collect_failure_drains(self):
        for fail in ("submit", "collect"):
            done = []

            async def submit(item):
                if fail == "submit" and item == 1:
                    raise ValueError("submit failed")
                return item

            async def collect(item, handle):
                await asyncio.sleep(0.001)
                done.append(handle)
                if fail == "collect" and item == 0:
                    raise ValueError("collect failed")
                return item

            with self.assertRaisesRegex(ValueError, "failed"):
                await pipeline(range(5), submit, collect, submit_ahead=1)
            self.assertEqual(sorted(done), [0] if fail == "submit" else [0, 1])

    async def test_cancellation_drains_accepted_submission(self):
        submitting, release, collected = asyncio.Event(), asyncio.Event(), []

        async def submit(item):
            submitting.set()
            await release.wait()
            return item

        async def collect(item, handle):
            collected.append(handle)
            return item

        task = asyncio.create_task(pipeline(range(3), submit, collect, submit_ahead=1))
        await submitting.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(collected, [0])

    async def test_trainer_keeps_one_opt_and_same_loss(self):
        method = trainer_method("_run_optimizer_batch")
        for ahead, fail, chunk_size in (
            (0, False, 1),
            (1, False, 1),
            (0, False, 5),
            (1, False, 5),
            (1, True, 1),
        ):
            events, consumed = [], []

            class Holder:
                async def submit_forward_backward(self, *, datums_data, **kwargs):
                    events.append(("submit", datums_data[0]["id"]))
                    consumed.extend(datums_data)
                    return datums_data

                async def collect_forward_backward_result(self, handle, **kwargs):
                    await asyncio.sleep(0.001)
                    events.append(("done", handle[0]["id"]))
                    if fail and handle[0]["id"] == 0:
                        raise ValueError("FB failed")
                    return {"metrics": {"loss": sum(d["id"] for d in handle), "compute_ms": 1.0}}

                async def forward_backward(self, **kwargs):
                    return await self.collect_forward_backward_result(
                        await self.submit_forward_backward(**kwargs)
                    )

                async def optim_step(self, **kwargs):
                    events.append(("opt", None))
                    return {"metrics": {"update_successful": True}}

            counts = [i % 3 for i in range(5)]
            batch = [
                {"id": i, "loss_fn_inputs": {"loss_mask": [0] + [1] * count}}
                for i, count in enumerate(counts)
            ]
            mini_batches = [batch[i : i + chunk_size] for i in range(0, len(batch), chunk_size)]
            trainer = SimpleNamespace(
                _service_holder=Holder(),
                _custom_loss_spec=None,
                _submit_ahead=ahead,
                _train_step=0,
                _loss_fn="importance_sampling",
                _loss_agg_mode="prompt-mean",
                _beta1=0.9,
                _beta2=0.98,
                _weight_decay=0,
                _eps=1e-8,
                _grad_clip_norm=1,
                _effective_learning_rate=lambda: 1e-6,
                _loss_fn_config_for_microbatch=lambda chunk, config: config,
                _aggregate_batch_metrics=lambda rows, **kw: {"loss": sum(r["loss"] for r in rows)},
            )
            if fail:
                with self.assertRaisesRegex(ValueError, "FB failed"):
                    await method(
                        trainer,
                        batch,
                        mini_batches,
                        {"loss_agg_mode": "sum"},
                        token_counts=counts,
                    )
                self.assertFalse(any(e == "opt" for e, _ in events))
                self.assertEqual([i for e, i in events if e == "done"], [0, 1])
            else:
                result = await method(
                    trainer,
                    batch,
                    mini_batches,
                    {"loss_agg_mode": "sum"},
                    token_counts=counts,
                )
                self.assertEqual(result["loss"], 10)
                self.assertEqual(result["_response_tokens"], sum(counts))
                self.assertEqual(result["_loss_weight"], 1)
                self.assertEqual(consumed, batch)
                self.assertEqual(sum(e == "done" for e, _ in events), len(mini_batches))
                self.assertEqual(events[-1], ("opt", None))
                self.assertEqual(sum(e == "opt" for e, _ in events), 1)

    async def test_recompute_cancellation_drains_without_retrying_failed_chunk(self):
        method = trainer_method("_recompute_logprobs")
        trajectories = [{"tokens": [i, 10, 20]} for i in range(3)]
        started, release = asyncio.Event(), asyncio.Event()
        calls, completed = [], []

        async def compute(chunk, *, sampling_masks):
            index = chunk[0]["tokens"][0]
            calls.append(index)
            if len(calls) == 2:
                started.set()
            await release.wait()
            completed.append(index)
            if index == 0:
                raise RuntimeError("forward failed while draining cancellation")
            return SimpleNamespace(logprobs=[[-0.1, -0.2]])

        trainer = SimpleNamespace(
            _service_holder=object(),
            _mis_metrics_enabled=False,
            _mis_enabled=False,
            _mini_batch_size=1,
            _submit_ahead=1,
            _sampling_mask_for_compute_logprobs=lambda traj: None,
            _compute_recompute_logprobs_batch=compute,
            _after_recompute_logprobs=lambda *args: self.fail("Cancelled results were published"),
        )
        task = asyncio.create_task(method(trainer, trajectories, {}))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(calls, [0, 1])
        self.assertEqual(sorted(completed), [0, 1])
        self.assertTrue(all("old_log_probs" not in trajectory for trajectory in trajectories))

    async def test_recompute_alignment_and_complete_batch_publication(self):
        method = trainer_method("_recompute_logprobs")
        trajectories = [{"tokens": [i, i + 10, i + 20]} for i in range(5)]
        live = peak = 0

        async def compute(chunk, *, sampling_masks):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            self.assertTrue(all("old_log_probs" not in d for d in trajectories))
            await asyncio.sleep(0.002 * (5 - chunk[0]["tokens"][0]))
            live -= 1
            return SimpleNamespace(logprobs=[[-d["tokens"][0], -0.5] for d in chunk])

        trainer = SimpleNamespace(
            _service_holder=object(),
            _mis_metrics_enabled=False,
            _mis_enabled=False,
            _mini_batch_size=2,
            _submit_ahead=1,
            _sampling_mask_for_compute_logprobs=lambda traj: None,
            _compute_recompute_logprobs_batch=compute,
            _after_recompute_logprobs=lambda *args: None,
        )
        result = await method(trainer, trajectories, {})
        self.assertIs(result, trajectories)
        self.assertEqual(peak, 2)
        self.assertEqual([d["old_log_probs"] for d in result], [[0.0, -i, -0.5] for i in range(5)])


if __name__ == "__main__":
    unittest.main()
