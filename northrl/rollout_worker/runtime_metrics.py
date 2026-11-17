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

"""Per-execution inference observations for ordinary rollout workers."""

from typing import Any

from northrl.remote_rollout.metrics import RolloutMetrics


class ObservedInferenceClient:
    """Delegate a client's API while measuring logical generation calls only.

    A fresh collector is installed for each worker execution. Tokenization and
    evaluation are not model-generation requests; backend-internal retries remain
    part of one logical call's latency.
    """

    def __init__(self, client: Any, metrics: RolloutMetrics) -> None:
        self._client = client
        self.metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("generate", *args, **kwargs)

    async def completion(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("completion", *args, **kwargs)

    async def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("tools"):
            self.metrics.enable_tool_observations()
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            self.metrics.observe_messages("openai", messages)
        with self.metrics.track_request(operation) as request:
            result = await getattr(self._client, operation)(*args, **kwargs)
            if isinstance(result, dict):
                request.result = result
        if isinstance(result, dict):
            self.metrics.record_model_result(result)
            choices = result.get("choices") or []
            message = (
                choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
            )
            calls = message.get("tool_calls") or [] if isinstance(message, dict) else []
            if isinstance(calls, list):
                with self.metrics.emitted_tool_calls(len(calls)):
                    self.metrics.record_emitted_ids(result)
        return result
