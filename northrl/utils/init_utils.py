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

import json  # pylint: disable=unused-import


def create_inference_service_client(backend: str, config, **kwargs):
    """Create an inference service client.

    New schema: ``backend.type`` is constrained to ``weaver`` or ``mock`` by
    the pydantic schema. Mock backend doesn't need a holder; weaver gets
    one threaded through via ``weaver_service_holder`` kwarg.

    Args:
        backend: ``"weaver"`` or ``"mock"``.
        config: Root NorthRLConfig.
        **kwargs: ``weaver_service_holder`` (weaver backend only).

    Returns:
        An InferenceServiceClient instance.
    """
    if backend == "weaver":
        from ..weaver import (  # pylint: disable=relative-beyond-top-level
            WeaverInferenceServiceClient,
        )

        weaver_service_holder = kwargs.get("weaver_service_holder")
        if weaver_service_holder is None:
            raise ValueError(
                "backend='weaver' requires a weaver_service_holder kwarg "
                "(threaded through from the controller's WeaverServiceHolder)."
            )
        return WeaverInferenceServiceClient(
            config=config,
            weaver_service_holder=weaver_service_holder,
        )
    if backend == "mock":
        from ..mock.mock_inference_service_client import (  # pylint: disable=relative-beyond-top-level
            MockInferenceServiceClient,
        )

        return MockInferenceServiceClient(config)
    raise NotImplementedError(f"Inference service backend {backend!r} not supported")
