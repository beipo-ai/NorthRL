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
Weaver integration for NorthRL framework.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .weaver_inference_service_client import WeaverInferenceServiceClient
    from .weaver_service_holder import WeaverServiceHolder

_EXPORTS = {
    "WeaverInferenceServiceClient": (
        ".weaver_inference_service_client",
        "WeaverInferenceServiceClient",
    ),
    "WeaverServiceHolder": (".weaver_service_holder", "WeaverServiceHolder"),
}

__all__ = [
    "WeaverInferenceServiceClient",
    "WeaverServiceHolder",
]


def __getattr__(name: str) -> Any:
    """Load Weaver integrations only when callers request them.

    Keeping the package initializer dependency-free lets consumers import the
    training-only service holder without also loading the inference client's
    configuration and OpenAI dependencies.
    """

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
