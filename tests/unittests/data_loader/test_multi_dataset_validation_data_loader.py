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

from omegaconf import DictConfig

from northrl.data_loader.multi_dataset_data_loader import MultiDatasetValidationDataLoader


def test_empty_datasets_are_a_finished_noop() -> None:
    loader = MultiDatasetValidationDataLoader(DictConfig({"datasets": {}}))

    assert loader.is_finished()
    assert not loader.can_return_item()
    assert loader.get_next_item() is None
