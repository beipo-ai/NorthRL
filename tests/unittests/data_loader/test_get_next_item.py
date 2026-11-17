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
Simple test to see what get_next_item returns from SequentialDataLoader
"""

import json

from omegaconf import OmegaConf

from northrl.mock import MockDataLoader


def test_get_next_item_output():
    """Test to see what get_next_item returns"""
    # Create a simple config
    config = OmegaConf.create(
        {
            "batch_size": 3,
            "mock_api_type": "completion",  # or "generate" for message format
            "rollout_repeat_n": 1,  # Set to 2 or more to see repetition
        }
    )

    # Create the data loader
    loader = MockDataLoader(config)

    print("\n" + "=" * 80)
    print("Testing SequentialDataLoader.get_next_item()")
    print("=" * 80 + "\n")

    # Get a few items and print them
    for i in range(5):
        print(f"\n--- Item {i+1} ---")
        item = loader.get_next_item()

        if item is None:
            print("get_next_item() returned None")
            print(f"is_finished(): {loader.is_finished()}")
            break

        print(f"Type: {type(item)}")
        print(f"Keys: {list(item.keys())}")
        print("\nFull item (formatted):")
        print(json.dumps(item, indent=2, default=str))

        # Show specific fields if they exist
        if "prompt" in item:
            print(f"\nprompt type: {type(item['prompt'])}")
            if isinstance(item["prompt"], str):
                print(f"prompt (first 100 chars): {item['prompt'][:100]}...")
            else:
                print(f"prompt: {item['prompt']}")

        if "group_id" in item:
            print(f"group_id: {item['group_id']}")

        if "run_id" in item:
            print(f"run_id: {item['run_id']}")

        print(f"\nLoader state:")
        print(f"  _data_index: {loader._data_index}")
        print(f"  _buffer_index: {loader._buffer_index}")
        print(f"  buffer length: {len(loader._data_buffer)}")
        print(f"  can_return_item(): {loader.can_return_item()}")
        print(f"  is_finished(): {loader.is_finished()}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80 + "\n")


def test_get_next_item_with_repeat():
    """Test get_next_item with rollout_repeat_n to see repetition"""
    config = OmegaConf.create(
        {
            "batch_size": 2,
            "mock_api_type": "completion",
            "rollout_repeat_n": 3,  # Repeat each item 3 times
        }
    )

    loader = MockDataLoader(config)

    print("\n" + "=" * 80)
    print("Testing get_next_item with rollout_repeat_n=3")
    print("=" * 80 + "\n")

    for i in range(9):  # Should see 2 items repeated 3 times each = 6 items
        item = loader.get_next_item()
        if item is None:
            break

        print(
            f"Item {i+1}: id={item.get('id')}, run_id={item.get('run_id')}, "
            f"group_id={item.get('group_id', 'N/A')[:8]}..."
        )

    print("\n" + "=" * 80 + "\n")


def test_get_next_item_generate_format():
    """Test get_next_item with generate API type (message format)"""
    config = OmegaConf.create(
        {
            "batch_size": 2,
            "mock_api_type": "generate",  # Will return message format
        }
    )

    loader = MockDataLoader(config)

    print("\n" + "=" * 80)
    print("Testing get_next_item with generate API type (message format)")
    print("=" * 80 + "\n")

    item = loader.get_next_item()
    if item:
        print("Item keys:", list(item.keys()))
        print("\nprompt type:", type(item.get("prompt")))
        print("prompt:", item.get("prompt"))
        if hasattr(item.get("prompt"), "tolist"):
            print("prompt (as list):", item.get("prompt").tolist())

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Run the tests
    test_get_next_item_output()
    test_get_next_item_with_repeat()
    test_get_next_item_generate_format()
