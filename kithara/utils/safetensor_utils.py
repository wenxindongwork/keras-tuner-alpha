"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from typing import Optional, List, Dict, Tuple, Optional, Callable
from jaxtyping import Array
import numpy as np
import contextlib
import safetensors
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.preset_utils import (
    check_file_exists,
    get_file,
    load_json,
)

SAFE_TENSORS_CONFIG_FILE = "config.json"
SAFE_TENSORS_PEFT_CONFIG_FILE = "adapter_config.json"
SAFE_TENSORS_LORA_WEIGHTS_FILE = "adapter_model.safetensors"
SAFE_TENSORS_WEIGHTS_FILE = "model.safetensors"
SAFE_TENSORS_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024 * 3  # 3GB default


def shard_checkpoint(
    weights_dict: Dict[str, Array],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    weights_name: str = "model.safetensors",
) -> Tuple[Dict[str, Dict[str, Array]], Optional[Dict]]:
    """Shards a model checkpoint into smaller pieces based on size constraints.

    Args:
        weights_dict: Model weights dictionary to shard
        max_shard_size: Maximum size in bytes for each shard
        weights_name: Base filename for the shards

    Returns:
        Tuple of (sharded weights dict, optional index dict)
        Index contains metadata and weight mapping information
    """
    # Track current shard and accumulated sizes
    current_shard: Dict[str, Array] = {}
    shards: List[Dict[str, Array]] = [current_shard]
    current_size = 0
    total_size = 0

    # Iterate through weights in sorted order for deterministic sharding
    for key, tensor in sorted(weights_dict.items()):
        weight_size = tensor.numel() * tensor.itemsize
        # Start new shard if current one would exceed size limit
        if (current_size + weight_size > max_shard_size) and len(current_shard.items()):
            current_shard = {}
            shards.append(current_shard)
            current_size = 0

        # Add weight to current shard and update sizes
        current_shard[key] = tensor
        current_size += weight_size
        total_size += weight_size

    # Return single shard without index if no sharding needed
    if len(shards) == 1:
        return {weights_name: shards[0]}, None

    # Generate shard filenames and build index
    shard_dict = {}
    weight_map = {}

    for idx, shard in enumerate(shards, 1):
        # Create numbered shard filename
        shard_name = weights_name.replace(
            ".safetensors", f"-{idx:05d}-of-{len(shards):05d}.safetensors"
        )
        shard_dict[shard_name] = shard

        # Map each weight to its shard file
        for key in shard:
            weight_map[key] = shard_name

    return shard_dict, {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }


class MaxTextSafetensorLoader(contextlib.ExitStack):
    def __init__(self, preset, prefix=None, fname=None):
        super().__init__()

        self.preset = preset
        if check_file_exists(preset, SAFE_TENSORS_INDEX_FILE):
            self.safetensor_config = load_json(preset, SAFE_TENSORS_INDEX_FILE)
        else:
            self.safetensor_config = None
        self.safetensor_files = {}
        self.prefix = prefix

        if fname is not None and self.safetensor_config is not None:
            raise ValueError(
                f"Cannot specify `fname` if {SAFE_TENSORS_INDEX_FILE} exists. "
                f"Received: fname={fname}"
            )
        self.fname = fname  # Specify the name of the safetensor file.

    def get_prefixed_key(self, hf_weight_key, dict_like):
        """
        Determine and return a prefixed key for a given hf weight key.

        This method checks if there's a common prefix for the weight keys and caches it
        for future use.

        Args:
            hf_weight_key (str): The hf weight key to check for a prefix.
            dict_like (object): An object to get keys of safetensor file using keys() method.

        Returns:
            str: The full key including the prefix (if any).
        """
        if self.prefix is not None:
            return self.prefix + hf_weight_key

        for full_key in dict_like.keys():
            if full_key.endswith(hf_weight_key) and full_key != hf_weight_key:
                self.prefix = full_key[: -len(hf_weight_key)]
                return full_key

        self.prefix = ""
        return hf_weight_key

    def get_tensors(
        self, hf_weight_keys: List[str], hook_fn: List[Callable], target_shape
    ):
        """
        Loads and processes multiple tensors.

        Args:
            hf_weight_keys: List of weight keys to load
            hook_fn: List of processing functions to apply
            target_shape: Expected shape of the output tensor

        Returns:
            Stacked array of processed tensors
        """
        hf_tensors = []
        for hf_weight_key in hf_weight_keys:
            hf_tensor = self.get_tensor(hf_weight_key)
            for fn in hook_fn:
                hf_tensor = fn(hf_tensor, target_shape)
            hf_tensors.append(hf_tensor)
        return np.stack(hf_tensors, axis=1)

    def get_tensor(self, hf_weight_key):
        """
        Loads a single tensor from the safetensor file.

        Args:
            hf_weight_key: Key of the tensor to load

        Returns:
            The loaded tensor
        """
        if self.safetensor_config is None:
            fname = self.fname if self.fname is not None else SAFE_TENSORS_WEIGHTS_FILE
        else:
            full_key = self.get_prefixed_key(
                hf_weight_key, self.safetensor_config["weight_map"]
            )
            fname = self.safetensor_config["weight_map"][full_key]

        if fname in self.safetensor_files:
            file = self.safetensor_files[fname]
        else:
            path = get_file(self.preset, fname)
            file = self.enter_context(safetensors.safe_open(path, framework="np"))
            self.safetensor_files[fname] = file

        full_key = self.get_prefixed_key(hf_weight_key, file)
        return file.get_tensor(full_key)
