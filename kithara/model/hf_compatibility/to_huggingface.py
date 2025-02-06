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

"""
Utilities to convert Kithara model weights to HuggingFace format.
"""

import os
from typing import Union, List, Dict
import time
import jax
import numpy as np
from jaxtyping import Array
import json
from kithara.utils.gcs_utils import (
    find_cache_root_dir,
    upload_file_to_gcs,
)
from kithara.utils.safetensor_utils import (
    shard_checkpoint,
    SAFE_TENSORS_WEIGHTS_FILE,
    SAFE_TENSORS_INDEX_FILE,
    SAFE_TENSORS_CONFIG_FILE,
    SAFE_TENSORS_LORA_WEIGHTS_FILE,
    SAFE_TENSORS_PEFT_CONFIG_FILE,
)
from kithara.utils.torch_utils import convert_jax_weight_to_torch
from safetensors.torch import save_file
from concurrent.futures import ThreadPoolExecutor
from peft import LoraConfig, PeftConfig


def apply_hook_fns(weight, target_shape, hook_fns):
    if hook_fns is None:
        return weight
    if not isinstance(hook_fns, list):
        hook_fns = [hook_fns]
    for hook_fn in hook_fns:
        weight = hook_fn(weight, target_shape)
    return weight


def transform_single_weight(
    weight: np.ndarray,
    target_shape: tuple,
    hook_fns: Union[callable, List[callable]],
):
    processed_weight = apply_hook_fns(weight, target_shape, hook_fns)
    torch_weight = convert_jax_weight_to_torch(processed_weight)
    return torch_weight


def _transform_stacked_weights(
    num_modules: int,
    weight: np.ndarray,
    target_shape: tuple,
    hook_fns: Union[callable, List[callable]],
):
    sliced_weights = []
    for i in range(num_modules):
        weight_slice = weight.take(i, axis=1)
        processed_slice = apply_hook_fns(weight_slice, target_shape, hook_fns)
        torch_slice = convert_jax_weight_to_torch(processed_slice)
        sliced_weights.append(torch_slice)
    return sliced_weights


def process_weight(variable, mappings, debug=False):
    """Processes a single weight variable and returns transformed weights with their paths."""
    if debug:
        print(f"-> Processing {variable.path} with shape {variable.value.shape}...")
    weight_dict = {}

    # Get the final path from the absolution path
    variable_path = variable.path
    if variable.path.startswith("max_text_layer"):
        variable_path = variable.path.split("/")[-1]
    
    hf_paths = mappings["param_mapping"][variable_path]
    if isinstance(hf_paths, str):
        hf_paths = [hf_paths]

    target_shape = mappings["shape_mapping"][hf_paths[0]]
    hook_fns = (
        mappings["hook_fn_mapping"][variable_path]
        if variable_path in mappings["hook_fn_mapping"]
        else None
    )

    if len(hf_paths) == 1:
        # Single weight transformation
        weight = transform_single_weight(
            variable.value,
            target_shape,
            hook_fns,
        )
        weight_dict[hf_paths[0]] = weight
    else:
        # Stacked weights transformation
        weights = _transform_stacked_weights(
            len(hf_paths), variable.value, target_shape, hook_fns
        )
        for path, weight in zip(hf_paths, weights):
            weight_dict[path] = weight

    return weight_dict


def save_lora_files(
    lora_config: LoraConfig,
    adapter_weight_arrays,
    output_dir: str,
):
    if lora_config == None:
        print("WARNING: There is no LoRA adapter to be saved. ")
        return
    local_dir = _get_local_directory(output_dir)
    # Save adapter_config.json
    save_peft_config_file(lora_config, local_dir, output_dir)
    # Save adapter_model.safetensors
    save_safetensor_file(
        adapter_weight_arrays, local_dir, output_dir, SAFE_TENSORS_LORA_WEIGHTS_FILE
    )


def save_model_files(weight_arrays: Dict, config, output_dir: str, parallel_threads=8):
    """Saves model files (config and weights) to the specified directory."""
    start_time = time.time()
    print(f"\n-> Saving weights to {output_dir}...")

    local_dir = _get_local_directory(output_dir)

    # Save config.json
    save_config_file(config, local_dir, output_dir, SAFE_TENSORS_CONFIG_FILE)

    # Save .safetensors files
    shards, index = shard_checkpoint(weight_arrays)
    save_weight_files(shards, index, local_dir, output_dir, parallel_threads)

    print(
        f"\n✅ Saving completed in {time.time() - start_time}. Model saved at `{output_dir}`."
    )


def _get_local_directory(output_dir: str) -> str:
    """Determines the local directory for saving files."""
    local_dir = output_dir
    if local_dir.startswith("gs://"):
        local_dir = os.path.join(find_cache_root_dir(), "temp_ckpt")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def save_index_file(index: dict, local_dir: str, output_dir: str, file_name: str):
    """Saves the model index json file (model.safetensors.index.json)."""
    if jax.process_index() == 0:
        local_path = os.path.join(local_dir, file_name)
        with open(local_path, "w") as f:
            json.dump(index, f)
        if output_dir.startswith("gs://"):
            upload_file_to_gcs(
                local_path,
                os.path.join(output_dir, file_name),
                remove_local_file_after_upload=True,
            )


def save_config_file(config, local_dir: str, output_dir: str, file_name: str):
    """Saves the model configuration file(config.json)."""
    if jax.process_index() == 0:
        local_path = os.path.join(local_dir, file_name)
        config.to_json_file(local_path)
        if output_dir.startswith("gs://"):
            upload_file_to_gcs(
                local_path,
                os.path.join(output_dir, file_name),
                remove_local_file_after_upload=True,
            )


def save_peft_config_file(config: PeftConfig, local_dir: str, output_dir: str):
    """Saves the model configuration file."""
    if jax.process_index() == 0:
        local_path = os.path.join(local_dir, SAFE_TENSORS_PEFT_CONFIG_FILE)
        config.save_pretrained(local_dir)
        if output_dir.startswith("gs://"):
            upload_file_to_gcs(
                local_path,
                os.path.join(output_dir, SAFE_TENSORS_PEFT_CONFIG_FILE),
                remove_local_file_after_upload=True,
            )


def save_safetensor_file(state_dict, local_dir, output_dir, file_name):
    """Saves a single safetensor file."""
    if jax.process_index() == 0:
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        local_path = os.path.join(local_dir, file_name)
        save_file(state_dict, local_path, metadata={"format": "pt"})
        if output_dir.startswith("gs://"):
            cloud_path = os.path.join(output_dir, file_name)
            upload_file_to_gcs(
                local_path, cloud_path, remove_local_file_after_upload=True
            )


def save_weight_files(
    shards, index, local_dir: str, output_dir: str, parallel_threads=8
):
    """Saves weight files and index if needed.

    Requires local system to have at least `parallel_threads * DEFAULT_MAX_SHARD_SIZE`
    free disk space, as each thread will maintain a local cache of its shard during processing.
    """
    if index is None:
        save_safetensor_file(shards, local_dir, output_dir, SAFE_TENSORS_WEIGHTS_FILE)
    else:
        # Save sharded weights in parallel
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            shard_items = list(shards.items())
            futures = [
                executor.submit(
                    save_safetensor_file, shard_dict, local_dir, output_dir, shard_name
                )
                for shard_name, shard_dict in shard_items
            ]
            for future in futures:
                future.result()

        # Save index file
        save_index_file(index, local_dir, output_dir, SAFE_TENSORS_INDEX_FILE)
