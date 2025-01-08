"""
Module to convert MaxText model weights to HuggingFace format.
"""

import os
from typing import Optional, Union, List, Dict, Optional, Any
import torch
import contextlib
import time
import jax
import numpy as np
from jaxtyping import Array
import json
from kithara.model.maxtext.ckpt_compatibility.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
    SHAPE_MAPPING,
)
from kithara.model.maxtext.ckpt_compatibility.model_configs import MODEL_CONFIGS
from kithara.utils.gcs_utils import (
    find_cache_root_dir,
    upload_file_to_gcs,
)
from kithara.utils.safetensor_utils import shard_checkpoint, SAFE_TENSORS_WEIGHTS_FILE, SAFE_TENSORS_INDEX_FILE
from jax.experimental import multihost_utils
from safetensors.torch import save_file
from concurrent.futures import ThreadPoolExecutor

def _apply_hook_fns(weight, target_shape, hook_fns):
    if hook_fns is None:
        return weight
    if not isinstance(hook_fns, list):
        hook_fns = [hook_fns]
    for hook_fn in hook_fns:
        weight = hook_fn(weight, target_shape)
    return weight


def _convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: Optional[str] = None
) -> torch.Tensor:
    expected_dtype = str(weight.dtype) if dtype is None else dtype
    weight = multihost_utils.process_allgather(weight)
    weight = np.array(weight, dtype="float32")
    torch_dtype = getattr(torch, expected_dtype)
    return torch.from_numpy(weight).to(torch_dtype)


def _transform_single_weight(
    weight: np.ndarray,
    target_shape: tuple,
    hook_fns: Union[callable, List[callable]],
):
    processed_weight = _apply_hook_fns(weight, target_shape, hook_fns)
    torch_weight = _convert_jax_weight_to_torch(processed_weight)
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
        processed_slice = _apply_hook_fns(weight_slice, target_shape, hook_fns)
        torch_slice = _convert_jax_weight_to_torch(processed_slice)
        sliced_weights.append(torch_slice)
    return sliced_weights


def _get_model_mappings(model_name: str, scan_layers: bool, config: dict):
    """Retrieves parameter, shape, and hook function mappings for the model."""
    return {
        "param_mapping": PARAM_MAPPING[model_name](config.to_dict(), scan_layers),
        "shape_mapping": SHAPE_MAPPING[model_name](config.to_dict()),
        "hook_fn_mapping": HOOK_FNS[model_name](
            config.to_dict(), scan_layers, saving_to_hf=True
        ),
    }


def _process_weight(variable, mappings):
    """Processes a single weight variable and returns transformed weights with their paths."""
    print(f"\n-> Processing {variable.path} with shape {variable.value.shape}...")
    weight_dict = {}

    hf_paths = mappings["param_mapping"][variable.path]
    if isinstance(hf_paths, str):
        hf_paths = [hf_paths]

    target_shape = mappings["shape_mapping"][hf_paths[0]]
    hook_fns = mappings["hook_fn_mapping"][variable.path]

    if len(hf_paths) == 1:
        # Single weight transformation
        weight = _transform_single_weight(
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


def _save_model_files(weight_arrays: Dict, config, output_dir: str, parallel_threads=8):
    """Saves model files (config and weights) to the specified directory."""
    start_time = time.time()
    print(f"\n-> Saving weights to {output_dir}...")

    local_dir = _get_local_directory(output_dir)

    # Save config.json
    _save_config_file(config, local_dir, output_dir)

    # Save .safetensors files
    shards, index = shard_checkpoint(weight_arrays)
    _save_weight_files(shards, index, local_dir, output_dir, parallel_threads)

    print(
        f"\n✅ Saving completed in {time.time() - start_time}. Model saved at `{output_dir}`."
    )


def _get_local_directory(output_dir: str) -> str:
    """Determines the local directory for saving files."""
    if output_dir.startswith("gs://"):
        local_dir = os.path.join(find_cache_root_dir(), "temp_ckpt")
        os.makedirs(local_dir, exist_ok=True)
        return local_dir
    return output_dir


def _save_config_file(config, local_dir: str, output_dir: str):
    """Saves the model configuration file."""
    local_path = os.path.join(local_dir, "config.json")
    config.to_json_file(local_path)
    if output_dir.startswith("gs://"):
        upload_file_to_gcs(
            local_path,
            os.path.join(output_dir, "config.json"),
            remove_local_file_after_upload=True,
        )


def _save_weight_files(
    shards, index, local_dir: str, output_dir: str, parallel_threads=8
):
    """Saves weight files and index if needed.

    Requires local system to have at least `parallel_threads * DEFAULT_MAX_SHARD_SIZE`
    free disk space, as each thread will maintain a local cache of its shard during processing.
    """

    def save_safetensor_file(state_dict, file_name):
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        if jax.process_index() == 0:
            local_path = os.path.join(local_dir, file_name)
            save_file(state_dict, local_path, metadata={"format": "pt"})
            if output_dir.startswith("gs://"):
                cloud_path = os.path.join(output_dir, file_name)
                upload_file_to_gcs(
                    local_path, cloud_path, remove_local_file_after_upload=True
                )

    if index is None:
        save_safetensor_file(shards, SAFE_TENSORS_WEIGHTS_FILE)
    else:
        # Save sharded weights in parallel
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            shard_items = list(shards.items())
            futures = [
                executor.submit(save_safetensor_file, shard_dict, shard_name)
                for shard_name, shard_dict in shard_items
            ]
            for future in futures:
                future.result()

        # Save index file
        local_path = os.path.join(local_dir, SAFE_TENSORS_INDEX_FILE)
        with open(local_path, "w") as f:
            json.dump(index, f)
        if output_dir.startswith("gs://"):
            cloud_path = os.path.join(output_dir, SAFE_TENSORS_INDEX_FILE)
            upload_file_to_gcs(
                local_path, cloud_path, remove_local_file_after_upload=True
            )


def _save_checkpoint(
    maxtext_model: "kithara.MaxTextModel", output_dir: str, parallel_threads=8
):
    """Main function to save a MaxText model checkpoint in HuggingFace format."""
    if maxtext_model.model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model {maxtext_model.model_name} is not supported. "
            f"Supported models are {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[maxtext_model.model_name]
    mappings = _get_model_mappings(
        maxtext_model.model_name, maxtext_model.scan_layers, config
    )

    # Process weights
    start_time = time.time()
    weight_arrays = {}
    for variable in maxtext_model.weights:
        weight_dict = _process_weight(variable, mappings)
        weight_arrays.update(weight_dict)
    print(
        f"\n✅ Weights converted into HuggingFace format in {time.time() - start_time}s"
    )

    # Save all model files
    _save_model_files(weight_arrays, config, output_dir, parallel_threads)


def save_maxtext_model_in_hf_format(
    model: "MaxTextModel", output_dir: str, dtype: str = "auto", parallel_threads=8
):
    """Convert and save a MaxText model in HuggingFace format.

    Args:
        model: MaxTextModel instance to save
        output_dir: Directory to save the HuggingFace checkpoint
        dtype: dtype for the converted model ("auto" uses source model's dtype)
    """

    @contextlib.contextmanager
    def _set_default_tensor_type(dtype: torch.dtype):
        """Sets the default torch dtype to the given dtype."""
        torch.set_default_dtype(dtype)
        yield
        torch.set_default_dtype(torch.float)

    if dtype == "auto":
        dtype = model.weight_dtype

    print(f"-> Saving model with {dtype=}...")
    with _set_default_tensor_type(getattr(torch, dtype)):
        _save_checkpoint(model, output_dir, parallel_threads)
