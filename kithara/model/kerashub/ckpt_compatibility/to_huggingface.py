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
Module to convert KerasHub model weights to HuggingFace format.
"""

import torch
import contextlib
import time
from kithara.model.hf_compatibility import (
    MODEL_CONFIGS,
    process_weight,
    save_lora_files,
    transform_single_weight,
    save_model_files,
)
from kithara.model.kerashub.ckpt_compatibility.param_mapping import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LORA_BASE_SUFFIX,
    HF_LORA_A_SUFFIX,
    HF_LORA_B_SUFFIX,
    HOOK_FNS,
    PARAM_MAPPING
)
from kithara.model.hf_compatibility import SHAPE_MAPPING
from peft import LoraConfig
import jax

def _get_model_mappings(model_name: str, scan_layers: bool, config: dict):
    """Retrieves parameter, shape, and hook function mappings for the model."""
    return {
        "param_mapping": PARAM_MAPPING[model_name](config.to_dict()),
        "shape_mapping": SHAPE_MAPPING[model_name](config.to_dict()),
        "hook_fn_mapping": HOOK_FNS[model_name](config.to_dict()),
    }

def _save_checkpoint(
    model: "kithara.KerasHubModel",
    output_dir: str,
    parallel_threads=8,
    only_save_adapters=False,
    save_adapters_separately=False,
):
    """Main function to save a KerasHub model checkpoint in HuggingFace format."""

    if only_save_adapters:
        save_adapters_separately = True

    # Validate model type
    if model.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model.model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model.model_name]
    mappings = _get_model_mappings(model.model_name, model.scan_layers, config)
    
    start_time = time.time()
    weight_arrays = {}
    adapter_weights_arrays = {}
    target_modules = set()

    # Process regular model weights
    if not only_save_adapters:
        for variable in model.weights:
            if variable.path.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)):
                continue
            weight_arrays.update(process_weight(variable, mappings))

    # Process LoRA weights
    for var_a in model.weights:
        if not var_a.path.endswith(LORA_A_SUFFIX):
            continue
        # Find corresponding LoRA B weights
        var_b = next(w for w in model.weights 
                    if w.path == var_a.path.replace(LORA_A_SUFFIX, LORA_B_SUFFIX))
        
        base_path = var_a.path.replace(LORA_A_SUFFIX, LORA_BASE_SUFFIX)
        hf_base_path = mappings["param_mapping"][base_path]
        target_modules.add(hf_base_path.rstrip(".weight")) 

        target_shape = mappings["shape_mapping"][hf_base_path]
        hook_fns = mappings["hook_fn_mapping"][base_path]
        
        weight_a = var_a.value # [n_head, hidden_dim, r]
        weight_b = var_b.value # [r, head_dim]

        lora_mask = weight_a @ weight_b
        lora_mask = transform_single_weight(lora_mask, target_shape, hook_fns)

        if save_adapters_separately:
            # Decompose using QR factorization
            Q, R = torch.linalg.qr(lora_mask)
            lora_r = weight_b.shape[0]
            
            # Store decomposed weights
            lora_path_a = f"base_model.model.{hf_base_path.rstrip('.weight')}{HF_LORA_A_SUFFIX}"
            lora_path_b = f"base_model.model.{hf_base_path.rstrip('.weight')}{HF_LORA_B_SUFFIX}"

            adapter_weights_arrays[lora_path_a] = R[:lora_r, :].contiguous() #[r, hidden_dim]
            adapter_weights_arrays[lora_path_b] = Q[:, :lora_r].contiguous() #[n_head * head_dim, r]
            
        else:
            # Merge LoRA weights with base model
            assert lora_mask.shape == weight_arrays[hf_base_path].shape
            weight_arrays[hf_base_path] = weight_arrays[hf_base_path] + lora_mask

    print(f"✅ Conversion completed in {time.time() - start_time:.2f}s")

    # Save weights
    lora_config = LoraConfig(
        r=model.lora_rank,
        lora_alpha=model.lora_rank,
        target_modules=list(target_modules)
    ) if model.lora_rank else None

    if only_save_adapters:
        save_lora_files(lora_config, adapter_weights_arrays, output_dir)
    else:
        save_model_files(weight_arrays, config, output_dir, 
                        parallel_threads=parallel_threads)
        if save_adapters_separately:
            save_lora_files(lora_config, adapter_weights_arrays, output_dir)


def save_kerashub_model_in_hf_format(
    model: "KerasHubModel",
    output_dir: str,
    dtype: str = "auto",
    parallel_threads=8,
    only_save_adapters=False,
    save_adapters_separately=False,
):
    """Convert and save a KerasHubModel model in HuggingFace format.

    Args:
        model: KerasHubModel instance to save
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
        _save_checkpoint(
            model,
            output_dir,
            parallel_threads,
            only_save_adapters,
            save_adapters_separately,
        )

    jax.experimental.multihost_utils.sync_global_devices("saving_completed")