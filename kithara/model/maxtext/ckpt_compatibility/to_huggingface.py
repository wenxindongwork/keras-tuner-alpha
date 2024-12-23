"""
Module to convert MaxText model weights to HuggingFace format.
"""
from typing import Optional, Union, List
import transformers
import torch
import contextlib
import os
import shutil
import numpy as np
from kithara.model.maxtext.ckpt_compatibility.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from google.cloud import storage
from kithara.utils.gcs_utils import upload_folder_to_gcs, find_cache_root_dir


gemma2_2b_config = transformers.Gemma2Config(
    num_hidden_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    hidden_size=2304,
    intermediate_size=9216,
)

gemma2_9b_config = transformers.Gemma2Config(
    num_hidden_layers=42,
    num_attention_heads=16,
    num_key_value_heads=8,
    hidden_size=3584,
    intermediate_size=14336,
    final_logit_softcapping=30.0,
    attn_logit_softcapping=50.0,
    head_dim=256,
    sliding_window=4096,
    query_pre_attn_scalar=224,
)

gemma2_27b_config = transformers.Gemma2Config(
    num_hidden_layers=46,
    num_attention_heads=32,
    num_key_value_heads=16,
    hidden_size=4608,
    intermediate_size=36864,
    final_logit_softcapping=30.0,
    attn_logit_softcapping=50.0,
    head_dim=128,
    sliding_window=4096,
    query_pre_attn_scalar=144,
)

MODEL_CONFIGS = {
    "gemma2-2b": gemma2_2b_config,
    "gemma2-9b": gemma2_9b_config,
    "gemma2-27b": gemma2_27b_config,
}


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def apply_hook_fns(keras_weight, target_shape, hook_fns):
    if hook_fns is None:
        return keras_weight
    if not isinstance(hook_fns, list):
        hook_fns = [hook_fns]
    for hook_fn in hook_fns:
        keras_weight = hook_fn(keras_weight, target_shape)
    return keras_weight


def _convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: Optional[str] = None
) -> torch.Tensor:
    expected_dtype = str(weight.dtype) if dtype is None else dtype
    weight = np.array(weight, dtype="float32")
    torch_dtype = getattr(torch, expected_dtype)
    return torch.from_numpy(weight).to(torch_dtype)


def _save_single_weight(
    module: torch.nn.Module,
    weight: np.ndarray,
    target_shape: tuple,
    hook_fns: Union[callable, List[callable]],
):
    processed_weight = apply_hook_fns(weight, target_shape, hook_fns)
    torch_weight = _convert_jax_weight_to_torch(processed_weight)
    module.state_dict()["weight"].copy_(torch_weight)


def _save_split_weights(
    modules: List[torch.nn.Module],
    weight: np.ndarray,
    target_shape: tuple,
    hook_fns: Union[callable, List[callable]],
):
    for i, module in enumerate(modules):
        weight_slice = weight.take(i, axis=1)
        processed_slice = apply_hook_fns(weight_slice, target_shape, hook_fns)
        torch_slice = _convert_jax_weight_to_torch(processed_slice)
        module.state_dict()["weight"].copy_(torch_slice)


def _save_checkpoint(maxtext_model: "kithara.MaxTextModel", output_dir):
    # Validate model type
    if maxtext_model.model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model {maxtext_model.model_name} is not supported. "
            f"Supported models are {list(MODEL_CONFIGS.keys())}"
        )

    # Get model configuration and mappings
    config = MODEL_CONFIGS[maxtext_model.model_name]
    param_mapping = PARAM_MAPPING[maxtext_model.model_name](
        config.to_dict(), maxtext_model.scan_layers
    )
    hook_fn_mapping = HOOK_FNS[maxtext_model.model_name](
        config.to_dict(), maxtext_model.scan_layers, saving_to_hf=True
    )

    print("-> Loading the transformer model ...")
    hf_model = transformers.Gemma2ForCausalLM(config)
    print(f"✅ Successfully loaded the transformer model")

    for variable in maxtext_model.weights:
        print(f"\n-> Processing {variable.path} with shape {variable.value.shape}...")

        # Get target paths and hooks
        hf_paths = param_mapping[variable.path]
        if isinstance(hf_paths, str):
            hf_paths = [hf_paths]

        # Clean up paths and get modules
        hf_paths = [path.strip(".weight") for path in hf_paths]
        hf_modules = [hf_model.get_submodule(path) for path in hf_paths]
        target_shape = hf_modules[0].state_dict()["weight"].shape

        # Save weights
        if len(hf_paths) == 1:
            _save_single_weight(
                hf_modules[0],
                variable.value,
                target_shape,
                hook_fn_mapping[variable.path],
            )
        else:
            _save_split_weights(
                hf_modules, variable.value, target_shape, hook_fn_mapping[variable.path]
            )

        print(f"✅ Successfully saved {variable.path}")

    print("\n✅ Weights converted successfully.")
    
    local_dir = output_dir
    if output_dir.startswith("gs://"):
        local_dir = find_cache_root_dir()
        local_dir = os.path.join(local_dir, "temp_ckpt")
        os.makedirs(local_dir, exist_ok=True)
        
    print(f"\n-> Saving HuggingFace model to `{local_dir}`...")

    hf_model.save_pretrained(local_dir)

    print(f"\n✅ Saving complete. Model saved at `{local_dir}`.")
    
    if output_dir.startswith("gs://"):
        print(f"\n-> Uploading `{local_dir}` to `{output_dir}`...")
        upload_folder_to_gcs(local_dir, output_dir)
        print(f"\n✅ Saving complete. Model saved at `{output_dir}`.")

        # Delete local cache
        print(f"\n-> Deleting local cache at `{local_dir}`...")
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"\n✅ Cache deleted.")

        
def save_maxtext_model_in_hf_format(
    model: "MaxTextModel", output_dir: str, dtype: str = "auto"
):
    """Convert and save a MaxText model in HuggingFace format.

    Args:
        model: MaxTextModel instance to save
        output_dir: Directory to save the HuggingFace checkpoint
        dtype: dtype for the converted model ("auto" uses source model's dtype)
    """
    if dtype == "auto":
        dtype = model.weight_dtype

    print(f"-> Saving model with {dtype=}...")
    with _set_default_tensor_type(getattr(torch, dtype)):
        _save_checkpoint(model, output_dir)
