from typing import Optional
import transformers 
import torch
import contextlib
import os
import numpy as np
from keras_tuner.model.ckpt_compatibility.maxtext.config import (
    GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING, GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN
)

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

CONFIG_MAPPING = {
    "gemma2-2b": gemma2_2b_config,
    "gemma2-9b": gemma2_9b_config,
    "gemma2-27b": gemma2_27b_config
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
        keras_weight = hook_fn(keras_weight, target_shape, saving_to_hf=True)
    return keras_weight
    
def save_checkpoint(model_name:str, maxtext_model: 'model.MaxTextModel', output_dir, scan_layers= False):
    assert model_name in CONFIG_MAPPING, f"model_name is not one of {CONFIG_MAPPING.keys()}"
    config = CONFIG_MAPPING[model_name]
    print("-> Loading the transformer model ...")
    hf_model = transformers.Gemma2ForCausalLM(config)
    print(f"✅ Successfully loaded the transformer model")

    param_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config.to_dict(), scan_layers)
    hook_fn_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config.to_dict(), scan_layers)

    for variable in maxtext_model.weights:
        
        maxtext_weight = variable.value
        hf_weight_keys=param_mapping[variable.path]
        
        hook_fns = hook_fn_mapping[variable.path]
        
        print(f"\n-> Saving `{variable.path}` with shape {maxtext_weight.shape}...")

        if isinstance(hf_weight_keys, str):
            hf_module_path= hf_weight_keys.strip(".weight")
            hf_module = hf_model.get_submodule(hf_module_path)
            target_shape = hf_module.state_dict()["weight"].shape
            maxtext_weight = apply_hook_fns(maxtext_weight, target_shape, hook_fns)
            print("maxtext_weight dtype original", maxtext_weight.dtype)
            dtype = maxtext_weight.dtype
            maxtext_weight = np.asarray(maxtext_weight, dtype="float32")
            maxtext_weight = torch.from_numpy(maxtext_weight).to(getattr(torch, dtype))
            print("maxtext_weight dtype being saved", maxtext_weight.dtype)
            hf_module.state_dict()["weight"].copy_(maxtext_weight)
            
        elif isinstance(hf_weight_keys, list):

            n_layers = len(hf_weight_keys)
            hf_module_path= [key.strip(".weight") for key in hf_weight_keys]
            hf_module_list = [hf_model.get_submodule(path) for path in hf_module_path]
            target_shape = hf_module_list[0].state_dict()["weight"].shape

            for i in range(n_layers):
                maxtext_weight_slice = apply_hook_fns(maxtext_weight.take(i, axis=1), target_shape, hook_fns)
                dtype = maxtext_weight_slice.dtype
                maxtext_weight_slice= np.asarray(maxtext_weight_slice, dtype="float32")
                
                maxtext_weight_slice = torch.from_numpy(maxtext_weight_slice, dtype=dtype) 
                hf_module_list[i].state_dict()["weight"].copy_(maxtext_weight_slice)
        
        print(f"\n✅ Successfully saved {hf_module_path}")

    print("\n✅ Weights converted successfully.")
    print(f"\n-> Saving HuggingFace model to `{output_dir}`...")

    # Save model to HF Transformers format
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)

    print(f"\n✅ Saving complete. Model saved at `{output_dir}`.")


def save_maxtext_model_in_hf_format(model_name:str, model: "MaxTextModel", output_dir:str, dtype: str = "auto", scan_layers=False):

    if dtype== "auto":
        dtype = model.weight_dtype 

    print(f"Saving model with {dtype=}")

    with _set_default_tensor_type(getattr(torch, dtype)):
        save_checkpoint(model_name, model, output_dir, scan_layers)


# python3 -m pip install --upgrade torch torchaudio torchvision