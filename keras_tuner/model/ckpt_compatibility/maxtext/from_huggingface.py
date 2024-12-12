import numpy as np
from typing import Union, List, Optional , Callable
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup, load_json
from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader
from keras_tuner.model.ckpt_compatibility.maxtext.config import (
    GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING, GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN
)

class MaxTextSafetensorLoader(SafetensorLoader):
    def get_tensors(self, hf_weight_keys: List[str], hook_fn: List[Callable], target_shape):
        hf_tensors = []
        for hf_weight_key in hf_weight_keys:
            hf_tensor = self.get_tensor(hf_weight_key)
            for fn in hook_fn:
                hf_tensor = fn(hf_tensor, target_shape, saving_to_hf=False)
            hf_tensors.append(hf_tensor)
        return np.stack(hf_tensors, axis=1)

    def port_weight(
        self, keras_variable, hf_weight_key: Union[str | List[str]], hook_fn=Optional[Union[List | Callable]], scan_layers=False, expected_dtype = None
    ):
        target_shape = list(keras_variable.shape)
        target_is_stacked = scan_layers and isinstance(hf_weight_key, list)
       
        if target_is_stacked:
            target_shape = (target_shape[0], *target_shape[2:])
        
        if hook_fn:
            if not isinstance(hook_fn, list):
                hook_fn = [hook_fn]
        else:
            hook_fn = []

        if isinstance(hf_weight_key, str):
            hf_tensor = self.get_tensor(hf_weight_key)
            for fn in hook_fn:
                hf_tensor = fn(hf_tensor, target_shape, saving_to_hf=False)
        else:
            hf_tensor = self.get_tensors(hf_weight_key, hook_fn, target_shape)
        
        if expected_dtype:
            hf_tensor = hf_tensor.astype(expected_dtype)
        
        keras_variable.assign(hf_tensor)


def load_hf_weights_into_maxtext_model(preset_hande: str, maxtext_model, scan_layers = False):
    """Load weights from HuggingFace Hub into a MaxText model (an kithara.model.MaxTextModel instance)"""
    config = load_json(preset_hande)

    params_mapping = None
    hook_fn_mapping = None
    if config["model_type"] == "gemma2":
        params_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers)
        hook_fn_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers)
    if params_mapping is None:
        raise ValueError(
            f"Model type {config['model_type']} is not current supported.")

    jax_memory_cleanup(maxtext_model)

    with MaxTextSafetensorLoader(preset_hande) as loader:
        for variable in maxtext_model.weights:
            print(f"-> Loading weight ({variable.path}) into model...")
            expected_dtype = variable.value.dtype
            assert (
                variable.path in params_mapping
            ), f"Variable path {variable.path} does not exist in the provided weight_mapping"
            hook_fn = None
            if variable.path in hook_fn_mapping:
                hook_fn = hook_fn_mapping[variable.path]
            try:
                loader.port_weight(
                    keras_variable=variable, hf_weight_key=params_mapping[
                        variable.path], hook_fn=hook_fn, scan_layers=scan_layers, expected_dtype = expected_dtype
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load HF weight ({params_mapping[variable.path]}) into MaxText model({variable.path}). Error: {e}"
                )
            assert variable.value.dtype == expected_dtype, f"Expected weight dtype is {expected_dtype}, but weight dtype is {variable.value.dtype}"
            print(f"✅ Successfull loaded weight ({variable.path}) into model. ")
    return maxtext_model
