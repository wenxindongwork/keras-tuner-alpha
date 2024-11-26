import numpy as np
from typing import Union, List
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup, load_json
from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader
from keras_tuner.model.checkpoint_loader.maxtext.config import (
    GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING, GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN
)
from keras_tuner.model.checkpoint_loader.maxtext.utils import match_tensor_shape


class MaxTextSafetensorLoader(SafetensorLoader):
    def get_tensors(self, hf_weight_keys: List[str]):
        hf_tensors = []
        for hf_weight_key in hf_weight_keys:
            hf_tensor = self.get_tensor(hf_weight_key)
            hf_tensors.append(hf_tensor)
        return np.stack(hf_tensors, axis=1)

    def port_weight(
        self, keras_variable, hf_weight_key: Union[str | List[str]], hook_fn=None
    ):
        if isinstance(hf_weight_key, str):
            hf_tensor = self.get_tensor(hf_weight_key)
        else:
            hf_tensor = self.get_tensors(hf_weight_key)
        try:
            hf_tensor = match_tensor_shape(hf_tensor, keras_variable.shape)
        except Exception:
            print(
                f"{hf_weight_key=}, {hf_tensor.shape}, keras_variable={keras_variable.path} {keras_variable.shape}")
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_variable.shape))
        keras_variable.assign(hf_tensor)


def load_hf_weights_into_maxtext_model(preset_hande: str, maxtext_model):
    """Load weights from HuggingFace Hub into a MaxText model (an kithara.model.MaxTextModel instance)"""
    config = load_json(preset_hande)

    params_mapping = None
    hook_fn_mapping = None
    if config["model_type"] == "gemma2":
        params_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config)
        hook_fn_mapping = GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config)
    if params_mapping is None:
        raise ValueError(
            f"Model type {config['model_type']} is not current supported.")

    jax_memory_cleanup(maxtext_model)

    with MaxTextSafetensorLoader(preset_hande) as loader:
        for variable in maxtext_model.weights:
            assert (
                variable.path in params_mapping
            ), f"Variable path {variable.path} does not exist in the provided weight_mapping"
            hook_fn = None
            if variable.path in hook_fn_mapping:
                hook_fn = hook_fn_mapping[variable.path]
            try:
                loader.port_weight(
                    keras_variable=variable, hf_weight_key=params_mapping[
                        variable.path], hook_fn=hook_fn
                )
            except Exception as e:
                print(
                    f"Failed to load HF weight ({params_mapping[variable.path]}) into MaxText model({variable.path}). Error: {e}"
                )

    return maxtext_model
