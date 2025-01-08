"""
This module provides functionality to load weights from HuggingFace's 
checkpoint into MaxText models.
"""

import numpy as np
import contextlib
import safetensors
from typing import Union, List, Optional, Callable
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup, load_json
from kithara.model.maxtext.ckpt_compatibility.param_mapping import (
    PARAM_MAPPING,
    HOOK_FNS,
)
from kithara.model.maxtext.ckpt_compatibility.utils import (
    get_maxtext_model_name_from_hf_handle,
)
from keras_hub.src.utils.preset_utils import (
    SAFETENSOR_CONFIG_FILE,
    SAFETENSOR_FILE,
    check_file_exists,
    get_file,
    load_json,
)


class MaxTextSafetensorLoader(contextlib.ExitStack):
    def __init__(self, preset, prefix=None, fname=None):
        super().__init__()

        self.preset = preset
        if check_file_exists(preset, SAFETENSOR_CONFIG_FILE):
            self.safetensor_config = load_json(preset, SAFETENSOR_CONFIG_FILE)
        else:
            self.safetensor_config = None
        self.safetensor_files = {}
        self.prefix = prefix

        if fname is not None and self.safetensor_config is not None:
            raise ValueError(
                f"Cannot specify `fname` if {SAFETENSOR_CONFIG_FILE} exists. "
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
            fname = self.fname if self.fname is not None else SAFETENSOR_FILE
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

    def port_weight(
        self,
        keras_variable,
        hf_weight_key: Union[str | List[str]],
        hook_fn=Optional[Union[List | Callable]],
        scan_layers=False,
        expected_dtype=None,
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
                hf_tensor = fn(hf_tensor, target_shape)
        else:
            hf_tensor = self.get_tensors(hf_weight_key, hook_fn, target_shape)

        if expected_dtype:
            hf_tensor = hf_tensor.astype(expected_dtype)

        keras_variable.assign(hf_tensor)


def load_hf_weights_into_maxtext_model(
    preset_handle: str, maxtext_model: "kithara.MaxtextModel", scan_layers=False
):
    """
    Loads weights from HuggingFace Hub into a MaxText model.

    Args:
        preset_handle: HuggingFace model preset identifier
        maxtext_model: A randomly initialized kithara.MaxTextModel instance
        scan_layers: Whether the MaxText model is initialize with the scan_layer option

    Returns:
        The loaded MaxText model
    """

    model_name = get_maxtext_model_name_from_hf_handle(preset_handle)

    if model_name not in PARAM_MAPPING:
        raise ValueError(
            f"Model {model_name} is not supported. "
            f"Supported models are {list(PARAM_MAPPING.keys())}"
        )

    config = load_json(preset_handle)
    params_mapping = PARAM_MAPPING[model_name](config, scan_layers)
    hook_fn_mapping = HOOK_FNS[model_name](config, scan_layers, saving_to_hf=False)

    if params_mapping is None:
        raise ValueError(f"Model type {config['model_type']} is not current supported.")

    jax_memory_cleanup(maxtext_model)

    with MaxTextSafetensorLoader(preset_handle) as loader:
        for variable in maxtext_model.weights:
            print(f"-> Loading weight ({variable.path}) into model...")

            if variable.path not in params_mapping:
                raise ValueError(
                    f"Variable path {variable.path} does not exist in the provided weight_mapping"
                )

            try:
                expected_dtype = variable.value.dtype
                hook_fn = (
                    hook_fn_mapping.get(variable.path) if hook_fn_mapping else None
                )
                loader.port_weight(
                    keras_variable=variable,
                    hf_weight_key=params_mapping[variable.path],
                    hook_fn=hook_fn,
                    scan_layers=scan_layers,
                    expected_dtype=expected_dtype,
                )

                assert (
                    variable.value.dtype == expected_dtype
                ), f"Expected weight dtype is {expected_dtype}, but weight dtype is {variable.value.dtype}"
                print(f"✅ Successfully loaded weight ({variable.path}) into model.")

            except Exception as e:
                raise ValueError(
                    f"Failed to load HF weight ({params_mapping[variable.path]}) into MaxText model({variable.path}). "
                    f"Error: {e}"
                )
    return maxtext_model
