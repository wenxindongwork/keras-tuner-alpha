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
Module to convert MaxText model weights to HuggingFace format.
"""

import torch
import contextlib
import time
from kithara.model.hf_compatibility import (
    MODEL_CONFIGS,
)
from kithara.model.hf_compatibility import process_weight, save_model_files
from kithara.model.maxtext.ckpt_compatibility.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from kithara.model.hf_compatibility import SHAPE_MAPPING
import jax 

def _get_model_mappings(model_name: str, scan_layers: bool, config: dict):
    """Retrieves parameter, shape, and hook function mappings for the model."""
    return {
        "param_mapping": PARAM_MAPPING[model_name](config.to_dict(), scan_layers),
        "shape_mapping": SHAPE_MAPPING[model_name](config.to_dict()),
        "hook_fn_mapping": HOOK_FNS[model_name](
            config.to_dict(), scan_layers, saving_to_hf=True
        ),
    }


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
        weight_dict = process_weight(variable, mappings)
        weight_arrays.update(weight_dict)
    print(
        f"\n✅ Weights converted into HuggingFace format in {time.time() - start_time}s"
    )

    # Save all model files
    save_model_files(weight_arrays, config, output_dir, parallel_threads)


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

    jax.experimental.multihost_utils.sync_global_devices("saving_completed")