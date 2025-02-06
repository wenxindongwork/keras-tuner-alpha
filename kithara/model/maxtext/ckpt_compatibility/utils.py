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

from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader
from transformers import AutoModelForCausalLM

def get_hf_safetensor_weight_keys(preset_handle):
    with SafetensorLoader(preset_handle) as loader:
        weight_map_keys = loader.safetensor_config["weight_map"].keys()
        return list(weight_map_keys)

def get_hf_model_weight_shapes(preset_handle):
    from transformers import AutoModelForCausalLM
    weight_keys = get_hf_safetensor_weight_keys(preset_handle)
    model = AutoModelForCausalLM.from_pretrained(preset_handle)

    weight_to_shape = {}
    for path in weight_keys: 
        path = path.removesuffix(".weight")
        hf_module = model.get_submodule(path) 
        target_shape = hf_module.state_dict()["weight"].shape
        weight_to_shape[path] = target_shape
    return weight_to_shape
