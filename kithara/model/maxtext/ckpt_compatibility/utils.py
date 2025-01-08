from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader
from keras_nlp.src.utils.preset_utils import load_json
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

def get_maxtext_model_name_from_hf_handle(preset_handle):
    config = load_json(preset_handle)
    model_type = config["model_type"]
    if model_type == "gemma2":
        n_layers = config["num_hidden_layers"]
        if n_layers == 26:
            return "gemma2-2b"
        elif n_layers == 42:
            return "gemma2-9b"
        elif n_layers == 46:
            return "gemma2-27b"
    print(f"model type {model_type} is currently unsupported.")
    return None
