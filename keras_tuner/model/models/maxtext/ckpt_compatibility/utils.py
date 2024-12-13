from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader
from keras_nlp.src.utils.preset_utils import load_json

def print_maxtext_model_variables(model):
    for variable in model.weights:
        print(variable.path, variable.shape)


def print_hg_safetensor_weight_keys(preset_handle):
    with SafetensorLoader(preset_handle) as loader:
        weight_map_keys = loader.safetensor_config["weight_map"].keys()
        print(weight_map_keys)


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
