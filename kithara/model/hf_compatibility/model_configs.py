import transformers
from keras_hub.src.utils.preset_utils import load_json
from kithara.model import supported_models


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


def get_model_name_from_preset_handle(preset_handle):
    # TODO(wenxindongwork): Support parsing presets other than HF
    config = load_json(preset_handle)
    model_type = config["model_type"]
    if model_type == "gemma2":
        n_layers = config["num_hidden_layers"]
        if n_layers == 26:
            return supported_models.GEMMA2_2B
        elif n_layers == 42:
            return supported_models.GEMMA2_9B
        elif n_layers == 46:
            return supported_models.GEMMA2_27B
    print(f"model type {model_type} is currently unsupported.")
    return None

MODEL_CONFIGS = {
    supported_models.GEMMA2_2B: gemma2_2b_config,
    supported_models.GEMMA2_9B: gemma2_9b_config,
    supported_models.GEMMA2_27B: gemma2_27b_config,
}
