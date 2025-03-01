import transformers
from keras_nlp.src.utils.preset_utils import load_json
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

llama31_8b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    bos_token_id=128000,
    eos_token_id=128001,

    # Additional attributes from your JSON:
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    pretraining_tp=1,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=False,
    use_cache=True,
)

llama31_70b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=80,
    num_attention_heads=64,
    head_dim=128,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    rms_norm_eps=1e-05,
    bos_token_id=128000,
    eos_token_id=128001,
)

llama31_405b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=16384,
    intermediate_size=53248,
    num_hidden_layers=126,
    num_attention_heads=128,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=131072,
    rms_norm_eps=1e-05,
    bos_token_id=128000,
    eos_token_id=128001,
)

llama32_1b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    bos_token_id=128000,
    eos_token_id=128001,

    # Additional attributes from your JSON:
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    pretraining_tp=1,
    rope_scaling={
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=True,
    use_cache=True,
)

llama32_3b_config = transformers.LlamaConfig(
    vocab_size=128256,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=28,
    num_attention_heads=24,
    num_key_value_heads=8,
    max_position_embeddings=131072,
    rms_norm_eps=1e-5,
    bos_token_id=128000,
    eos_token_id=128001,

    # Additional attributes from your JSON:
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    initializer_range=0.02,
    mlp_bias=False,
    model_type="llama",
    pretraining_tp=1,
    rope_scaling={
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=True,
    use_cache=True,
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
    elif model_type == "llama":
        n_layers = config["num_hidden_layers"]
        if n_layers == 32:
            return supported_models.LLAMA31_8B
        elif n_layers == 80:
            return supported_models.LLAMA31_70B
        elif n_layers == 126:
            return supported_models.LLAMA31_405B
    print(f"model type {model_type} is currently unsupported.")
    return None

MODEL_CONFIGS = {
    supported_models.GEMMA2_2B: gemma2_2b_config,
    supported_models.GEMMA2_9B: gemma2_9b_config,
    supported_models.GEMMA2_27B: gemma2_27b_config,
    supported_models.LLAMA31_8B: llama31_8b_config,
    supported_models.LLAMA31_70B: llama31_70b_config,
    supported_models.LLAMA31_405B: llama31_405b_config,
}
