import numpy as np

def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config):
    # MaxText abstracted two Gemma layers into 1
    nlayers =  config["num_hidden_layers"]
    return {
        "max_text_layer/params-token_embedder-embedding": "model.embed_tokens.weight",
        "max_text_layer/params-decoder-decoder_norm-scale": "model.norm.weight",
        "max_text_layer/params-decoder-layers-pre_self_attention_norm_global-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_global-wo-kernel": [
            f"model.layers.{i}.mlp.down_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_global-wi_1-kernel": [
            f"model.layers.{i}.mlp.gate_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_global-wi_0-kernel": [
            f"model.layers.{i}.mlp.up_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-post_self_attention_norm_global-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight"
            for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-post_ffw_norm_global-scale": [
            f"model.layers.{i}.post_feedforward_layernorm.weight"
            for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-pre_ffw_norm_global-scale": [
            f"model.layers.{i}.pre_feedforward_layernorm.weight"
            for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_global-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_global-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_global-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_global-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(1, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-pre_self_attention_norm_local-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_local-wo-kernel": [
            f"model.layers.{i}.mlp.down_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_local-wi_1-kernel": [
            f"model.layers.{i}.mlp.gate_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-mlp_local-wi_0-kernel": [
            f"model.layers.{i}.mlp.up_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-post_self_attention_norm_local-scale": [
            f"model.layers.{i}.post_attention_layernorm.weight"
            for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-post_ffw_norm_local-scale": [
            f"model.layers.{i}.post_feedforward_layernorm.weight"
            for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-pre_ffw_norm_local-scale": [
            f"model.layers.{i}.pre_feedforward_layernorm.weight"
            for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_local-key-kernel": [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_local-out-kernel": [
            f"model.layers.{i}.self_attn.o_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_local-query-kernel": [
            f"model.layers.{i}.self_attn.q_proj.weight" for i in range(0, nlayers, 2)
        ],
        "max_text_layer/params-decoder-layers-self_attention_local-value-kernel": [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(0, nlayers, 2)
        ],
    }

def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config):
    def pad_hf_embedding_layer(hf_tensor, target_shape):
        # hf_tensor shape =  [256000,d_model]
        # Target shape = [256128,d_model]
        # MaxText padded embedding to 256128 for better performance.
        padded_hf_tensor = np.zeros(target_shape, dtype=hf_tensor.dtype)
        padded_hf_tensor[:hf_tensor.shape[0], :hf_tensor.shape[1]] = hf_tensor
        return padded_hf_tensor
    return {
        "max_text_layer/params-token_embedder-embedding": pad_hf_embedding_layer,
    }
