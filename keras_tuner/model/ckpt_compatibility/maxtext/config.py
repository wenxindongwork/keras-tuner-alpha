import numpy as np

def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
    nlayers =  config["num_hidden_layers"]
    mapping = {
        "max_text_layer/params-token_embedder-embedding": "model.embed_tokens.weight",
        "max_text_layer/params-decoder-decoder_norm-scale": "model.norm.weight",
    }
    if scan_layers:
        mapping = {
            **mapping, 
            "max_text_layer/params-decoder-layers-pre_self_attention_norm_global-scale": [
            f"model.layers.{i}.input_layernorm.weight" for i in range(1, nlayers, 2)
            ],
            "max_text_layer/params-decoder-layers-mlp_global-wo-kernel": [
                f"model.layers.{i}.mlp.down_proj.weight" for i in range(1, nlayers, 2)
            ],
            "max_text_layer/params-decoder-layers-mlp_global-wi_1-kernel": [
            f"model.layers.{i}.mlp.up_proj.weight" for i in range(1, nlayers, 2)
            ],
            "max_text_layer/params-decoder-layers-mlp_global-wi_0-kernel": [
                f"model.layers.{i}.mlp.gate_proj.weight" for i in range(1, nlayers, 2)
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
                f"model.layers.{i}.mlp.up_proj.weight" for i in range(0, nlayers, 2)
            ],
            "max_text_layer/params-decoder-layers-mlp_local-wi_0-kernel": [
                f"model.layers.{i}.mlp.gate_proj.weight" for i in range(0, nlayers, 2)
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

    else:
        for maxtext_layer_idx in range(0, nlayers//2):
            local_layer_idx = maxtext_layer_idx * 2 
            global_layer_idx = maxtext_layer_idx * 2 + 1
            layer_mapping = {
                # MaxText abstracted two Gemma decoder layers into 1 due to local and global attention
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": 
                    f"model.layers.{global_layer_idx}.input_layernorm.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": 
                    f"model.layers.{global_layer_idx}.mlp.down_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel": 
                f"model.layers.{global_layer_idx}.mlp.up_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": 
                    f"model.layers.{global_layer_idx}.mlp.gate_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": 
                    f"model.layers.{global_layer_idx}.post_attention_layernorm.weight"
                
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": 
                    f"model.layers.{global_layer_idx}.post_feedforward_layernorm.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": 
                    f"model.layers.{global_layer_idx}.pre_feedforward_layernorm.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": 
                    f"model.layers.{global_layer_idx}.self_attn.k_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": 
                    f"model.layers.{global_layer_idx}.self_attn.o_proj.weight"
                    
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": 
                    f"model.layers.{global_layer_idx}.self_attn.q_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": 
                    f"model.layers.{global_layer_idx}.self_attn.v_proj.weight"
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": 
                    f"model.layers.{local_layer_idx}.input_layernorm.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": 
                    f"model.layers.{local_layer_idx}.mlp.down_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": 
                    f"model.layers.{local_layer_idx}.mlp.up_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": 
                    f"model.layers.{local_layer_idx}.mlp.gate_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": 
                    f"model.layers.{local_layer_idx}.post_attention_layernorm.weight"
                    
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": 
                    f"model.layers.{local_layer_idx}.post_feedforward_layernorm.weight"
                    
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": 
                    f"model.layers.{local_layer_idx}.pre_feedforward_layernorm.weight"
                    
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": 
                    f"model.layers.{local_layer_idx}.self_attn.k_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": 
                    f"model.layers.{local_layer_idx}.self_attn.o_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": 
                    f"model.layers.{local_layer_idx}.self_attn.q_proj.weight" 
                ,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": 
                    f"model.layers.{local_layer_idx}.self_attn.v_proj.weight" 
                ,
            }
            mapping = {
                **mapping, 
                **layer_mapping
            }
    return mapping
    

def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False):
    nlayers =  config["num_hidden_layers"]
    def pad_hf_embedding_layer(input_tensor, target_shape, saving_to_hf):
        """
        hf_tensor shape =  [256000,d_model]
        Target shape = [256128,d_model]
        MaxText padded embedding to 256128 for better performance.
        """
        normalizer = np.dtype(input_tensor.dtype).type(config["hidden_size"]**0.5)
        if saving_to_hf:
            target_tensor = input_tensor[:target_shape[0], :target_shape[1]]
            target_tensor = target_tensor / normalizer
            return target_tensor
        
        target_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
        target_tensor[:input_tensor.shape[0], :input_tensor.shape[1]] = input_tensor
        target_tensor = target_tensor * normalizer
        return target_tensor

    def reshape_kernel(input_tensor, target_shape, saving_to_hf):
        if saving_to_hf:
            target_shape = np.flip(np.array(target_shape))
            return input_tensor.reshape(target_shape).transpose()

        return input_tensor.transpose().reshape(target_shape)
    
    def scale_rmsnorm_layer(input_tensor, target_shape, saving_to_hf):
        if saving_to_hf:
            return (input_tensor - 1.0).reshape(target_shape)
        return (input_tensor + 1.0).reshape(target_shape)
    
    def scale_query_layer(input_tensor, target_shape, saving_to_hf):
        if saving_to_hf:
            depth_scale = np.dtype(input_tensor.dtype).type(np.sqrt(config["head_dim"]) )
            return input_tensor * depth_scale
        
        depth_scale = np.dtype(input_tensor.dtype).type(1/ np.sqrt(config["head_dim"]) )
        return input_tensor * depth_scale

    mapping = {
        "max_text_layer/params-token_embedder-embedding": pad_hf_embedding_layer,
        "max_text_layer/params-decoder-decoder_norm-scale": scale_rmsnorm_layer
        }
    if scan_layers:
        mapping = {
            **mapping, 
                f"max_text_layer/params-decoder-layers-self_attention_global-query-kernel": [reshape_kernel, scale_query_layer],
                f"max_text_layer/params-decoder-layers-self_attention_local-query-kernel": [reshape_kernel, scale_query_layer],
                f"max_text_layer/params-decoder-layers-self_attention_global-key-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-self_attention_local-key-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-self_attention_global-value-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-self_attention_local-value-kernel": reshape_kernel,


                f"max_text_layer/params-decoder-layers-mlp_global-wo-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-mlp_global-wi_1-kernel":reshape_kernel,
                f"max_text_layer/params-decoder-layers-mlp_global-wi_0-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-self_attention_global-out-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-mlp_local-wo-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-mlp_local-wi_1-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-mlp_local-wi_0-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers-self_attention_local-out-kernel": reshape_kernel,

                f"max_text_layer/params-decoder-layers-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-post_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-post_ffw_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers-pre_ffw_norm_local-scale": scale_rmsnorm_layer,
        }
    else:
        for maxtext_layer_idx in range(nlayers//2):
            mapping = {
                **mapping, 
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-query-kernel": [reshape_kernel, scale_query_layer],
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-query-kernel": [reshape_kernel, scale_query_layer],
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-key-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-value-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-value-kernel": reshape_kernel,

                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wo-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_1-kernel":reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_global-wi_0-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_global-out-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wo-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_1-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-mlp_local-wi_0-kernel": reshape_kernel,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-self_attention_local-out-kernel": reshape_kernel,

                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_global-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_self_attention_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-post_ffw_norm_local-scale": scale_rmsnorm_layer,
                f"max_text_layer/params-decoder-layers_{maxtext_layer_idx}-pre_ffw_norm_local-scale": scale_rmsnorm_layer,

            }
    return mapping

