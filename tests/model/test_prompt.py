TEST_PROMPT="""# Activation dtypes.
dtype: "bfloat16"
# Used to configure quantization in the transformer layers, defaults to null implying bf16.
# Possible alternative settings are as follows:
# 'int8' for dynamic range quantization using 8-bits
# 'int8w' for weights only quantization using 8-bits
# 'int4w' for weights only quantization using 4-bits
# 'intmp' for mixed precision weight only quantization based on config file
# 'fp8' for 8-bit floating-point GeMMs on NVIDIA GPUs.
quantization: ""
# Choose one of default, high, and highest.
# https://kolonist26-jax-kr.readthedocs.io/en/latest/jax.lax.html#jax.lax.Precision
matmul_precision: "default"
activations_in_float32: False # Sets activations to float32 before nonlinearity it true, else dtype
# Path to file with quantization config - only used for mixed precision.
# Example configs in ../Maxtext/configs/quantization
# Allows us to configure different bits, tiling and scale for quantizing selected weights.
# Bits represents number of bits to quantize to,
# tile-size represents the tiling sized used in AQT tiled_dot_general,
# Value of scale is used to scale the abs_max value used for AQT quantization
# Defaults values are 8 bits, tile-size=-1 (no tiling) and scale=1.
quant_cfg_path: ""
quantize_kvcache: False # Set to True to quantize KV Cache values, defaults to False
# Valid kv_quant_axis values:
#   - "" is valid only when quantize_kvcache is False
#   - "dkv" indicates quantize kv cache over the cache_kv, i.e. kv dimension axis
#   - "heads_and_dkv" indicates quantize kv cache over cache_heads and cache_kv axes
# Default to "heads_and_dkv" for faster compution, kv_quant_axis is not used when quantize_kvcache is False
#   - "dkv" is expected with better accuracy but degraded computation
kv_quant_axis: "heads_and_dkv"
kv_quant_dtype: "int8"
checkpoint_is_quantized: False # Set to True if reading from a saved aqt quantized checkpoint
# Saves params quantized on fly at following path
save_quantized_params_path: ""
# Shard the range finding operation for quantization. By default this is set to number of slices.
quantization_local_shard_count: -1
decoder_block: "llama2" # which style of DecoderBlock to use.
# Global parameter scale needs to be a power of 2. If you want finer grained control of the model sizes
# then you should explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads,
# base_mlp_dim, base_num_decoder_layers and/or head_dim.
weight_dtype: float32
global_parameter_scale: 1
base_emb_dim: 2048
base_num_query_heads: 16
base_num_kv_heads: 16
base_mlp_dim: 7168
base_num_decoder_layers: 16
head_dim: 128
mlp_activations: ["silu", "linear"]
dropout_rate: 0
logits_via_embedding: False
normalize_embedding_logits: True  # whether to normlize pre-softmax logits if logits_via_embedding is true
logits_dot_in_fp32: True  # whether to use fp32 in logits_dense or shared_embedding dot product for stability"""