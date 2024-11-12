from keras_tuner.sharding._mesh import Axis

GEMMA_FSDP = {
    ".*token_embedding.embeddings.*": (None, Axis.FSDP),
    ".*decoder_block.*attention.*(query|key|value).kernel.*": (
        None,
        Axis.FSDP,
    ),
    ".*decoder_block.*attention_output.kernel.*": (None, None, Axis.FSDP),
    ".*decoder_block.*ffw_gating.kernel.*": (None, Axis.FSDP),
    ".*decoder_block.*ffw_gating_2.kernel.*": (None, Axis.FSDP),
    ".*decoder_block.*ffw_linear.kernel.*": (Axis.FSDP, None),
    # Lora layers
    ".*decoder_block.*attention.*(query|key|value).lora_kernel.*": (
        None,
        Axis.FSDP,
    ),
}

GEMMA_LAYOUT = {
    "fsdp": GEMMA_FSDP,
}
