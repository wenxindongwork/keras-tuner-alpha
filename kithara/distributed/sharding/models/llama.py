from kithara.distributed.sharding._mesh import Axis

LLAMA_FSDP = {
    "*token_embedding.embeddings.*": (None, Axis.FSDP),
    "*transformer_layer.*attention.*(query|key|value).kernel*": (
        None,
        Axis.FSDP,
    ),
    "*transformer_layer.*attention_output.kernel*": (None, None, Axis.FSDP),
    "*transformer_layer.*feedforward_gate_dense.kernel*": (None, Axis.FSDP),
    "*transformer_layer.*feedforward_intermediate_dense.kernel*": (None, Axis.FSDP),
    ".*decoder_block.*ffw_linear.kernel.*": (Axis.FSDP, None),
}

LLAMA_LAYOUT = {
    "fsdp": LLAMA_FSDP,
}
