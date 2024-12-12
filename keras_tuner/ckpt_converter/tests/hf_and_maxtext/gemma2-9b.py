import os
os.environ["KERAS_BACKEND"] = "jax"
from keras_tuner.model import MaxTextModel
import numpy as np
from keras_tuner.preprocessor import PretrainingPreprocessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TMP_DIR = '/dev/shm/temp/hf/checkpoint/9b'

# Create Model
model = MaxTextModel.from_preset(
    preset_handle="hf://google/gemma-2-9b",
    seq_len=7,
    per_device_batch_size=1,
    precision="float32",
    maxtext_config_args= [
        "scan_layers=true",
        "dtype=float32",
        "weight_dtype=float32"
    ]
)

# Save model
model.save_in_hf_format(TMP_DIR)

# Load saved model 
model = AutoModelForCausalLM.from_pretrained(TMP_DIR, torch_dtype=torch.float32)

# Load reference model
golden_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", torch_dtype=torch.float32)

# List out all layers of the Gemma2 model
modules = [
    "model.embed_tokens",
    "model.norm",
    "model.layers.0.input_layernorm" ,
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.gate_proj",
    f"model.layers.0.post_attention_layernorm",
    f"model.layers.0.post_feedforward_layernorm",
    f"model.layers.0.pre_feedforward_layernorm",
    f"model.layers.0.self_attn.k_proj" ,
    f"model.layers.0.self_attn.o_proj" ,
    f"model.layers.0.self_attn.q_proj" ,
    f"model.layers.0.self_attn.v_proj" ,
    f"model.layers.0.input_layernorm" ,
    f"model.layers.0.mlp.down_proj" ,
    f"model.layers.0.mlp.up_proj" ,
    f"model.layers.0.mlp.gate_proj" ,
    f"model.layers.0.post_attention_layernorm",
    f"model.layers.0.post_feedforward_layernorm",
    f"model.layers.0.pre_feedforward_layernorm",
    f"model.layers.0.self_attn.k_proj" ,
    f"model.layers.0.self_attn.o_proj" ,
    f"model.layers.0.self_attn.q_proj" ,
    "model.layers.0.self_attn.v_proj",
]

for module in modules:
    gloden_weights = golden_model.get_submodule(module).state_dict()["weight"]
    model_weight = model.get_submodule(module).state_dict()["weight"]
    if not torch.allclose(gloden_weights, model_weight, atol=0.01):
        is_close_idx = torch.isclose(gloden_weights, model_weight, atol=0.01)
        print(f"Number of mismatch elements in {gloden_weights.shape}", len(gloden_weights[is_close_idx==False]))
        print(gloden_weights[is_close_idx==False])
        print(model_weight[is_close_idx==False])
        raise ValueError(f"Failed to match {module}.")
    else:
        print(f"module {module} matched")



# Compare logits
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
inputs = tokenizer("Hello world", return_tensors="pt")
# inputs {'input_ids': tensor([[   2, 4521, 2134]]), 'attention_mask': tensor([[1, 1, 1]])}

logits = model(**inputs, output_hidden_states=True).logits
golden_logits = golden_model(**inputs, output_hidden_states=True).logits
is_close_idx = torch.isclose(golden_logits, logits, atol=1.0)
print(golden_logits[is_close_idx==False])
print(logits[is_close_idx==False])

assert torch.allclose(logits, golden_logits, atol=1.0)

