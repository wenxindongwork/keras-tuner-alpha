"""End to end test for MaxText <> HF Llama3.1-8b checkpoint conversion correctness. 

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python tests/model/models/maxtext/ckpt_compatibility/llama31-8b.py
"""
import os
os.environ["KERAS_BACKEND"] = "jax"
from kithara import MaxTextModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from kithara.utils.gcs_utils import find_cache_root_dir
import os 
import shutil

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
shutil.rmtree(TMP_DIR, ignore_errors=True)

# Create Model
model = MaxTextModel.from_preset(
    preset_handle="hf://meta-llama/Llama-3.1-8B",
    seq_len=4096,
    per_device_batch_size=1,
    scan_layers=False, 
    precision="mixed_float16"
)


# Save model
model.save_in_hf_format(TMP_DIR)

# Load model 
model = AutoModelForCausalLM.from_pretrained(TMP_DIR, torch_dtype=torch.float32)

# Load reference model
golden_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.float32)

def get_all_modules(model, prefix=''):
    modules = []
    for name, _ in model.named_modules():
        if name and hasattr(model.get_submodule(name), 'weight'):
            modules.append(name)
    return modules

modules = get_all_modules(golden_model)

# Compare weights 
for module in modules:
    gloden_weights = golden_model.get_submodule(module).state_dict()["weight"]
    model_weight = model.get_submodule(module).state_dict()["weight"]
    if not torch.allclose(gloden_weights, model_weight, atol=0.01):
        is_close_idx = torch.isclose(gloden_weights, model_weight, atol=0.01)
        print(f"Number of mismatch weight elements in {gloden_weights.shape}", len(gloden_weights[is_close_idx==False]))
        print(gloden_weights[is_close_idx==False])
        print(model_weight[is_close_idx==False])
        raise ValueError(f"Failed to match {module}.")
    else:
        print(f"Module {module} matched")

# Compare logits
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
inputs = tokenizer("Hello world", return_tensors="pt")

logits = model(**inputs, output_hidden_states=True).logits
golden_logits = golden_model(**inputs, output_hidden_states=True).logits
if not torch.allclose(logits, golden_logits, atol=1.0):
    is_close_idx = torch.isclose(golden_logits, logits, atol=1.0)
    print(f"Number of mismatch logits elements in {golden_logits.shape}", len(golden_logits[is_close_idx==False]))
    print(golden_logits[is_close_idx==False])
    print(logits[is_close_idx==False])
    raise ValueError(f"Failed to match logits.")
else:
    print("Logits matched")

# Delete cache
shutil.rmtree(TMP_DIR, ignore_errors=True)
