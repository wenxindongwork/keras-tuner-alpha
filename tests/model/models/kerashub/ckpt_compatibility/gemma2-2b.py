"""End to end test for KerasHub <> HF Gemma2-2b checkpoint conversion correctness. (Without LoRA)

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: 
HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python tests/model/models/kerashub/ckpt_compatibility/gemma2-2b.py
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from kithara.model.kerashub.keras_hub_model import KerasHubModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from kithara.utils.gcs_utils import find_cache_root_dir
import shutil
from tests.model.models.utils import check_logits_match, get_all_modules_from_hf_model
from kithara.distributed.sharding import PredefinedShardingStrategy

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
shutil.rmtree(TMP_DIR, ignore_errors=True)

# Create Model
model = KerasHubModel.from_preset(
    model_handle="hf://google/gemma-2-2b",
    precision="mixed_float16",
    sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
)

# Save model
model.save_in_hf_format(TMP_DIR)

# Load model. Loaded model should match reference model.
model = AutoModelForCausalLM.from_pretrained(TMP_DIR, torch_dtype=torch.float32)

# Load reference model
golden_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", torch_dtype=torch.float32
)

modules = get_all_modules_from_hf_model(golden_model)

# Compare weights
for module in modules:
    gloden_weights = golden_model.get_submodule(module).state_dict()["weight"]
    model_weight = model.get_submodule(module).state_dict()["weight"]
    check_logits_match(gloden_weights, model_weight, 0.01)

# Compare logits
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
inputs = tokenizer("Hello world", return_tensors="pt")

logits = model(**inputs, output_hidden_states=True).logits
golden_logits = golden_model(**inputs, output_hidden_states=True).logits
check_logits_match(golden_logits, logits, 1.0)

# Delete cache
print("Test passed.")
shutil.rmtree(TMP_DIR, ignore_errors=True)
