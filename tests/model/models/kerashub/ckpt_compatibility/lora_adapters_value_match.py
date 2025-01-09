"""End to end test for KerasHub <> HF LoRA weights conversion correctness.  

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: 
HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python tests/model/models/kerashub/ckpt_compatibility/lora_adapters_value_match.py
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from kithara.utils.gcs_utils import find_cache_root_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from kithara.model.kerashub.keras_hub_model import KerasHubModel
from kithara.distributed.sharding import PredefinedShardingStrategy
from tests.model.models.utils import check_logits_match
from kithara import PretrainingPreprocessor
import shutil
import torch

MODEL_HANDLE = "google/gemma-2-2b"
LORA_RANK = 4
TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
TEST_INPUT = "Hello world" 

def run_test(save_adapter_separately=True):
    
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Create model with lora adapter
    model = KerasHubModel.from_preset(
        f"hf://{MODEL_HANDLE}",
        precision="float32",
        lora_rank=LORA_RANK,
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )

    # Compute golden logits
    preprocessor = PretrainingPreprocessor(
        tokenizer_handle=f"hf://{MODEL_HANDLE}",
        seq_len=10,
    )
    inputs = preprocessor.prepare_training_input(TEST_INPUT)["x"]
    golden_logits, _ = model.stateless_call(
        model.trainable_variables, model.non_trainable_variables, inputs
    )
    golden_logits = golden_logits[:, :2, :]  # logits for the first 2 tokens
    
    if save_adapter_separately:
        model.save_in_hf_format(TMP_DIR, only_save_adapters=True)
    else: 
        model.save_in_hf_format(TMP_DIR)

    # Compare logits
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HANDLE)
    inputs = tokenizer(TEST_INPUT, return_tensors="pt")
    
    if save_adapter_separately:
        peft_model = AutoModelForCausalLM.from_pretrained(MODEL_HANDLE)
        peft_model.load_adapter(TMP_DIR)
        logits_from_peft_model = peft_model(**inputs).logits[:, :2, :]
        check_logits_match(golden_logits, logits_from_peft_model, 1.0)
    else: 
        merged_model = AutoModelForCausalLM.from_pretrained(TMP_DIR, torch_dtype=torch.float32)
        logits_from_merged_model = merged_model(**inputs).logits
        logits_from_merged_model = logits_from_merged_model[:, :2, :]
        check_logits_match(golden_logits, logits_from_merged_model, 1.0)


    # Cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    run_test(save_adapter_separately=True)
    run_test(save_adapter_separately=False)