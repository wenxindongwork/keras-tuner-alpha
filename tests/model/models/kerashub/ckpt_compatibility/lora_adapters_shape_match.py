"""End to end test for KerasHub <> HF LoRA weights conversion correctness.  

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: python tests/model/models/kerashub/ckpt_compatibility/lora_adapters_shape_match.py
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import shutil
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from kithara.distributed.sharding import PredefinedShardingStrategy
from kithara import KerasHubModel
from kithara.utils.gcs_utils import find_cache_root_dir


# Setup directories
TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
MODEL_ID = "google/gemma-2-2b"
LORA_RANK = 16


def test():
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    test_dir = os.path.join(TMP_DIR, "test")
    golden_dir = os.path.join(TMP_DIR, "golden")

    # Initialize and save Kithara model
    kithara_model = KerasHubModel.from_preset(
        f"hf://{MODEL_ID}",
        lora_rank=LORA_RANK,
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )
    kithara_model.save_in_hf_format(test_dir, only_save_adapters=True)

    # Initialize and save PEFT model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.save_pretrained(golden_dir)

    # Compare model tensors
    test_tensors = {}
    golden_tensors = {}

    # Load and compare golden tensors
    with safe_open(
        os.path.join(golden_dir, "adapter_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as f:
        golden_tensors = {key: f.get_tensor(key) for key in f.keys()}

    # Load and compare test tensors
    with safe_open(
        os.path.join(test_dir, "adapter_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as f:
        for key in f.keys():
            print(f"Processing key: {key}")
            assert key in golden_tensors, f"Missing key in golden tensors: {key}"
            test_tensors[key] = f.get_tensor(key)
            assert (
                golden_tensors[key].shape == test_tensors[key].shape
            ), f"Shape mismatch: {key}. Golden shape is {golden_tensors[key].shape}, actual shape is {test_tensors[key].shape}"

    assert len(golden_tensors) == len(test_tensors), "Number of tensors mismatch"

    # Cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("Passed")


if __name__ == "__main__":
    test()
