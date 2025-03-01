"""This test validates the bidirectional conversion of model weights
between KerasHub and HuggingFace implementations. 

Steps:
    1. Load the HuggingFace model into KerasHub
    2. Save the KerasHub model in HuggingFace format
    3. Load the converted model back in HuggingFace
    4. Comparing weights and outputs with the original HuggingFace model

Metrics: 
    Max absolute difference between model weights: Expected to be in the range of 0.01
    Max absolute difference between the logits for the first 5 tokens: Expected to be in the range of 2.0
    Disagreement among top1 tokens: Expected to be in the range of 0.1

Usage: 
    Run script on single host VM: python tests/model/kerashub/ckpt_compatibility/saving_gemma2.py

Notes: 
    # TODO Currently this test needs to be triggered manually. It will be refactored
    and run with `pytest` in the future. 
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from kithara import KerasHubModel, PredefinedShardingStrategy
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.test_prompt import TEST_PROMPT
from tests.model.utils import (
    check_arrays_match,
    check_predicted_tokens_match,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

def check_weights_match(model, golden_model, tol):
    def get_all_modules(model):
        modules = []
        for name, _ in model.named_modules():
            if name and hasattr(model.get_submodule(name), "weight"):
                modules.append(name)
        return modules

    modules = get_all_modules(golden_model)

    # Compare weights
    for module in modules:
        gloden_weights = golden_model.get_submodule(module).state_dict()["weight"]
        model_weight = model.get_submodule(module).state_dict()["weight"]
        check_arrays_match(gloden_weights, model_weight, tol)


def get_logits(model_id, model, golden_model):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    target_length = 512
    inputs = tokenizer.encode(TEST_PROMPT, return_tensors="pt")[:, :target_length]

    logits = model(inputs, output_hidden_states=True).logits
    golden_logits = golden_model(inputs, output_hidden_states=True).logits

    return logits, golden_logits

def test(model_id, weight_tol, logits_tol, top1_token_tol):

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Create Model
    model = KerasHubModel.from_preset(
        model_handle=f"hf://{model_id}",
        precision="bfloat16",
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )

    # Save model
    model.save_in_hf_format(TMP_DIR)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(TMP_DIR, torch_dtype=torch.float32)

    # Load reference model
    golden_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )

    # Run forward pass to get logits
    logits, golden_logits = get_logits(model_id, model, golden_model)
    # Run comparion tests
    check_weights_match(model, golden_model, weight_tol)
    # Compare logits from the first 5 tokens of the first sequence
    check_arrays_match(logits[0, :5, :], golden_logits[0, :5, :], logits_tol)
    check_predicted_tokens_match(logits, golden_logits, top1_token_tol)

    # Delete cache
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Passed.")

if __name__ == "__main__":
    test(
        "meta-llama/Llama-3.1-8B", weight_tol=0.0001, logits_tol=0.0001, top1_token_tol=0.001
    )
    test("meta-llama/Llama-3.1-8B", weight_tol=0.0001, logits_tol=1.5, top1_token_tol=0.1)
