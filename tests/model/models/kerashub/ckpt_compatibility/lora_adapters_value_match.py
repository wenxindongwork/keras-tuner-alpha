"""End to end test for KerasHub <> HF LoRA weights conversion correctness.  

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: python tests/model/models/kerashub/ckpt_compatibility/lora_adapters_value_match.py
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

from kithara.utils.gcs_utils import find_cache_root_dir
from kithara import KerasHubModel, PredefinedShardingStrategy
from tests.model.models.utils import check_arrays_match
from tests.model.models.test_prompt import TEST_PROMPT
from tests.model.models.utils import (
    get_hf_logits,
    get_kerashub_model_input,
    check_arrays_match,
    check_predicted_tokens_match,
)
from transformers import AutoModelForCausalLM
import shutil
import torch

LORA_RANK = 16
TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")


def get_kerashub_logits(hf_input_ids, model):
    input = get_kerashub_model_input(hf_input_ids)
    logits, _ = model.stateless_call(
        model.trainable_variables,
        model.non_trainable_variables,
        input,
    )
    return logits


def test(model_id, save_adapter_separately, logits_tol, top1_token_tol):

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Create model with LoRA adapter
    model = KerasHubModel.from_preset(
        f"hf://{model_id}",
        precision="float32",
        lora_rank=LORA_RANK,
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )

    # Save model
    if save_adapter_separately:
        model.save_in_hf_format(TMP_DIR, only_save_adapters=True)
    else:
        model.save_in_hf_format(TMP_DIR)

    # Load checkpoint with HuggingFace
    if save_adapter_separately:
        hf_model = AutoModelForCausalLM.from_pretrained(model_id)
        hf_model.load_adapter(TMP_DIR)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            TMP_DIR, torch_dtype=torch.float32
        )

    # Compare logits
    input_ids, logits_hf = get_hf_logits(
        model_id, TEST_PROMPT, target_length=512, return_input_ids=True, model=hf_model
    )
    logits_kerashub = get_kerashub_logits(input_ids, model)

    check_arrays_match(logits_kerashub[0, :5, :], logits_hf[0, :5, :], logits_tol)
    check_predicted_tokens_match(logits_kerashub, logits_hf, top1_token_tol)

    # Cleanup
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("Passed.")


if __name__ == "__main__":
    test("google/gemma-2-2b", save_adapter_separately=True, logits_tol=1.0, top1_token_tol = 0.05)
    test("google/gemma-2-2b", save_adapter_separately=False, logits_tol=1.0, top1_token_tol = 0.05)

    test("google/gemma-2-9b", save_adapter_separately=True, logits_tol=1.0, top1_token_tol = 0.05)
    test("google/gemma-2-9b", save_adapter_separately=True, logits_tol=1.0, top1_token_tol = 0.05)

