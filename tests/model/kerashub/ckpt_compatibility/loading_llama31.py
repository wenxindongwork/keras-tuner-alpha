"""Compare model outputs between HuggingFace and KerasHub model. 

This test validates that the KerasHub implementation of Gemma-2 models produces logits that are
numerically comparable to the reference HuggingFace implementation. 

Steps: 
    1. Load HF model into KerasHub model.
    2. Get logits from HF model 
    3. Get logits from KerasHub model.
    4. Compare logits 
    
Usage:
    Run script on single host VM: python tests/model/kerashub/ckpt_compatibility/loading_gemma2.py
"""

from kithara import KerasHubModel, PredefinedShardingStrategy
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.test_prompt import TEST_PROMPT
import tests.model.utils as utils
from tests.model.utils import check_arrays_match, check_predicted_tokens_match
import os
os.environ["KERAS_BACKEND"] = "jax"
# Test configuration
MAX_TARGET_LENGTH = 512
TMP_DIR = os.path.join(find_cache_root_dir(), "test/")

def get_kerashub_logits(model_id, input_ids):

    model = KerasHubModel.from_preset(
        model_handle=f"hf://{model_id}",
        precision="float32",
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )

    # Run forward pass 
    input = utils.get_kerashub_model_input(input_ids)
    logits, _ = model.stateless_call(
        model.trainable_variables,
        model.non_trainable_variables,
        input,
    )
    return logits


def test(model_id, logits_tol, top1_token_tol):

    input_ids, logits_hf = utils.get_hf_logits(
        model_id, TEST_PROMPT, MAX_TARGET_LENGTH, return_input_ids=True
    )
    logits_kerashub = get_kerashub_logits(model_id, input_ids)
    
    # Extract logits for the first input sequence for comparison
    logits_hf = logits_hf[0, :, :]
    logits_kerashub = logits_kerashub[0, :, :]

    # Check logits match closely for the first 5 tokens
    check_arrays_match(logits_hf[:5, :], logits_kerashub[:5, :], atol = logits_tol)
    # Check predicted tokens match 95% of the time
    check_predicted_tokens_match(logits_hf, logits_kerashub, tolerance=top1_token_tol)
    print("Passed")

if __name__ == "__main__":
    test("meta-llama/Llama-3.1-8B", logits_tol=0.5, top1_token_tol=0.2)
    test("meta-llama/Llama-3.1-8B", logits_tol=1.0, top1_token_tol=0.2)
