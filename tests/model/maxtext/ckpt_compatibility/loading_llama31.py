"""Compare model outputs between HuggingFace and MaxText model. 

This test validates that the MaxText implementation of Llama-3.1 models produces logits that are
numerically comparable to the reference HuggingFace implementation. 

Steps: 
    1. Load HF model into MaxText model.
    2. Get logits from HF model 
    3. Get logits from MaxText model.
    4. Compare logits 

Usage:
    Run script on single host VM: python tests/model/maxtext/ckpt_compatibility/loading_llama31.py
"""

from kithara import MaxTextModel
from tests.model.test_prompt import TEST_PROMPT
from tests.model.utils import check_arrays_match, check_predicted_tokens_match
import tests.model.utils as utils

# Test configuration
MAX_TARGET_LENGTH = 512


def get_maxtext_logits(model_id, input_ids):

    model = MaxTextModel.from_preset(
        preset_handle=f"hf://{model_id}",
        seq_len=MAX_TARGET_LENGTH,
        per_device_batch_size=1,
        scan_layers=True,
        precision="float32",
    )

    # Run forward pass
    input = utils.get_maxtext_model_input(input_ids)
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
    # Extract logits from the first input sequence
    logits_hf = logits_hf[0, :, :]
    # Sometimes MaxText pad vocab with 0s (e.g. 256000 to 256128),
    # hence here we "unpad" the vocab
    vocab_size = logits_hf.shape[1]
    logits_maxtext = get_maxtext_logits(model_id, input_ids)
    logits_maxtext = logits_maxtext[0, :, :vocab_size]

    # Check logits match closely for the first 5 tokens
    check_arrays_match(logits_hf[:5, :], logits_maxtext[:5, :], atol=logits_tol)
    check_predicted_tokens_match(logits_hf, logits_maxtext, tolerance=top1_token_tol)

    print("Passed.")


if __name__ == "__main__":
    test("meta-llama/Llama-3.1-8B", logits_tol=1.0, top1_token_tol=0.02)
