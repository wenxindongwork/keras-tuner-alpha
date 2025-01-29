"""End to end test for model checkpointing correctness 

Run script on single host VM: python tests/callbacks/checkpointer_kerashub.py 
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from kithara import KerasHubModel, PredefinedShardingStrategy
import numpy as np
from kithara.callbacks.checkpointer import Checkpointer
from kithara.utils.gcs_utils import find_cache_root_dir
import shutil
from tests.model.utils import check_arrays_match

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

def test():
    
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b", 
        lora_rank=6,
        sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma")
    )

    model_input = {
        "token_ids": np.array([[1,2,3]]),
        "padding_mask": np.array([[1,1,1]]),
    }
    
    logits_before, _ = model.stateless_call(
        model.trainable_variables,
        model.non_trainable_variables,
        model_input,
    )
    # Save and load model
    checkpointer = Checkpointer(TMP_DIR, model=model)
    checkpointer.save(0, blocking=True)
    checkpointer.load()

    logits_after, _ = model.stateless_call(
        model.trainable_variables,
        model.non_trainable_variables,
        model_input,
    )

    check_arrays_match(logits_before, logits_after, atol=0.0001)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    
    print("Passed.")
if __name__ == "__main__":
    test()