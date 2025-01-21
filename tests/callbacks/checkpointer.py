"""End to end test for model checkpointing correctness 

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: python tests/callbacks/checkpointer.py 
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from kithara import MaxTextModel
import numpy as np
from kithara.callbacks.checkpointer import Checkpointer
from kithara.utils.gcs_utils import find_cache_root_dir
import shutil

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

def test():
    
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    model = MaxTextModel.from_random(
        model_name="default", seq_len=100, per_device_batch_size=1, scan_layers=True
    )

    pred_before = model.generate(
        "What is your name?",
        tokenizer_handle="hf://google/gemma-2-2b",
        max_length=30,
        strip_prompt=True,
        return_decoded=True,
    )

    # Save and load model
    checkpointer = Checkpointer(TMP_DIR, model=model)
    checkpointer.save(0, blocking=True)
    checkpointer.load()

    pred_after = model.generate(
        "What is your name?",
        tokenizer_handle="hf://google/gemma-2-2b",
        max_length=30,
        strip_prompt=True,
        return_decoded=True,
    )

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    if not np.array_equal(pred_before["token_ids"], pred_after["token_ids"]):
        raise ValueError(
            f"Prediction between model and checkpoint did not match: {pred_before} vs {pred_after}"
        )
    else:
        print("Test passed.")

if __name__ == "__main__":
    test()