"""End to end test for model checkpointing correctness 

Run script on single host VM: python tests/callbacks/checkpointer_maxtext.py 
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
from kithara import MaxTextModel
import numpy as np
from kithara.callbacks.checkpointer import Checkpointer
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.utils import check_arrays_match
import shutil
import jax

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")


def test():

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    model = MaxTextModel.from_random(
        model_name="default", seq_len=100, per_device_batch_size=1, scan_layers=True
    )

    model_input = {
        "tokens": np.array([[i for i in range(100)]] * jax.device_count()),
        "positions": np.array([[i for i in range(100)]] * jax.device_count()),
        "segment_ids": np.array([[1 for _ in range(100)]] * jax.device_count()),
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
