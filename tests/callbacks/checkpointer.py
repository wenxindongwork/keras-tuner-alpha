"""End to end test for model checkpointing correctness 

# TODO Currently this test needs to be triggered manually. It will be refactored
and run with `pytest` in the future. 

Run script on single host VM: HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python tests/callbacks/checkpointer.py 
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
import jax 
from typing import Optional
from keras_tuner import Dataloader, PretrainingPreprocessor, Trainer
from keras_tuner.model.models.maxtext.maxtext_model import MaxTextModel
from examples.example_datasets import example_datasets
from keras_tuner.model.models.kerashub.keras_hub_model import KerasHubModel
from keras_tuner.model.sharding import PredefinedShardingStrategy

import numpy as np
import orbax.checkpoint as ocp
import jax

from keras_tuner.callbacks.checkpointer import Checkpointer
from keras_tuner.utils.gcs_utils import find_cache_root_dir

TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

model = MaxTextModel.from_random(
    model_name="default",
    seq_len=100,
    per_device_batch_size=1,
    scan_layers=True
)

preprocessor = PretrainingPreprocessor(
    tokenizer_handle="hf://google/gemma-2-2b",
    seq_len=100,
    model_type="maxtext"
)

model_input = preprocessor.prepare_inference_input(["What is your name?"])
pred_before = model.generate(model_input, stop_token_ids=None, max_length= 10, strip_prompt=True)

# Save and load model
checkpointer = Checkpointer(TMP_DIR, model=model)
checkpointer.save(0, blocking=True)
checkpointer.load()

model_input = preprocessor.prepare_inference_input(["What is your name?"])
pred_after = model.generate(model_input, stop_token_ids=None, max_length= 10, strip_prompt=True)

if not np.array_equal(pred_before["token_ids"] , pred_after["token_ids"]):
    raise ValueError(f"Prediction between model and checkpoint did not match: {pred_before} vs {pred_after}")
else:
    print("Test passed.")