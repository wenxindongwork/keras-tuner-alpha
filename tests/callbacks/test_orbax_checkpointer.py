"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""End to end test for model checkpointing correctness 

Run script on single host VM: python -m unittest tests/callbacks/test_orbax_checkpointer.py 
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import shutil
import unittest
import numpy as np
from kithara import KerasHubModel, MaxTextModel, PredefinedShardingStrategy
from kithara.callbacks.checkpointer import Checkpointer
from kithara.utils.gcs_utils import find_cache_root_dir
from tests.model.utils import check_arrays_match
import jax


class TestOrbaxCheckpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

    def setUp(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def _test_checkpoint_save_load(self, model, model_input):
        """Test that model state is preserved after save and load operations."""
        # Get initial model output
        logits_before, _ = model.stateless_call(
            model.trainable_variables,
            model.non_trainable_variables,
            model_input,
        )

        # Save and load model
        checkpointer = Checkpointer(self.TMP_DIR, model=model)
        checkpointer.save(0, blocking=True)
        checkpointer.load()

        # Get model output after loading
        logits_after, _ = model.stateless_call(
            model.trainable_variables,
            model.non_trainable_variables,
            model_input,
        )

        # Verify outputs match within tolerance
        try:
            check_arrays_match(logits_before, logits_after, atol=0.0001)
        except AssertionError as e:
            self.fail(f"Model outputs don't match after save/load: {str(e)}")

    def test_kerashub_models(self):
        # Initialize model
        model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            lora_rank=6,
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )

        # Prepare test input
        model_input = {
            "token_ids": np.array([[1, 2, 3]]),
            "padding_mask": np.array([[1, 1, 1]]),
        }
        self._test_checkpoint_save_load(model, model_input)

    def test_maxtext_model(self):
        model = MaxTextModel.from_random(
            model_name="default", seq_len=100, per_device_batch_size=1, scan_layers=True
        )

        model_input = {
            "tokens": np.array([[i for i in range(100)]] * jax.device_count()),
            "positions": np.array([[i for i in range(100)]] * jax.device_count()),
            "segment_ids": np.array([[1 for _ in range(100)]] * jax.device_count()),
        }
        self._test_checkpoint_save_load(model, model_input)


if __name__ == "__main__":
    unittest.main(verbosity=2)
