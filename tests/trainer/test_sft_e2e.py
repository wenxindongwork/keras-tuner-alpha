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

"""Unit tests for running SFT on singlehost

Run test on a TPU VM: python -m unittest tests/trainer/test_sft_e2e.py 
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import unittest
from kithara import (
    MaxTextModel,
    KerasHubModel,
    TextCompletionDataset,
    Dataloader,
    Checkpointer,
    Trainer,
)
import unittest.result

import ray
from kithara.utils.gcs_utils import find_cache_root_dir
import shutil
import keras
import jax


class TestRunningSFT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")
        cls.MODEL_HANDLE = "hf://google/gemma-2-2b"
        cls.TOKENIZER_HANDLE = "hf://google/gemma-2-2b"
        cls.SEQ_LEN = 1024
        cls.PRECISION = "mixed_bfloat16"
        cls.TRAINING_STEPS = 10
        cls.EVAL_STEPS_INTERVAL = 5
        cls.LOG_STEPS_INTERVAL = 5
        cls.PER_DEVICE_BATCH_SIZE = 1
        cls.MAX_EVAL_SAMPLES = 10
        cls.LEARNING_RATE = 5e-5

    def setUp(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)

    def _create_datasets(self, packing=False):
        dataset_items = [
            {"text": f"{i} What is your name? My name is Mary."} for i in range(2000)
        ]
        dataset = ray.data.from_items(dataset_items)
        train_source, eval_source = dataset.train_test_split(test_size=1000)

        train_dataset = TextCompletionDataset(
            train_source,
            tokenizer_handle=self.TOKENIZER_HANDLE,
            max_seq_len=self.SEQ_LEN,
        )
        eval_dataset = TextCompletionDataset(
            eval_source,
            tokenizer_handle=self.TOKENIZER_HANDLE,
            max_seq_len=self.SEQ_LEN,
        )
        if packing:
            train_dataset = train_dataset.to_packed_dataset()
            eval_dataset = eval_dataset.to_packed_dataset()

        return train_dataset, eval_dataset

    def _run_sft(self, model, train_dataset, eval_dataset, save_model=True):

        train_dataloader = Dataloader(
            train_dataset,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            dataset_is_sharded_per_host=False,
        )
        eval_dataloader = Dataloader(
            eval_dataset,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            dataset_is_sharded_per_host=False,
        )

        # Create Keras optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.LEARNING_RATE,
            weight_decay=0.01,
        )

        # Create Checkpointer
        checkpointer = Checkpointer(
            self.TMP_DIR,
            model=model,
            save_interval_steps=20,
            max_to_keep=5,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            steps=self.TRAINING_STEPS,
            eval_steps_interval=self.EVAL_STEPS_INTERVAL,
            log_steps_interval=self.LOG_STEPS_INTERVAL,
            max_eval_samples=self.MAX_EVAL_SAMPLES,
            checkpointer=checkpointer,
            tensorboard_dir=os.path.join(self.TMP_DIR, "tensorboard"),
        )

        # Start training
        trainer.train()

        # Save model in HuggingFace format
        if save_model:
            model.save_in_hf_format(os.path.join(self.TMP_DIR, "hf"))

    @unittest.skipIf(int(os.getenv("RUN_LIGHT_TESTS_ONLY", 0)) == 1, "Heavy Test")
    def test_sft_with_maxtext_model(self):
        train_dataset, eval_dataset = self._create_datasets()
        model = MaxTextModel.from_preset(
            preset_handle=self.MODEL_HANDLE,
            seq_len=self.SEQ_LEN,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            precision=self.PRECISION,
            scan_layers=True,
        )
        self._run_sft(model, train_dataset, eval_dataset)

    @unittest.skipIf(int(os.getenv("RUN_LIGHT_TESTS_ONLY", 0)) == 1, "Heavy Test")
    def test_sft_with_kerashub_model(self):
        train_dataset, eval_dataset = self._create_datasets()
        model = KerasHubModel.from_preset(
            preset_handle=self.MODEL_HANDLE, precision=self.PRECISION, lora_rank=16
        )
        self._run_sft(model, train_dataset, eval_dataset)

    @unittest.skipIf(int(os.getenv("RUN_LIGHT_TESTS_ONLY", 0)) == 1, "Heavy Test")
    def test_sft_with_packing(self):
        packed_train_dataset, packed_eval_dataset = self._create_datasets(packing=True)
        model = MaxTextModel.from_random(
            "default",
            seq_len=self.SEQ_LEN,
            per_device_batch_size=self.PER_DEVICE_BATCH_SIZE,
            precision=self.PRECISION,
            scan_layers=True,
        )
        self._run_sft(
            model, packed_train_dataset, packed_eval_dataset, save_model=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
