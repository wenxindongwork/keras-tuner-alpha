"""Unit tests for creating Kithara Dataloaders

Run test on a TPU VM: python -m unittest tests/dataset/test_dataloader_creation.py 
"""

import unittest
from kithara import TextCompletionDataset, Dataloader
import time
import unittest.result
import tests.dataset.utils as dataset_utils
import numpy as np
import os

class TestDataloaderCreation(unittest.TestCase):

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    def _run_test(self, data_source):
        dataset = TextCompletionDataset(
            data_source,
            tokenizer_handle="hf://google/gemma-2-2b",
            model_type="KerasHub",
            max_seq_len=100,
        )
        dataloader = Dataloader(dataset, per_device_batch_size=2)
        for batch in dataloader:
            self.assertTrue(isinstance(batch, dict))
            self.assertTrue("x" in batch)
            self.assertTrue("y" in batch)
            self.assertTrue("token_ids" in batch["x"])
            self.assertTrue("padding_mask" in batch["x"])
            self.assertTrue(isinstance(batch["x"]["token_ids"], np.ndarray))
            self.assertTrue(isinstance(batch["x"]["padding_mask"], np.ndarray))
            self.assertTrue(isinstance(batch["y"], np.ndarray))
            self.assertEqual(len(batch["x"]["token_ids"].shape), 2)
            break

    def test_creating_dataloader_for_fixed_data(self):
        dict_dataset = dataset_utils.create_dict_ray_dataset()
        self._run_test(dict_dataset)
    
    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_creating_dataloader_for_streaming_data(self):
        streaming_dataset = dataset_utils.create_hf_streaming_ray_dataset()
        self._run_test(streaming_dataset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
