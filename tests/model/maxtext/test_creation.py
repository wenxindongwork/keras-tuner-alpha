"""Unit tests for creating a MaxText model

Run test on a TPU VM: python -m unittest tests/model/maxtext/test_creation.py 
"""
import unittest
from kithara import MaxTextModel
import time
import unittest.result
import os 

class TestModelCreation(unittest.TestCase):

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    def test_creating_random_gemma2_2b_model(self):
        MaxTextModel.from_random(
            "gemma2-2b",
            seq_len=100, 
            per_device_batch_size=1,
        )

    def test_creating_preset_gemma2_2b_model(self):
        MaxTextModel.from_preset(
            "hf://google/gemma-2-2b",
            seq_len=100, 
            per_device_batch_size=1,
        )

    def test_creating_preset_gemma2_2b_model_with_scan(self):
        MaxTextModel.from_preset(
            "hf://google/gemma-2-2b",
            seq_len=100, 
            per_device_batch_size=1,
            scan_layers=True
        )
        
    def test_creating_random_model_with_precision(self):
        model = MaxTextModel.from_random(
            "gemma2-2b",
            seq_len=100, 
            per_device_batch_size=1,
            precision="bfloat16",
        )
        self.assertTrue(str(model.trainable_variables[0].value.dtype) == "bfloat16")

    def test_creating_preset_model_with_precision(self):
        model = MaxTextModel.from_preset(
            "hf://google/gemma-2-2b",
            seq_len=100, 
            per_device_batch_size=1,
            precision="bfloat16",
        )
        self.assertTrue(str(model.trainable_variables[0].value.dtype) == "bfloat16")

    def test_creating_random_gemma2_2b_model(self):
        MaxTextModel.from_random(
            "gemma2-9b",
            seq_len=100, 
            per_device_batch_size=1,
        )

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS', 0)) != 1, "Heavy Test")
    def test_creating_preset_gemma2_9b_model(self):
        MaxTextModel.from_preset(
            "hf://google/gemma-2-9b",
            seq_len=100, 
            per_device_batch_size=1,
        )

    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS', 0)) != 1, "Heavy Test")
    def test_creating_preset_gemma2_9b_model_with_scan(self):
        MaxTextModel.from_preset(
            "hf://google/gemma-2-9b",
            seq_len=100, 
            per_device_batch_size=1,
            scan_layers=True
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
