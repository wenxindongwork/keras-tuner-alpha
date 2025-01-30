"""Unit tests for creating a KerasHub model

Run test on a TPU VM: python -m unittest tests/model/kerashub/test_creation.py 
"""
import unittest
from kithara import KerasHubModel, PredefinedShardingStrategy
import time
import unittest.result

class TestModelCreation(unittest.TestCase):

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    def test_creating_gemma2_2b_model(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )
    def test_creating_gemma2_2b_model_with_lora(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            lora_rank=6,
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )
        
    def test_creating_model_with_precision(self):
        model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            precision="float32",
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )
        self.assertTrue(str(model.trainable_variables[0].value.dtype) == "float32")

    def test_creating_gemma2_9b_model(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-9b",
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )
    def test_creating_gemma2_2b_model_with_lora(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-9b",
            lora_rank=16,
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
