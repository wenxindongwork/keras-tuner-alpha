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

"""Unit tests for creating a KerasHub model

Run test on a TPU VM: python -m unittest tests/model/kerashub/test_creation.py 
"""
import unittest
from kithara import KerasHubModel
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
            "hf://google/gemma-2-2b"
        )
    def test_creating_gemma2_2b_model_with_lora(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            lora_rank=6
        )
        
    def test_creating_model_with_precision(self):
        model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            precision="float32"
        )
        self.assertTrue(str(model.trainable_variables[0].value.dtype) == "float32")

    def test_creating_gemma2_9b_model(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-9b"
        )
    def test_creating_gemma2_2b_model_with_lora(self):
        KerasHubModel.from_preset(
            "hf://google/gemma-2-9b",
            lora_rank=16
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
