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

"""Unit tests for creating Kithara Datasets

Run test on a TPU VM: python -m unittest tests/dataset/test_dataset_creation.py 
"""

import unittest
from kithara import SFTDataset, TextCompletionDataset
import time
import unittest.result
import tests.dataset.utils as dataset_utils
import numpy as np
import os

class TestDatasetCreation(unittest.TestCase):

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    def _check_dataset_item_format(self, element):
        self.assertTrue(isinstance(element, dict))
        self.assertTrue("x" in element)
        self.assertTrue("y" in element)
        self.assertTrue("token_ids" in element["x"])
        self.assertTrue("padding_mask" in element["x"])
        self.assertTrue(isinstance(element["x"]["token_ids"], np.ndarray))
        self.assertTrue(isinstance(element["x"]["padding_mask"], np.ndarray))
        self.assertTrue(isinstance(element["y"], np.ndarray))

    def test_creating_sft_dataset_from_dict(self):
        dict_dataset = dataset_utils.create_dict_ray_dataset()
        dataset = SFTDataset(
            dict_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            column_mapping={"prompt": "text", "answer": "text"},
            model_type="KerasHub",
        )
        self.assertIsNotNone(len(dataset))
        for element in dataset:
            self._check_dataset_item_format(element)
            break

    def test_creating_sft_dataset_fail_when_tokenizer_is_not_provided(self):
        dict_dataset = dataset_utils.create_dict_ray_dataset()
        with self.assertRaises(AssertionError):
            SFTDataset(dict_dataset, model_type="KerasHub")

    def test_creating_sft_dataset_from_csv(self):
        csv_dataset = dataset_utils.create_csv_ray_dataset()
        dataset = SFTDataset(
            csv_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            column_mapping={
                "prompt": "sepal length (cm)",
                "answer": "sepal length (cm)",
            },
            model_type="KerasHub",
        )
        self.assertIsNotNone(len(dataset))
        for element in dataset:
            self._check_dataset_item_format(element)
            break
    
    @unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
    def test_creating_text_completion_dataset_from_hf(self):
        streaming_dataset = dataset_utils.create_hf_streaming_ray_dataset()
        dataset = TextCompletionDataset(
            streaming_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            model_type="KerasHub",
        )
        self.assertTrue(len(dataset) == 364608)
        for element in dataset:
            self._check_dataset_item_format(element)
            break


if __name__ == "__main__":
    unittest.main(verbosity=2)
