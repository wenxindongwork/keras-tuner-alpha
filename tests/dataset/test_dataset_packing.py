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

Run test on a TPU VM: python -m unittest tests/dataset/test_dataset_packing.py 
"""

import unittest
from kithara import SFTDataset, TextCompletionDataset
import time
import unittest.result
import tests.dataset.utils as dataset_utils
import numpy as np
import os
import ray


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
        self.assertTrue("tokens" in element["x"])
        self.assertTrue("segment_ids" in element["x"])
        self.assertTrue("positions" in element["x"])
        self.assertTrue(isinstance(element["x"]["tokens"], np.ndarray))
        self.assertTrue(isinstance(element["x"]["segment_ids"], np.ndarray))
        self.assertTrue(isinstance(element["x"]["positions"], np.ndarray))
        self.assertTrue(isinstance(element["y"], np.ndarray))

    def test_creating_text_completion_packing_dataset(self):
        dataset_items = [{"text": f"T"} for i in range(10 * 2)]
        ray_dataset = ray.data.from_items(dataset_items)
        dataset = TextCompletionDataset(
            ray_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            model_type="MaxText",
            max_seq_len=30,
        ).to_packed_dataset()

        num_packed_elements = 0
        for element in dataset:
            num_packed_elements += 1
            self._check_dataset_item_format(element)
            # Segment id should look like 1,1,1,2,2,2,3,3,3,...10,10,10
            self.assertEqual(np.max(element["x"]["segment_ids"]), 10)
            self.assertEqual(np.min(element["x"]["segment_ids"]), 1)

            # Positions should look like 0, 1,2,0, 1,2,,...., 0, 1,2
            self.assertEqual(np.max(element["x"]["positions"]), 2)
            self.assertEqual(np.min(element["x"]["positions"]), 0)
        self.assertEqual(num_packed_elements, 2)

    def test_creating_small_dataset(self):
        dataset_items = [{"text": f"T"} for i in range(50)]
        ray_dataset = ray.data.from_items(dataset_items)
        dataset = TextCompletionDataset(
            ray_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            model_type="MaxText",
            max_seq_len=300,
        ).to_packed_dataset()

        self.assertEqual(len(dataset), 50)
        self.assertIsNotNone(next(iter(dataset)))

    def test_creating_sft_packing_dataset(self):
        dataset_items = [{"prompt": "T", "answer": "0"} for i in range(1000)]
        ray_dataset = ray.data.from_items(dataset_items)
        dataset = SFTDataset(
            ray_dataset,
            tokenizer_handle="hf://google/gemma-2-2b",
            model_type="MaxText",
            max_seq_len=400,
        ).to_packed_dataset()

        for element in dataset:
            self._check_dataset_item_format(element)
            # Segment id should look like 1,1,1,1,2,2,2,2,3,3,3,...100,100,100, 100
            self.assertEqual(np.max(element["x"]["segment_ids"]), 100)
            self.assertEqual(np.min(element["x"]["segment_ids"]), 1)

            # Positions should look like 0, 1,2,3,4, 0, 1,2,3,4,...., 0, 1,2,3,4
            self.assertEqual(np.max(element["x"]["positions"]), 3)
            self.assertEqual(np.min(element["x"]["positions"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
