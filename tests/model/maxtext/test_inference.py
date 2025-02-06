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

"""Unit tests for correctness of MaxTextModel.generate() function 

Run test on a TPU VM: python -m unittest tests/model/maxtext/test_inference.py 
"""

import unittest
import numpy as np
from transformers import AutoTokenizer
from kithara import MaxTextModel
import time
import unittest.result
import jax
from tests.test_utils import timeout
import os 

@unittest.skipIf(int(os.getenv('RUN_LIGHT_TESTS_ONLY', 0)) == 1, "Heavy Test")
class TestModelGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MaxTextModel.from_random("gemma2-2b", seq_len=100)
        cls.model_input = {
            "tokens": np.array([[i for i in range(100)]] * jax.device_count()),
            "positions": np.array([[i for i in range(100)]] * jax.device_count()),
            "segment_ids": np.array([[1 for _ in range(100)]] * jax.device_count()),
        }

        cls.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        cls.test_prompt = "hello world"

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    @timeout(30)
    def test_generate_without_tokenizer(self):
        with self.assertRaises(AssertionError):
            self.model.generate(self.test_prompt)

        with self.assertRaises(AssertionError):
            self.model.generate(self.model_input, return_decoded=True, max_length=-1)

    @timeout(200)
    def test_generate_with_string_input_decoded(self):
        pred = self.model.generate(
            self.test_prompt,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=True,
        )
        self.assertIsInstance(pred[0], str)
        self.assertLess(len(pred[0].split(" ")), 10)

    @timeout(200)
    def test_generate_with_string_input_not_decoded(self):
        pred = self.model.generate(
            self.test_prompt,
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
        )
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred["token_ids"]), 1)
        self.assertEqual(len(pred["token_ids"][0]), 5)

    @timeout(200)
    def test_generate_with_string_input_strip_prompt_decoded(self):
        pred = self.model.generate(
            self.test_prompt,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=True,
            strip_prompt=True,
        )
        self.assertIsInstance(pred[0], str)
        self.assertFalse(pred[0].startswith(self.test_prompt))

    @timeout(200)
    def test_generate_with_string_input_strip_prompt_not_decoded(self):
        pred = self.model.generate(
            self.test_prompt,
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
            strip_prompt=True,
        )
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred["token_ids"][0]), 2)
        self.assertEqual(len(pred["padding_mask"][0]), 2)
        self.assertEqual(np.sum(pred["padding_mask"][0]), 2)

    @timeout(200)
    def test_generate_with_mutiple_string_input_strip_prompt_not_decoded(self):
        pred = self.model.generate(
            [self.test_prompt, self.test_prompt],
            max_length=5,
            stop_token_ids=[],
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
            strip_prompt=True,
        )
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred["token_ids"][0]), 2)
        self.assertEqual(len(pred["padding_mask"][0]), 2)
        self.assertEqual(np.sum(pred["padding_mask"][0]), 2)

    @timeout(200)
    def test_generate_with_model_input_not_decoded(self):
        pred = self.model.generate(
            self.model_input,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=False,
        )
        print(f"Generated token dictionary keys: {pred.keys()}")
        self.assertIsInstance(pred, dict)

    @timeout(200)
    def test_generate_with_model_input_decoded(self):
        pred = self.model.generate(
            self.model_input,
            max_length=5,
            tokenizer_handle="hf://google/gemma-2-2b",
            return_decoded=True,
        )
        self.assertIsInstance(pred[0], str)

    @timeout(200)
    def test_generate_with_tokenizer_object(self):
        pred = self.model.generate(
            self.model_input,
            tokenizer=self.tokenizer,
            max_length=5,
            return_decoded=True,
        )
        self.assertIsInstance(pred[0], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
