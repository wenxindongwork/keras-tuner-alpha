"""Unit tests for correctness of KerasHubModel.generate() function 

Run test on a TPU VM: python -m unittest tests/model/kerashub/inference.py 
"""
import unittest
import numpy as np
from transformers import AutoTokenizer
from kithara import KerasHubModel, PredefinedShardingStrategy
import time
import signal
from functools import wraps
import unittest.result

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")

            # Set the timeout handler
            original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the original handler and disable the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            return result
        return wrapper
    return decorator

class TestModelGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Starting test setup...")
        start_time = time.time()
        
        print("Initializing KerasHubModel...")
        cls.model = KerasHubModel.from_preset(
            "hf://google/gemma-2-2b",
            lora_rank=6,
            sharding_strategy=PredefinedShardingStrategy("fsdp", "gemma"),
        )
        print(f"Model initialization took {time.time() - start_time:.2f} seconds")
        
        cls.model_input = {
            "token_ids": np.array([[1, 2, 3, 0, 0]]),
            "padding_mask": np.array([[1, 1, 1, 0, 0]]),
        }
        
        print("Loading tokenizer...")
        tokenizer_start = time.time()
        cls.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        print(f"Tokenizer loading took {time.time() - tokenizer_start:.2f} seconds")
        
        cls.test_prompt = "hello world"
        print("Test setup completed successfully")

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
            return_decoded=True
        )
        self.assertIsInstance(pred[0], str)
    

if __name__ == '__main__':
    unittest.main(verbosity=2)
