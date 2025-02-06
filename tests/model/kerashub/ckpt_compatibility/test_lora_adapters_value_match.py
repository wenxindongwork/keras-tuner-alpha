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

"""End to end test for KerasHub <> HF LoRA weights conversion correctness.  

Steps: 
    1. Initiate a KerasHub model with LoRA adapters
    2. Save the KerasHub model in PEFT format 
    3. Reload saved model using HuggingFace 
    4. Compare the logits from the model from step 1 and the HuggingFace model

Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/kerashub/ckpt_compatibility/test_lora_adapters_value_match.py
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import unittest
import shutil
import torch
from kithara.utils.gcs_utils import find_cache_root_dir
from kithara import KerasHubModel, PredefinedShardingStrategy
from tests.model.utils import (
    check_arrays_match,
    get_hf_logits,
    get_kerashub_model_input,
    check_predicted_tokens_match,
)
from tests.model.test_prompt import TEST_PROMPT
from transformers import AutoModelForCausalLM


class TestLoRAAdaptersValueMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LORA_RANK = 16
        cls.TMP_DIR = os.path.join(find_cache_root_dir(), "test/ckpt")

    def setUp(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)
    
    def tearDown(self):
        shutil.rmtree(self.TMP_DIR, ignore_errors=True)
    
    def get_kerashub_logits(self, hf_input_ids, model):
        input = get_kerashub_model_input(hf_input_ids)
        logits, _ = model.stateless_call(
            model.trainable_variables,
            model.non_trainable_variables,
            input,
        )
        return logits
    
    def check_adapters_value_match(self, model_id, save_adapter_separately, logits_tol, top1_token_tol):
        # Create model with LoRA adapter
        model = KerasHubModel.from_preset(
            f"hf://{model_id}",
            precision="float32",
            lora_rank=self.LORA_RANK,
            sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
        )

        # Save model
        if save_adapter_separately:
            model.save_in_hf_format(self.TMP_DIR, only_save_adapters=True)
        else:
            model.save_in_hf_format(self.TMP_DIR)

        # Load checkpoint with HuggingFace
        if save_adapter_separately:
            hf_model = AutoModelForCausalLM.from_pretrained(model_id)
            hf_model.load_adapter(self.TMP_DIR)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.TMP_DIR, torch_dtype=torch.float32
            )

        # Compare logits
        input_ids, logits_hf = get_hf_logits(
            model_id, TEST_PROMPT, target_length=512, return_input_ids=True, model=hf_model
        )
        logits_kerashub = self.get_kerashub_logits(input_ids, model)

        with self.subTest("Testing logits match"):
            check_arrays_match(logits_kerashub[0, :5, :], logits_hf[0, :5, :], logits_tol) 
        
        with self.subTest("Testing predicted tokens match"):
            check_predicted_tokens_match(logits_kerashub, logits_hf, top1_token_tol) 

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2b_separate_adapter(self):
        self.check_adapters_value_match(
            "google/gemma-2-2b",
            save_adapter_separately=True,
            logits_tol=1.0,
            top1_token_tol=0.05
        )

    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2b_full_model(self):
        self.check_adapters_value_match(
            "google/gemma-2-2b",
            save_adapter_separately=False,
            logits_tol=1.0,
            top1_token_tol=0.05
        )
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_9b_separate_adapter(self):
        self.check_adapters_value_match(
            "google/gemma-2-9b",
            save_adapter_separately=True,
            logits_tol=1.0,
            top1_token_tol=0.05
        )
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_9b_full_model(self):
        self.check_adapters_value_match(
            "google/gemma-2-9b",
            save_adapter_separately=False,
            logits_tol=1.5,
            top1_token_tol=0.1
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
