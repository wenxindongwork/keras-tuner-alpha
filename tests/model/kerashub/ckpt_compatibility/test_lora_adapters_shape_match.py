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
    2. Load the reference PEFT model from HuggingFace 
    3. Save the KerasHub model in PEFT format 
    4. Compare the shape of the saved adapter tensors. 

Run script on single host VM: RUN_SKIPPED_TESTS=1 python -m unittest tests/model/kerashub/ckpt_compatibility/test_lora_adapters_shape_match.py
"""
import os
os.environ["KERAS_BACKEND"] = "jax"

import unittest
import shutil
from dataclasses import dataclass
from typing import List, Optional
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from kithara.distributed.sharding import PredefinedShardingStrategy
from kithara import KerasHubModel
from kithara.utils.gcs_utils import find_cache_root_dir

@dataclass
class ModelTestConfig:
    model_id: str
    lora_rank: int
    target_modules: List[str]
    sharding_parallelism: str
    sharding_model: str

class TestLoRAShapeMatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = os.path.join(find_cache_root_dir(), "test/ckpt")
        cls.test_dir = os.path.join(cls.tmp_dir, "test")
        cls.golden_dir = os.path.join(cls.tmp_dir, "golden")

    def setUp(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _init_and_save_kithara_model(self, config: ModelTestConfig):
        kithara_model = KerasHubModel.from_preset(
            f"hf://{config.model_id}",
            lora_rank=config.lora_rank,
            sharding_strategy=PredefinedShardingStrategy(
                parallelism=config.sharding_parallelism,
                model=config.sharding_model
            ),
        )
        kithara_model.save_in_hf_format(self.test_dir, only_save_adapters=True)

    def _init_and_save_peft_model(self, config: ModelTestConfig):
        base_model = AutoModelForCausalLM.from_pretrained(config.model_id)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.target_modules,
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.save_pretrained(self.golden_dir)

    def _compare_weights_shape(self):
        # Load golden adapter weights
        with safe_open(
            os.path.join(self.golden_dir, "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        ) as f:
            golden_tensors = {key: f.get_tensor(key) for key in f.keys()}

        # Load and compare test adapter weights
        with safe_open(
            os.path.join(self.test_dir, "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        ) as f:
            test_tensors = {}
            for key in f.keys():
                self.assertIn(key, golden_tensors, f"Missing key in golden tensors: {key}")
                test_tensors[key] = f.get_tensor(key)
                self.assertEqual(
                    golden_tensors[key].shape,
                    test_tensors[key].shape,
                    f"Shape mismatch: {key}. Golden shape is {golden_tensors[key].shape}, "
                    f"actual shape is {test_tensors[key].shape}"
                )

        self.assertEqual(
            len(golden_tensors),
            len(test_tensors),
            "Number of tensors mismatch"
        )

    def _run_test_for_config(self, config: ModelTestConfig):
        self._init_and_save_kithara_model(config)
        self._init_and_save_peft_model(config)
        self._compare_weights_shape()
    
    @unittest.skipIf(int(os.getenv('RUN_SKIPPED_TESTS', 0)) != 1, "Manual Test")
    def test_gemma_2b(self):
        config = ModelTestConfig(
            model_id="google/gemma-2-2b",
            lora_rank=16,
            target_modules=["q_proj", "v_proj"],
            sharding_parallelism="fsdp",
            sharding_model="gemma"
        )
        self._run_test_for_config(config)

if __name__ == "__main__":
    unittest.main(verbosity=2)
