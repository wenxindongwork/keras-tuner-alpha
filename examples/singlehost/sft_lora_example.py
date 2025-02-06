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

"""SFT a Gemma2 2B model using LoRA on TPU or GPU.

This script demonstrates how to:
1. Set up a Gemma2 model for LoRA SFT
2. Load HuggingFace Gemma2 checkpoint
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices

This script can be run on both single-host and multi-host. For multi-host set up, please follow `ray/readme.md`.

Singlehost: python examples/singlehost/sft_lora_example.py 
Multihost:  kithara multihost examples/multihost/ray/TPU/sft_lora_example.py --hf-token <TOKEN>
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
from transformers import AutoTokenizer
from typing import Union, Optional, List
from kithara import (
    KerasHubModel,
    Dataloader,
    Trainer,
    PredefinedShardingStrategy,
    SFTDataset,
)
import jax 

config = {
    "model": "gemma",
    "model_handle": "google/gemma-2-2b",
    "seq_len": 4096,
    "use_lora": True,
    "lora_rank": 4,
    "precision": "mixed_bfloat16",
    "training_steps": 100,
    "eval_steps_interval": 10,
    "log_steps_interval": 10,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
}


def run_workload(
    train_source: ray.data.Dataset,
    eval_source: Optional[ray.data.Dataset] = None,
    dataset_is_sharded_per_host: bool = False,
):
    # Log TPU device information
    devices = jax.devices()
    print(f"Available devices: {devices}")

    # Create model
    model = KerasHubModel.from_preset(
        f"hf://{config['model_handle']}",
        precision=config["precision"],
        lora_rank=config["lora_rank"] if config["use_lora"] else None,
        sharding_strategy=PredefinedShardingStrategy(
            parallelism="fsdp", model=config["model"]
        ),
    )
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_handle"])

    # Creates datasets
    train_dataset = SFTDataset(
        train_source,
        tokenizer=tokenizer,
        max_seq_len=config["seq_len"],
    )
    eval_dataset = SFTDataset(
        eval_source, tokenizer=tokenizer, max_seq_len=config["seq_len"]
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    # Create data loaders
    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=dataset_is_sharded_per_host,
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=dataset_is_sharded_per_host,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=config["training_steps"],
        eval_steps_interval=config["eval_steps_interval"],
        max_eval_samples=config["max_eval_samples"],
        log_steps_interval=config["log_steps_interval"],
    )

    # Start training
    trainer.train()

    # Test after tuning
    pred = model.generate(
        "What is your name?", max_length=30, tokenizer=tokenizer, return_decoded=True
    )
    print("Tuned model generates:", pred)


if __name__ == "__main__":

    dataset_items = [{
        "prompt": "What is your name?",
        "answer": "My name is Mary",
    }  for _ in range(1000)]
    dataset = ray.data.from_items(dataset_items)
    train_ds, eval_ds= dataset.train_test_split(test_size=500)

    run_workload(
        train_ds,
        eval_ds,
        dataset_is_sharded_per_host=False,
    )
