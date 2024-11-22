
"""SFT a Gemma2 2B model using LoRA on TPU or GPU.

This script demonstrates how to:
1. Set up a Gemma model for LoRA SFT
2. Configure data loading and preprocessing
3. Run training across TPU/GPU devices

This script can be run on both single-host and multi-host

Singlehost: python3 examples/singlehost/hf_gemma_sft_example.py 
Multihost:  python orchestration/multihost/ray/submit_ray_job.py "python3 examples/multihost/ray/TPU/hf_gemma_sft_example_via_ray.py" --hf-token <TOKEN>
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import ray
import keras
import jax
from typing import Union, Optional, List
from keras_tuner.trainer import Trainer
from keras_tuner.preprocessor import SFTPreprocessor
from keras_tuner.model.sharding import PredefinedShardingStrategy
from keras_tuner.dataset import Dataloader
from keras_tuner.model import KerasModel
from examples.example_datasets import example_datasets


config = {
    "model": "gemma",
    "model_handle": "hf://google/gemma-2-2b",
    "seq_len": 4096,
    "lora_rank": 4,
    "precision": "mixed_bfloat16",
    "training_steps": 100,
    "eval_steps_interval": 10,
    "log_steps_interval": 10,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
}


def run_workload(
    train_dataset: Union[ray.data.Dataset, List[str]],
    dataset_is_sharded_per_host: bool,
    eval_dataset: Optional[ray.data.Dataset] = Optional[
        Union[ray.data.Dataset, List[str]]
    ],
    dataset_processing_fn=None,
):
    # Log TPU device information
    devices = keras.distribution.list_devices()
    print(f"Available devices: {devices}")
    
    # Create model
    model = KerasModel(
        model_handle=config["model_handle"],
        precision=config["precision"],
        lora_rank=config["lora_rank"],
        sharding_strategy=PredefinedShardingStrategy(
            parallelism="fsdp", model=config["model"]
        ),
    )

    # Creates preprocessor
    preprocessor = SFTPreprocessor(
        tokenizer_handle=config["model_handle"], seq_len=config["seq_len"]
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    # Optional processing step for multi-host datasets
    if dataset_processing_fn:
        dataset_processing_fn(train_dataset, eval_dataset)

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
        preprocessor=preprocessor,
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
    pred = trainer.generate("What is your name?")
    print("Tuned model generates:", pred)


if __name__ == "__main__":

    train_ds, eval_ds = example_datasets("sft_toy")
    run_workload(
        train_ds, eval_dataset=eval_ds, dataset_is_sharded_per_host=False,
    )
