import keras
import os

os.environ["KERAS_BACKEND"] = "jax"
import ray
import jax
from typing import Union, Optional, List
from datasets import load_dataset
from keras_tuner.trainer import Trainer
from keras_tuner.preprocessor import PretrainingPreprocessor
from keras_tuner.sharding import PredefinedShardingStrategy
from keras_tuner.dataset import Dataloader
from keras_tuner.model import KerasModel

"""Fine-tune Gemma2 2B model using LoRA on TPU devices.

This script demonstrates how to:
1. Set up a Gemma model for LoRA fine-tuning
2. Configure data loading and preprocessing
3. Run training across TPU devices

This script can be run on both single-host and multi-host

Singlehost: python3 examples/ray/TPU/hf_gemma_example_via_ray.py 
Multihost:  python examples/ray/submit_ray_job.py "python3 examples/ray/TPU/hf_gemma_example_via_ray.py" --hf-token <TOKEN>
"""

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
    "dataset_type": "toy",
}


def run_workload(
    train_dataset: ray.data.Dataset,
    data_is_already_split_across_hosts: bool,
    eval_dataset: Optional[ray.data.Dataset] = None,
    train_files: List[str] = None,
    eval_files: List[str] = None,
    dataset_processing_fn=None,
):
    # Log TPU device information
    devices = keras.distribution.list_devices()
    print(f"Available devices: {devices}")
    # Load model
    model = KerasModel(
        model_handle=config["model_handle"],
        precision=config["precision"],
        lora_rank=config["lora_rank"],
        sharding_strategy=PredefinedShardingStrategy(
            parallelism="fsdp", model=config["model"]
        ),
    )

    # Creates preprocessor
    preprocessor = PretrainingPreprocessor(
        tokenizer_handle=config["model_handle"], seq_len=config["seq_len"]
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    # Create dataloaders
    if train_files and eval_files:
        train_dataset, eval_dataset = dataset_processing_fn(train_dataset, eval_dataset)

    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=data_is_already_split_across_hosts,
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=data_is_already_split_across_hosts,
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
        global_batch_size= config["per_device_batch_size"] * jax.device_count()

    )

    # Start training
    trainer.train()

    # Test after tuning
    pred = trainer.generate("What is your name?")
    print("Tuned model generates:", pred)


def create_dataset(option: str) -> ray.data.Dataset:
    train_ds, eval_ds = None, None

    if option == "hf":
        hf_train_dataset = load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )
        hf_val_dataset = load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )

        train_ds = ray.data.from_huggingface(hf_train_dataset)
        eval_ds = ray.data.from_huggingface(hf_val_dataset)

    elif option == "toy":
        dataset_items = [
            {"text": f"{i} What is your name? My name is Mary."} for i in range(1000)
        ]
        dataset = ray.data.from_items(dataset_items)
        train_ds, eval_ds = dataset.train_test_split(test_size=500)

    return train_ds, eval_ds


if __name__ == "__main__":

    train_ds, eval_ds = create_dataset("toy")
    run_workload(
        train_ds, data_is_already_split_across_hosts=False, eval_dataset=eval_ds
    )
