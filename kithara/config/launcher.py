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

"""Config Launcher

1. Creates model, optimizer, and datasets according to the config
2. Runs training loop
3. Saves model in HuggingFace format

Supports both singlehost and multihost training.
Supports both SFT and continued pretraining tasks.

Usage:
  Singlehost: 
    # Use default config
    python config/launcher.py

    # Use a different base config
    python config/launcher.py --config=your_base_config.yaml

    # Apply YAML overrides
    python config/launcher.py --override_config=your_override_config.yaml

    # Use command line overrides for quick experiments
    python config/launcher.py --override learning_rate=5e-5 training_steps=10000

  Multihost:  
    Wrap your singlehost command in ray/submit_job.py and pass your HuggingFace token:
    
    python ray/submit_job.py "python3.11 kithara/config/launcher.py --single_host=False --tpu_generation=v5e" --hf-token your_token

If you experience OOM error during model checkpoint loading/saving, it is because your host VM
does not have enough capacity to load/save the model. Consider mounting extra memory onto your VM,
and launch this script with:
  `HF_HOME=new_hf_cache_dir KERAS_HOME=new_keras_cache_dir python config/launcher.py`

E.g. `HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python config/launcher.py`
"""

import os
from typing import Dict, Optional, Tuple, Union, Any

# Set Keras backend before any other imports
os.environ["KERAS_BACKEND"] = "jax"

from absl import app
import keras
import ray
import jax
from datasets import load_dataset
from typing import List, Any
from kithara import (
    MaxTextModel,
    KerasHubModel,
    Dataloader,
    TextCompletionDataset,
    SFTDataset,
    Trainer,
    Checkpointer,
)
from kithara.config.pyconfig import load_config
from kithara.distributed.data import split_dataset


def create_model(config: Dict[str, Any]) -> Union[KerasHubModel, MaxTextModel]:
    """Creates and returns a model based on configuration.
    
    Args:
        config: Dictionary containing model configuration parameters.
        
    Returns:
        A KerasHubModel (if using LoRA) or MaxTextModel instance.
    """
    if config["use_lora"]:
        model = KerasHubModel.from_preset(
            config['model_handle'],
            precision=config["precision"],
            lora_rank=config["lora_r"],
        )
    else:
        model = MaxTextModel.from_preset(
            config['model_handle'],
            seq_len=config["seq_len"],
            per_device_batch_size=config["per_device_batch_size"],
            precision=config["precision"],
            scan_layers=config["scan_layers"],
        )
    return model


def create_optimizer(config: Dict[str, Any]) -> keras.optimizers.Optimizer:
    """Creates and returns an optimizer based on configuration.
    
    Args:
        config: Dictionary containing optimizer configuration parameters.
        
    Returns:
        A Keras optimizer instance.
        
    Raises:
        ValueError: If an invalid optimizer type is specified.
    """
    optimizer_type = config["optimizer"].lower()
    optimizer_params = {
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
    }
    
    optimizer_map = {
        "adamw": keras.optimizers.AdamW,
        "adafactor": keras.optimizers.Adafactor,
        "sgd": keras.optimizers.SGD,
        "lamb": keras.optimizers.Lamb,
    }
    
    if optimizer_type not in optimizer_map:
        raise ValueError(f"Invalid optimizer '{optimizer_type}'. Valid options: {list(optimizer_map.keys())}")
    
    return optimizer_map[optimizer_type](**optimizer_params)


def create_datasets(config: Dict[str, Any]) -> Tuple[Union[SFTDataset, TextCompletionDataset], 
                                                   Optional[Union[SFTDataset, TextCompletionDataset]]]:
    """Creates and returns training and evaluation datasets based on configuration.
    
    Args:
        config: Dictionary containing dataset configuration parameters.
        
    Returns:
        A tuple of (train_dataset, eval_dataset) where eval_dataset may be None.
        
    Raises:
        ValueError: If invalid dataset type or task is specified.
    """
    # Determine dataset class based on task
    if config["task"] == "sft":
        dataset_class = SFTDataset
    elif config["task"] == "continued_pretraining":
        dataset_class = TextCompletionDataset
    else:
        raise ValueError(f"Invalid task '{config['task']}'. Valid options: 'sft', 'continued_pretraining'")

    # Create training dataset
    if config["train_dataset_type"] == "huggingface":
        hf_train_dataset = load_dataset(
            path=config["train_hf_dataset_path"],
            name=config["train_hf_dataset_name"],
            data_dir=config["train_hf_dataset_dir"],
            data_files=config["train_hf_data_files"],
            split=config["train_dataset_hf_split"],
            streaming=config["train_streaming_mode"]
        )
        ray_train_dataset = ray.data.from_huggingface(hf_train_dataset)
    else:
        raise ValueError(f"Invalid dataset type '{config['train_dataset_type']}'. Currently only 'huggingface' is supported.")
    
    # Handle train/eval split if specified
    ray_eval_dataset = None
    if config["train_eval_split"]:
        ray_train_dataset, ray_eval_dataset = ray_train_dataset.train_test_split(
            test_size=config["train_eval_split"], 
            shuffle=config["train_eval_split_shuffle"]
        )
    # Otherwise, load separate eval dataset if specified
    elif config["eval_dataset_type"] is not None:
        if config["eval_dataset_type"] == "huggingface":
            hf_eval_dataset = load_dataset(
                path=config["eval_hf_dataset_path"],
                name=config["eval_hf_dataset_name"],
                data_dir=config["eval_hf_dataset_dir"],
                data_files=config["eval_hf_data_files"],
                split=config["eval_dataset_hf_split"],
                streaming=config["eval_streaming_mode"]
            )
            ray_eval_dataset = ray.data.from_huggingface(hf_eval_dataset)
        else:
            raise ValueError(f"Invalid dataset type '{config['eval_dataset_type']}'. Currently only 'huggingface' is supported.")
    
    # Convert Ray datasets to Kithara datasets
    train_dataset = dataset_class(
        ray_train_dataset,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
        column_mapping=config["train_dataset_column_mapping"],
    )
    
    # Apply sample packing if configured
    if config["train_sample_packing"]:
        train_dataset = train_dataset.to_packed_dataset()
    
    # Create and process evaluation dataset if available
    eval_dataset = None
    if ray_eval_dataset is not None:
        eval_dataset = dataset_class(
            ray_eval_dataset,
            tokenizer_handle=config["tokenizer_handle"],
            max_seq_len=config["seq_len"],
            column_mapping=config["eval_dataset_column_mapping"],
        )
        
        if config["eval_sample_packing"]:
            eval_dataset = eval_dataset.to_packed_dataset()
    
    return train_dataset, eval_dataset


def launch_task(
    config: Dict[str, Any],
    train_dataset: Union[SFTDataset, TextCompletionDataset],
    eval_dataset: Optional[Union[SFTDataset, TextCompletionDataset]],
    dataset_is_sharded_per_host: bool = False,
) -> None:
    """Sets up and runs the training process.
    
    Args:
        config: Dictionary containing training configuration parameters
        train_dataset: Dataset to use for training
        eval_dataset: Optional dataset to use for evaluation
        dataset_is_sharded_per_host: Whether datasets are pre-sharded per host
    """
    # Create model and optimizer
    model = create_model(config)
    optimizer = create_optimizer(config)
    
    # Create dataloaders
    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=dataset_is_sharded_per_host,
    )
    
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = Dataloader(
            eval_dataset,
            per_device_batch_size=config["per_device_batch_size"],
            dataset_is_sharded_per_host=dataset_is_sharded_per_host,
        )
    
    # Set up checkpointing
    if config["save_checkpoints"]:
        checkpointer = Checkpointer(
            config["checkpoint_dir"],
            model=model,
            save_interval_steps=config["save_checkpoint_interval"],
            max_to_keep=config["max_checkpoints_to_keep"]
        )
    else:
        checkpointer = None
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=config["training_steps"],
        epochs=config["epochs"],
        log_steps_interval=config["log_steps_interval"],
        eval_steps_interval=config["eval_steps_interval"],
        eval_epochs_interval=config["eval_epochs_interval"],
        max_eval_samples=config["max_eval_samples"],
        tensorboard_dir=config["tensorboard_dir"],
        checkpointer=checkpointer,
    )
    
    # Start training
    trainer.train()
    
    # Save model in HuggingFace format
    if config["save_model"]:
        model.save_in_hf_format(
            config["model_output_dir"],
            only_save_adapters=config["only_save_adapters"],
            save_adapters_separately=config["save_adapters_separately"]
        )

def single_host_launcher(config):
    
    train_dataset, eval_dataset = create_datasets(config)
    
    launch_task(
        config,
        train_dataset,
        eval_dataset
    )

def multi_host_launcher(config):
    
    ray.init()

    # 4 chips per host: v2, v4, v5
    # 8 chips per host: v3, v4e, v5e, v6e
    num_chips_per_host = 4 if config["tpu_generation"] in ["v2", "v4", "v5"] else 8
    num_tpu_devices = int(ray.cluster_resources()["TPU"])
    num_tpu_hosts = num_tpu_devices // num_chips_per_host

    @ray.remote(resources={"TPU": num_chips_per_host})
    def ray_launch_task(config, train_dataset, eval_dataset, dataset_is_sharded_per_host=False):
        
        # HuggingFace login
        from huggingface_hub import login
        import os

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        jax.distributed.initialize()
        
        launch_task(config, train_dataset, eval_dataset, dataset_is_sharded_per_host)
    
    train_dataset, eval_dataset = create_datasets(config)
    
    if config["split_data_across_host"]:
        train_dataset: List[Any] = split_dataset(train_dataset, num_hosts=num_tpu_hosts)
        eval_dataset: List[Any] = split_dataset(eval_dataset, num_hosts=num_tpu_hosts)
        
        ray.get(
            [
                ray_launch_task.remote(config, train_dataset[i], eval_dataset[i], dataset_is_sharded_per_host=True)
                for i in range(num_tpu_hosts)
            ]
        )
    else:
        ray.get(
            [
                ray_launch_task.remote(config, train_dataset, eval_dataset)
                for _ in range(num_tpu_hosts)
            ]
        )

    ray.shutdown()

def main():
    """Main function to load config and start training.
    
    Args:
        argv: Command line arguments
    """
    # Load configuration from YAML file
    config = load_config()
    
    # Start training
    if config["single_host"]:
        single_host_launcher(config)
    else:
        multi_host_launcher(config)

if __name__ == "__main__":
    main()