# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import yaml
import argparse
import pprint

from typing import Optional, Union
import yaml
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Base configuration for all training tasks."""

    # Task definition
    task: str = Field("sft", description="Task: sft, continued_pretraining")

    # Model configuration
    model_handle: str = Field(..., description="HuggingFace model name")
    precision: str = Field("mixed_bfloat16", description="Model precision. Supported: float32 , float16, bfloat16, mixed_float16, mixed_bfloat16")
    tokenizer_handle: Optional[str] = Field(
        None, description="HuggingFace tokenizer name"
    )
    seq_len: int = Field(2048, description="Maximum sequence length")

    # Training duration parameters
    num_steps: Optional[int] = Field(None, description="Number of training steps")
    num_epochs: Optional[int] = Field(None, description="Number of training epochs")

    # Evaluation parameters
    eval_steps: Optional[int] = Field(None, description="Evaluate every N steps")
    eval_epochs: Optional[int] = Field(None, description="Evaluate every N epochs")
    logging_steps: int = Field(10, description="Log every N steps")

    # Dataset configuration
    dataset_name: str = Field(..., description="HuggingFace dataset name")
    train_eval_split: float = Field(
        0.1,
        description="Fraction of training data to use for evaluation. >1 to specify number of test samples. 0 to disable evaluation",
    )
    per_device_batch_size: int = Field(1, description="Per device batch size")
    stream_dataset: bool = Field(
        False, description="Whether to load the dataset in streaming mode"
    )

    # Optimization parameters
    optimizer: str = Field(
        "adamw", description="Optimizer type: adamw, adafactor, sgd, or lamb"
    )
    learning_rate: float = Field(2e-5, description="Learning rate for the optimizer")
    lr_scheduler: str = Field(
        "constant",
        description="Learning rate scheduler: exponential, linear, cosine, cosine_w_restarts, polynomial, constant, or inverse_time",
    )

    # Checkpoint configuration
    save_checkpoint_interval: int = Field(
        100, description="Save checkpoint every N steps"
    )
    max_checkpoints_to_keep: int = Field(
        0,
        description="Maximum number of checkpoints to keep. 0 to disable checkpointing",
    )

    # Output directories
    checkpoint_dir: str = Field(
        "checkpoints/", description="Local directory or GCS path for checkpoints"
    )
    tensorboard_dir: str = Field(
        "tensorboard_logs/",
        description="Local directory or GCS path for TensorBoard logs",
    )
    model_output_dir: str = Field(
        "model_output/",
        description="Local directory or GCS path for saving the final model",
    )


class SFTConfig(TrainingConfig):
    """Configuration specific to Supervised Fine-Tuning (SFT) tasks."""

    # Dataset column mappings
    prompt_column: str = Field("prompt", description="Column name for prompts")
    answer_column: str = Field("answer", description="Column name for answers")

    # LoRA parameters
    lora_r: Optional[int] = Field(None, description="LoRA rank. None to disable LoRA.")
    only_save_adapters: bool = Field(
        True, description="Only save LoRA adapters (ignored if lora_r=None)"
    )
    save_adapters_separately: bool = Field(
        True, description="Save adapters separately or merge with base model"
    )


class ContinuedPretrainingConfig(TrainingConfig):
    """Configuration specific to Continued Pretraining tasks."""

    # Dataset configuration
    prompt_column: str = Field("text", description="Column name for prompts")
    sample_packing: bool = Field(
        True, description="Whether to pack multiple samples into a single sequence"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Config loader with overrides")

    parser.add_argument("--template", type=str, help="Path to base config YAML file")

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Command line overrides in key=value format",
    )

    args = parser.parse_args()
    return args


def load_config() -> Union[SFTConfig, ContinuedPretrainingConfig]:
    """Loads the YAML config from a file with a given name."""

    # Parse command line arguments
    args = parse_args()

    # Load base config
    config_file = args.template
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Apply command line overrides
    for arg in args.override:
        if "=" in arg:
            key, value = arg.split("=", 1)
            config[key] = value

    if config["task"] == "sft":
        config = SFTConfig(**config)
    elif config["task"] == "continued_pretraining":
        config = ContinuedPretrainingConfig(**config)
    else:
        raise ValueError(f"Task {config['task']} is not supported.")

    pprint.pprint(config)
    return config
