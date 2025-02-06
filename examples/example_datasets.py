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

from typing import Tuple
from datasets import load_dataset
import ray
from ray.data import Dataset


def example_datasets(option: str) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Load examples datasets based on the specified option.

    Returns:
        Tuple[ray.data.Dataset, ray.data.Dataset]: A tuple of (training_dataset, evaluation_dataset)

    Example usage:
        train_ds, eval_ds = example_datasets("hf")
    """
    valid_options = {"hf", "finetune_toy", "sft_toy", "files"}
    if option not in valid_options:
        raise ValueError(
            f"Invalid option: {option}. Must be one of {', '.join(valid_options)}"
        )

    if option == "hf":
        return _load_huggingface_dataset()
    elif option == "finetune_toy":
        return _create_finetune_toy_dataset()
    elif option == "sft_toy":
        return _create_sft_toy_dataset()
    elif option == "files":
        raise NotImplementedError(
            "Loading from files is not yet implemented. Please use another option."
        )


def _load_huggingface_dataset() -> Tuple[Dataset, Dataset]:
    """Load the C4 dataset from HuggingFace."""
    hf_train_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    hf_val_dataset = load_dataset(
        "allenai/c4", "en", split="validation", streaming=True
    )

    return (
        ray.data.from_huggingface(hf_train_dataset),
        ray.data.from_huggingface(hf_val_dataset),
    )


def _create_finetune_toy_dataset() -> Tuple[Dataset, Dataset]:
    """Create a toy dataset for finetuning with 1000 examples."""
    dataset_items = [
        {"text": f"{i} What is your name? My name is Mary."} for i in range(1000)
    ]
    dataset = ray.data.from_items(dataset_items)
    return dataset.train_test_split(test_size=500)


def _create_sft_toy_dataset() -> Tuple[Dataset, Dataset]:
    """Create a toy dataset for supervised fine-tuning with 1000 examples."""
    dataset_items = [{
        "prompt": "What is your name?",
        "answer": "My name is Mary",
    }  for _ in range(1000)]
    dataset = ray.data.from_items(dataset_items)
    return dataset.train_test_split(test_size=500)
