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

"""Full parameter continued pretraining a Gemma2 9B model.

This script demonstrates how to

1. Load HuggingFace Gemma2 checkpoint
2. Loading large HuggingFace dataset
3. Configure data loading and preprocessing with sequence packing
4. Run training across TPU/GPU devices
5. Save checkpoint to GCS periodically 
6. Generate text using the trained model
7. Save model in HuggingFace format to GCS

This script can be run on both single-host and multi-host. 
For mulit-host set up, please follow https://kithara.readthedocs.io/en/latest/scaling_with_ray.html.

Singlehost: python examples/singlehost/continued_pretraining_example.py 
Multihost:  python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/continued_pretraining_example.py" --hf-token your_token
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import ray
from kithara import (
    MaxTextModel,
    Dataloader,
    TextCompletionDataset,
    SFTDataset,
    Trainer,
    Checkpointer,
)
from datasets import load_dataset

config = {
    "model_handle": "hf://google/gemma-2-{}",
    "tokenizer_handle": "hf://google/gemma-2-2b",
    "seq_len": 4096,
    "precision": "mixed_bfloat16",
    "training_steps": 100,  # Set to a higher number for your actual training
    "eval_steps_interval": 20,
    "log_steps_interval": 1,
    "checkpoint_interval": 5,
    "max_checkpoints_to_keep": 5,
    "per_device_batch_size": 1,
    "max_eval_samples": 500,
    "learning_rate": 5e-5,
}


def run_workload(
    model_output_dir: str = "model_output",
    gemma2_model_size: str = "2b",  # Options: 2b, 9b, 27b
):
    assert gemma2_model_size in ["2b", "9b", "27b"]

    # Load data
    hf_train_dataset = load_dataset(
        "open-web-math/open-web-math", split="train", streaming=True
    )
    # We are showing in this example that it is possible in Kithara to mix SFT with TextCompletion Datasets for training.
    # For example, you can evaluate on a QA style evaluation dataset while training on a text completion style dataset.
    hf_eval_dataset = load_dataset("openai/gsm8k", "main", split="test")
    train_source = ray.data.from_huggingface(hf_train_dataset)
    eval_source = ray.data.from_huggingface(hf_eval_dataset)

    # Create Dataset
    train_dataset = TextCompletionDataset(
        train_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
    ).to_packed_dataset()  # Activate sequence packing

    eval_dataset = SFTDataset(
        eval_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
        column_mapping={"prompt": "question", "answer": "answer"},
    ).to_packed_dataset()

    # Create Dataloaders
    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=config["per_device_batch_size"],
    )

    # Create Model
    model = MaxTextModel.from_preset(
        config["model_handle"].format(gemma2_model_size),
        seq_len=config["seq_len"],
        per_device_batch_size=config["per_device_batch_size"],
        precision=config["precision"],
        scan_layers=True if gemma2_model_size in ["2b", "9b"] else False,
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
    )

    checkpointer = Checkpointer(
        os.path.join(model_output_dir, "ckpt"),
        model=model,
        save_interval_steps=config["checkpoint_interval"],
        max_to_keep=config["max_checkpoints_to_keep"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=config["training_steps"],
        eval_steps_interval=config["eval_steps_interval"],
        log_steps_interval=config["log_steps_interval"],
        max_eval_samples=config["max_eval_samples"],
        checkpointer=checkpointer,
        tensorboard_dir=os.path.join(model_output_dir, "tensorboard"),
    )

    trainer.train()

    # Test it out!
    pred = model.generate(
        "Harry slept 9 hours last night. His friend James slept only 2/3 of what Harry slept. How many more hours did Harry sleep than James?",
        max_length=1000,
        tokenizer_handle=config["model_handle"],
        skip_special_tokens=True,
        return_decoded=True,
        strip_prompt=True,
    )
    print(
        f"Since we've only trained this model for 100 steps, the output is not going to make sense. Tuned model generated {pred}"
    )

    model.save_in_hf_format(model_output_dir)


if __name__ == "__main__":

    run_workload(model_output_dir="model_output/")
