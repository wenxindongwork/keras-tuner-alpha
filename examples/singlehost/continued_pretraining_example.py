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

"""Full parameter finetune a Gemma2 9B model.

This script demonstrates how to:
1. Set up a Gemma2 model for full parameter continued pretraining
2. Load HuggingFace Gemma2 checkpoint
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices
5. Save checkpoint to GCS periodically 
6. Generate text using the trained model
7. Save model in HuggingFace format to GCS

This script should be run on multihost, since gemma2-9b will not fit on a single TPU host. However, 
you can change the model to `gemma2-2b` to run on single host. 

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
import jax
from datasets import load_dataset

config = {
    "model_handle": "hf://google/gemma-2-2b",
    "tokenizer_handle": "hf://google/gemma-2-2b",
    "seq_len": 4096,
    "precision": "mixed_bfloat16",
    "training_steps": 100, # Set to a higher number for actual training
    "eval_steps_interval": 20,
    "log_steps_interval": 1,
    "checkpoint_interval": 50,
    "max_checkpoints_to_keep": 5,
    "per_device_batch_size": 1,
    "max_eval_samples": 500,
    "model_output_dir": "gs://wenxindong-tpu-prod-env-multipod-bucket/ckpt",
    "learning_rate": 5e-5,
    "tensorboard_dir": "gs://wenxindong-tpu-prod-env-multipod-bucket/tensorboard",
}


def run_workload(
    train_source: ray.data.Dataset,
    eval_source: ray.data.Dataset,
    dataset_is_sharded_per_host: bool,
):

    # Create Model
    model = MaxTextModel.from_preset(
        config["model_handle"],
        seq_len=config["seq_len"],
        per_device_batch_size=config["per_device_batch_size"],
        precision=config["precision"],
        scan_layers=True,
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
    )

    # Create Dataset
    train_dataset = TextCompletionDataset(
        train_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
    )
    eval_dataset = SFTDataset(
        eval_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
        column_mapping={"prompt": "question", "answer": "answer"},
    )

    # Create Dataloaders
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

    # Create Checkpointer
    checkpointer = Checkpointer(
        config["model_output_dir"], 
        model=model, 
        save_interval_steps=config["checkpoint_interval"], 
        max_to_keep=config["max_checkpoints_to_keep"]
    )

    # Initialize trainer
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
        tensorboard_dir=config["tensorboard_dir"],
    )

    # Start training
    trainer.train()

    # Generate text after training
    pred = model.generate(
        "Harry slept 9 hours last night. His friend James slept only 2/3 of what Harry slept. How many more hours did Harry sleep than James?",
        max_length=1000,
        tokenizer_handle=config["model_handle"],
        skip_special_tokens=True,
        return_decoded=True,
        strip_prompt=True,
    )
    print(f"Tuned model generated {pred}")

    # Save model in HuggingFace format
    model.save_in_hf_format(config["model_output_dir"] + "hf/")


if __name__ == "__main__":

    # Pretraining style training dataset 
    hf_train_dataset = load_dataset(
        "open-web-math/open-web-math", split="train", streaming=True
    )

    # QA style evaluation dataset. You can also use a split of open-web-math for evaluation.
    hf_eval_dataset = load_dataset("openai/gsm8k", 'main', split="test")

    train_source = ray.data.from_huggingface(hf_train_dataset)
    eval_source = ray.data.from_huggingface(hf_eval_dataset)

    run_workload(
        train_source,
        eval_source,
        dataset_is_sharded_per_host=False,
    )
