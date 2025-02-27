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

"""SFT a Gemma2 2B model using LoRA on TPU or GPU on an Alpaca Dataset

This script demonstrates how to:
1. Set up a Gemma2 model for LoRA SFT
2. Load HuggingFace Gemma2 checkpoint
3. Load HuggingFace Dataset 
4. Configure data loading and preprocessing
5. Run training across TPU/GPU devices
6. Save the LoRA adapters

This script can be run on both single-host and multi-host. 
For mulit-host set up, please follow https://kithara.readthedocs.io/en/latest/scaling_with_ray.html.

Singlehost: python examples/singlehost/sft_lora_example.py 
Multihost:  python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token

"""

import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from datasets import load_dataset
from kithara import (
    KerasHubModel,
    Dataloader,
    Trainer,
    SFTDataset,
)

config = {
    "model_handle": "hf://google/gemma-2-2b",
    "tokenizer_handle": "hf://google/gemma-2-2b",
    "seq_len": 4096,
    "lora_rank": 4,
    "precision": "mixed_bfloat16",
    "training_steps": 100,
    "eval_steps_interval": 10,
    "log_steps_interval": 10,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
}


def run_workload(model_output_dir=None):

    # Load and split the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    datasets = dataset.train_test_split(test_size=200)
    train_source, eval_source = datasets["train"], datasets["test"]

    # Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:"""

    def formatting_prompts_func(examples):
        """Format examples using the Alpaca prompt template"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction=instruction, input=input)
            texts.append(text)
        return {"prompt": texts, "answer": outputs}

    # Creates datasets
    train_dataset = SFTDataset(
        train_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
        custom_formatting_fn=formatting_prompts_func,
    )
    eval_dataset = SFTDataset(
        eval_source,
        tokenizer_handle=config["tokenizer_handle"],
        max_seq_len=config["seq_len"],
        custom_formatting_fn=formatting_prompts_func,
    )

    # Create data loaders
    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=False,
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=config["per_device_batch_size"],
        dataset_is_sharded_per_host=False,
    )

    # Create model
    model = KerasHubModel.from_preset(
        config["model_handle"],
        precision=config["precision"],
        lora_rank=config["lora_rank"],
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

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

    print("Finished training. Prompting model...")
    
    # Prepare test prompt
    test_prompt = alpaca_prompt.format(
        instruction="Continue the fibonnaci sequence.",
        input="1, 1, 2, 3, 5, 8",
    )

    # Generate response
    pred = model.generate(
        test_prompt,
        max_length=500,
        tokenizer_handle=config["tokenizer_handle"],
        return_decoded=True
    )
    print("Generated response:", pred)
    
    if model_output_dir is not None:
        model.save_in_hf_format(
            model_output_dir,
            only_save_adapters=True, # You can also save the base model, or merge the base model with the adapters
        )


if __name__ == "__main__":

    run_workload(model_output_dir="./model_output")
