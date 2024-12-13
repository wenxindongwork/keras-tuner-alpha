"""Full parameter finetune a Gemma2 9B MaxText model.

This script demonstrates how to:
1. Set up a Gemma2 model for full parameter finetuning
2. Load HuggingFace Gemma2 checkpoint
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices

This script can be run on both single and multihost. 

Singlehost: python3 examples/singlehost/maxtext_example.py 
Multihost:  python orchestration/multihost/ray/submit_ray_job.py "python3 examples/multihost/ray/TPU/maxtext_example_via_ray.py" --hf-token <TOKEN>

If you experience OOM error during model checkpoint loading, it is because your host VM does not have enough 
capacity to load the model. Consider mounting extra memory onto your VM, and launch this script with 
`HF_HOME=new_hf_cache_dir python3 examples/singlehost/maxtext_example.py`
 
E.g. `HF_HOME=/dev/shm/temp/hf python3 examples/singlehost/maxtext_example.py`
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
import jax 
from typing import Optional
from keras_tuner import Dataloader, PretrainingPreprocessor, Trainer
from keras_tuner.model.models.maxtext.maxtext_model import MaxTextModel
from examples.example_datasets import example_datasets

# Cache JAX compilation to speed up future runs DO_NOT_SUBMIT
jax.config.update("jax_compilation_cache_dir", "/dev/shm/temp/xla_cache")

config = {
    "hf_handle": "hf://google/gemma-2-9b",
    "seq_len": 100,
    "precision": "mixed_bfloat16",
    "training_steps": 100,
    "eval_steps_interval": 10,
    "log_steps_interval": 1,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
    "model_output_dir": '/dev/shm/temp/hf/checkpoint/'
}

def run_workload(
    train_dataset: ray.data.Dataset,
    eval_dataset:ray.data.Dataset,
    dataset_is_sharded_per_host: bool,
):

    # Create Model
    model = MaxTextModel.from_random(
        model_name="gemma2-9b",
        # preset_handle=config["hf_handle"],
        seq_len=config["seq_len"],
        per_device_batch_size=config["per_device_batch_size"],
        precision=config["precision"],
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        # weight_decay=0.01,
    )

    # Create Preprocessor
    preprocessor = PretrainingPreprocessor(
        tokenizer_handle=config["hf_handle"],
        seq_len=config["seq_len"],
        model_type="maxtext",
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

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        preprocessor=preprocessor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=config["training_steps"],
        eval_steps_interval=config["eval_steps_interval"],
        log_steps_interval=config["log_steps_interval"],
        max_eval_samples=config["max_eval_samples"]
    )
        
    # Start training
    trainer.train()

    pred = trainer.generate(["What is your name?"]*4)
    print(f"Tuned model generated {pred}")

    model.save_in_hf_format(config["model_output_dir"])

if __name__ == "__main__":
    train_ds, eval_ds = example_datasets("finetune_toy")
    run_workload(
        train_ds,
        eval_ds,
        dataset_is_sharded_per_host=False,
    )
