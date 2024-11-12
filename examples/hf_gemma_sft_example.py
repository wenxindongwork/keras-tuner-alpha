from keras_tuner.trainer import Trainer
import keras
from datasets import load_dataset
import keras_nlp
from transformers import AutoTokenizer
from keras_tuner.preprocessor import SFTPreprocessor
from keras_tuner.sharding import (
    PredefinedShardingStrategy,
    set_global_sharding_strategy,
)
from tensorflow import data as tf_data
import tensorflow_datasets as tfds

"""This script runs LoRA supervised finetuning on Gemma2-2b."""

def run_workload():
    # Log TPU device information
    devices = keras.distribution.list_devices()
    num_devices = len(devices)
    print(f"Available devices: {devices}")

    # Use bf16 training
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    # Define workload parameters.
    seq_len = 30
    per_device_batch_size = 1
    global_batch_size = per_device_batch_size * num_devices

    # Define toy dataset
    train_dataset = {
        "prompt": ["What is your name?"] * global_batch_size,
        "answer": ["My name is Mary"] * global_batch_size,
    }
    train_dataset = tf_data.Dataset.from_tensor_slices(train_dataset).batch(
        global_batch_size
    )
    train_dataset = tfds.as_numpy(train_dataset)

    # Define global sharding strategy
    set_global_sharding_strategy(
        PredefinedShardingStrategy(parallelism="fsdp", model="gemma")
    )

    # Load model
    model_handle = "google/gemma-2-2b"
    model = keras_nlp.models.CausalLM.from_preset(
        f"hf://{model_handle}", preprocessor=None
    )

    # Optional. Apply Lora to QKV projections
    model.backbone.enable_lora(rank=4)

    # Creates preprocessor
    tokenizer = AutoTokenizer.from_pretrained(model_handle, pad_token="<pad>")
    preprocessor = SFTPreprocessor(tokenizer=tokenizer, seq_len=seq_len)

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        preprocessor=preprocessor,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        eval_steps=10,
        steps=100,
        log_steps=10,
    )

    # Start training
    trainer.train()

    # Test after tuning
    pred = trainer.generate("What is your name?")
    print("Tuned model generates:", pred)


if __name__ == "__main__":
    run_workload()
