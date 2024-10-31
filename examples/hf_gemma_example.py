from keras_tuner.trainer import Trainer
import keras
from datasets import load_dataset
import keras_nlp
from transformers import AutoTokenizer
from keras_tuner.trainer.preprocessing import ContinuedPretrainingPreprocessor
from keras_tuner.trainer.sharding import GemmaFDSP
from tensorflow import data as tf_data
import tensorflow_datasets as tfds

import jax
import types

def run_workload():
    # Log TPU device information
    devices = keras.distribution.list_devices()
    num_devices = len(devices)
    print(f"Available devices: {devices}")

    # Use bf16 training
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    # Define workload parameters.
    seq_len = 4096
    per_device_batch_size = 1
    global_batch_size = per_device_batch_size * num_devices
    
    # Option 1: Load HF dataset
    # hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # hf_dataset = hf_dataset.batch(batch_size=global_batch_size)

    # Option 2: Define toy dataset
    dataset_dict = {
        "text": [
            "What is your name? My name is Mary",
        ]
        * global_batch_size
    }
    train_dataset = tf_data.Dataset.from_tensor_slices(dataset_dict).batch(
        global_batch_size
    )
    train_dataset = tfds.as_numpy(train_dataset)

    # Define sharding strategy
    sharding_strategy = GemmaFDSP()
    keras.distribution.set_distribution(sharding_strategy.distribution)

    # Load model
    model_handle = "google/gemma-2-2b"
    model = keras_nlp.models.CausalLM.from_preset(
        f"hf://{model_handle}", preprocessor=None
    )

    # Optional. Apply Lora to QKV projections
    model.backbone.enable_lora(rank=4)

    # Creates preprocessor
    tokenizer = AutoTokenizer.from_pretrained(model_handle, pad_token="<pad>")
    preprocessor = ContinuedPretrainingPreprocessor(
        tokenizer=tokenizer, seq_len=seq_len, input_field="text"
    )

    # Create optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        preprocessor=preprocessor,
        train_dataset=train_dataset,
        sharding_strategy=sharding_strategy,
        steps=50,
        log_steps=1,
    )

    # Start training
    trainer.train()

    # Test after tuning
    pred = trainer.generate("What is your name?")
    print("Tuned model generates:", pred)


if __name__ == "__main__":
    run_workload()
