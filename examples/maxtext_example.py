import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras_tuner.trainer import Trainer
from keras_tuner.preprocessor import PretrainingPreprocessor
from keras_tuner.converter.maxtext import (
    convert_maxtext_model_to_keras_model,
    get_maxtext_config,
    get_maxtext_model,
)
from transformers import AutoTokenizer
from tensorflow import data as tf_data
import tensorflow_datasets as tfds
from datasets import load_dataset
from keras_tuner.sharding.strategy import set_global_sharding_strategy
from keras_tuner.sharding.maxtext import MaxTextSharding
import jax

"""This scripts runs full-parameter finetuning on a gemma2-2b model."""


def run_workload():

    # Use bf16 training
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    # Initialize a Maxtext model
    maxtext_config = get_maxtext_config("gemma2-2b")
    maxtext_model = get_maxtext_model(maxtext_config)

    # Set sharding strategy before initializing model
    set_global_sharding_strategy(MaxTextSharding(maxtext_config))

    # Define workload parameters
    seq_len = 512
    per_device_batch_size = 1
    global_batch_size = per_device_batch_size * len(jax.devices("tpu"))

    # Convert MaxText model into Keras model
    keras_model = convert_maxtext_model_to_keras_model(
        maxtext_model, seq_len, global_batch_size
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Create toy dataset
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

    # Option2: Load HF dataset
    # hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # train_dataset = hf_dataset.batch(batch_size=global_batch_size)

    # Initiaize data preprocessor
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", pad_token="<pad>")
    preprocessor = PretrainingPreprocessor(
        tokenizer=tokenizer, seq_len=seq_len, model_type="maxtext"
    )

    # Initialize trainer
    trainer = Trainer(
        model=keras_model,
        optimizer=optimizer,
        preprocessor=preprocessor,
        train_dataset=train_dataset,
        steps=10,
        log_steps=1,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    run_workload()
