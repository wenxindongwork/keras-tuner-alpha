# This file exists as maxtext/MaxText/train_using_keras.py
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.src.utils.jax_layer import FlaxLayer
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras_tuner.trainer.fsdp_trainer import FSDPTrainer
from keras_tuner.trainer.preprocessing import MaxTextDataPreparationStrategy
from transformers import AutoTokenizer
from maxtext.MaxText import max_utils
from maxtext.MaxText.layers.models import Transformer
from maxtext.MaxText.layers import quantizations
from jax.sharding import Mesh
import pyconfig
from tensorflow import data as tf_data
import tensorflow_datasets as tfds
import jax

seq_len = 128
per_device_batch_size = 1
global_batch_size = per_device_batch_size * len(jax.devices())


def convert_maxtext_model_to_keras_model(maxtext_model):
    def maxtext_wrapper(module, inputs, training):
        tokens, positions, segment_ids = inputs
        model_mode = "train" if training else "autoregressive"
        segment_ids = segment_ids if training else None
        return module(
            tokens,
            positions,
            segment_ids,
            enable_dropout=training,
            model_mode=model_mode,
        )

    keras_layer = FlaxLayer(
        module=maxtext_model,
        method=maxtext_wrapper,
    )

    # Build the Keras model
    tokens = Input(
        shape=(seq_len,), batch_size=global_batch_size, dtype="int32", name="tokens"
    )
    positions = Input(
        shape=(seq_len,), batch_size=global_batch_size, dtype="int32", name="positions"
    )
    segment_ids = Input(
        shape=(seq_len,),
        batch_size=global_batch_size,
        dtype="int32",
        name="segment_ids",
    )
    x = keras_layer([tokens, positions, segment_ids], training=True)
    keras_model = Model(inputs=[tokens, positions, segment_ids], outputs=x)

    return keras_model


# Initialize a Maxtext "default" model
argv = [
    "maxtext_default_example.py",
    "configs/base.yml",
    "run_name=running_maxtext_with_keras",
    "num_slices=1",
    f"max_target_length={seq_len}",
    f"per_device_batch_size={per_device_batch_size}",
]
pyconfig.initialize(argv)
config = pyconfig.config
devices_array = max_utils.create_device_mesh(config)
mesh = Mesh(devices_array, config.mesh_axes)
quant = quantizations.configure_quantization(config)

# Model is initiated with random parameters
maxtext_model = Transformer(config, mesh, quant)

# Convert MaxText model into a Keras model
keras_model = convert_maxtext_model_to_keras_model(maxtext_model)

# Create optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.build(keras_model.trainable_variables)

# Sanity check
trainable_variables = keras_model.trainable_variables
non_trainable_variables = keras_model.non_trainable_variables
optimizer_variables = optimizer.variables

print("trainable_variables", trainable_variables)
print("non_trainable_variables", non_trainable_variables)
print("optimizer_variables", optimizer_variables)

# Create simple dataset
dpo_dataset_dict = {
    "text": [
        "What is your name? My name is Mary",
    ]
    * global_batch_size
}
train_dataset = tf_data.Dataset.from_tensor_slices(dpo_dataset_dict).batch(
    global_batch_size
)
train_dataset = tfds.as_numpy(train_dataset)

# Initialize trainer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", pad_token="<pad>")
trainer = FSDPTrainer(
    keras_model,
    train_dataset,
    optimizer,
    tokenizer,
    seq_len=seq_len,
    steps=100,
    log_steps=1,
    input_field="text",
    preprocess_strategy=MaxTextDataPreparationStrategy(),
)

# Start training
trainer.train()
