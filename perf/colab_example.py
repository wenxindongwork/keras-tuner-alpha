import os

os.environ["KERAS_BACKEND"] = "jax"
import keras_hub
import keras
import numpy as np
from keras.src.backend.common.remat import RematScope

from huggingface_hub import login
login(token="", add_to_git_credential=False)

# Enable Flash Attention
keras.config.enable_flash_attention()
# Enable Remat         
with RematScope(mode="full"):
    model = keras_hub.models.GemmaCausalLM.from_preset("hf://google/gemma-2-2b")

batch_size = 1
seq_length = 1024
input_ids = np.ones((batch_size, seq_length), dtype=np.int32)
padding_mask = np.ones((batch_size, seq_length))
model_input =  {
        "token_ids": input_ids,
        "padding_mask": padding_mask,
        }

print("running forward pass...")

logits, _ = model.stateless_call(
    model.trainable_variables,
    model.non_trainable_variables,
    model_input,
)

print("Done.")


# modified 
# /home/wenxindong_google_com/miniconda3/envs/colab_env/lib/python3.11/site-packages/jax/_src/nn/functions.py