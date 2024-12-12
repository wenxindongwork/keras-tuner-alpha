"""
Singlehost: python3 debug/maxtext_load_hf_checkpoint.py 
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
from keras_tuner.model import MaxTextModel
import numpy as np
from keras_tuner.preprocessor import PretrainingPreprocessor


# Create Model
model = MaxTextModel.from_preset(
    preset_handle="hf://google/gemma-2-2b",
    seq_len=7,
    per_device_batch_size=1,
    scan_layers=False, 
    weight_dtype="bfloat16",
    activation_dtype="bfloat16"
)

# model = MaxTextModel.from_random(
#     model_name= "gemma2-2b",
#     seq_len=7,
#     per_device_batch_size=1,
#     scan_layers=True, 
#     weight_dtype="bfloat16",
#     activation_dtype="bfloat16"
# )


input = {
    "tokens": np.array([[2, 4521, 2134,0,0,0] for _ in range(4)]),
    "segment_ids": np.array([[1, 1, 1,0,0,0]for _ in range(4)] ),
    "positions": np.array([[0, 1, 2,0,0,0]  for _ in range(4)]),
}

# Create Preprocessor
preprocessor = PretrainingPreprocessor(
    tokenizer_handle="hf://google/gemma-2-2b",
    seq_len=7,
    model_type="maxtext",
)

pred = model.generate(input, max_new_tokens=3)
print("token_ids", pred["token_ids"][0])

print(preprocessor.tokenizer.decode(pred["token_ids"][0]))

# model.save_in_hf_format('/dev/shm/temp/hf/checkpoint/')


# logits, non_trainable_variables = model.stateless_call(
#     model.trainable_variables, model.non_trainable_variables, input
# )

# print("logits", logits[0])