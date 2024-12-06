"""
Singlehost: python3 debug/maxtext_load_hf_checkpoint.py 
"""

import numpy as np
from keras_tuner.model import MaxTextModel
import os
os.environ["KERAS_BACKEND"] = "jax"


# Create Model
model = MaxTextModel.from_preset(
    preset_handle="hf://google/gemma-2-9b",
    seq_len=100,
    per_device_batch_size=1,
)

input = {
    "tokens": np.array([[1, 2, 3, 0, 0, 0] for _ in range(4)]),
    "segment_ids": np.array([[1, 1, 1, 0, 0, 0] for _ in range(4)]),
    "positions": np.array([[0, 1, 2, 3, 4, 5] for _ in range(4)]),
}

logits, non_trainable_variables = model.stateless_call(
    model.trainable_variables, model.non_trainable_variables, input
)

print("logits", logits[0])


# Golden HF logits: 

#  [[[-23.75 7.75 -4.1875 ... -11.5 -6.96875 -23.125]
#    [-30 -15 -17.125 ... -26.875 -24.125 -30]
#    [8.875 18 -3.48438 ... 10.75 10.5625 7]
#    [9.6875 16.375 -5.96875 ... 8.625 8.0625 3.98438]
#    [-3.875 15.25 -13.6875 ... 5.40625 4.40625 -5.5625]
#    [-13.9375 13.8125 -19.25 ... -0.03125 -1.40625 -15.125]]]

# Maxtext logits: 

#    [[ 0.64453125 -0.01721191  0.11279297 ... 0.         0.      0.        ]
#    [ 0.63671875  0.06005859  0.22070312 ...  0.          0.      0.        ]
#    [ 0.50390625  0.00939941  0.08642578 ...  0.          0.      0.        ]
#    [ 0.29296875  0.11083984  0.08056641 ...  0.          0.      0.        ]
#    [ 0.31054688  0.10791016  0.08544922 ...  0.          0.      0.        ]
#    [ 0.29882812  0.11230469  0.08203125 ...  0.          0.      0.        ]]