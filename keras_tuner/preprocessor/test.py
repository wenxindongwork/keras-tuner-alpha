import numpy as np
from jax.tree_util import tree_map

batch = [
    {
        "x": {
            "token_ids": np.array([[1, 2, 3]]),
            "padding_mask": np.array([[1, 1, 0]]),
        },
        "y": np.array([[0]]),
    },
    {
        "x": {
            "token_ids": np.array([[4, 5, 6]]),
            "padding_mask": np.array([[1, 0, 0]]),
        },
        "y": np.array([[1]]),
    },
]

# Both functions will produce equivalent results
batched_data_1 = tree_map(lambda *arrays: np.concatenate(arrays), *batch)
print(batched_data_1)

expected_output = {
    "x": {
        "padding_mask": np.array([[1, 1, 0], [1, 0, 0]]),
        "token_ids": np.array([[1, 2, 3], [4, 5, 6]]),
    },
    "y": np.array([[0], [1]]),
}

