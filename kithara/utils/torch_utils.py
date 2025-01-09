import torch 
import numpy as np 
from jax.experimental import multihost_utils
from typing import Optional

def convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: Optional[str] = None
) -> torch.Tensor:
    expected_dtype = str(weight.dtype) if dtype is None else dtype
    weight = multihost_utils.process_allgather(weight)
    weight = np.array(weight, dtype="float32")
    torch_dtype = getattr(torch, expected_dtype)
    return torch.from_numpy(weight).to(torch_dtype)
