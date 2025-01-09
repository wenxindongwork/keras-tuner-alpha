import jax.numpy as jnp
import torch
import numpy as np 
from kithara.utils.torch_utils import convert_jax_weight_to_torch

def check_logits_match(logitsA, logitsB, atol=0.01):
        
    # Determine types and convert if needed
    is_A_torch = isinstance(logitsA, torch.Tensor)
    is_B_torch = isinstance(logitsB, torch.Tensor)
    
    # If one is torch and one is jax, convert jax to torch
    if is_A_torch and not is_B_torch:
        logitsB = convert_jax_weight_to_torch(logitsB)
    elif is_B_torch and not is_A_torch:
        logitsA = convert_jax_weight_to_torch(logitsA)
    
    # If both are now torch tensors
    if isinstance(logitsA, torch.Tensor):
        is_close = torch.isclose(logitsB, logitsA, atol=atol)
        if not torch.allclose(logitsA, logitsB, atol=atol):
            mismatch_indices = ~is_close
            print(f"Number of mismatch logits elements in {logitsB.shape}", mismatch_indices.sum().item())
            print(logitsA[mismatch_indices])
            print(logitsB[mismatch_indices])
            raise ValueError(f"Failed to match logits.")
    # If both are still jax arrays
    else:
        if not jnp.allclose(logitsA, logitsB, atol=atol):
            is_close_idx = jnp.isclose(logitsB, logitsA, atol=atol)
            print(f"Number of mismatch logits elements in {logitsB.shape}", len(logitsB[is_close_idx==False]))
            print(logitsA[is_close_idx==False])
            print(logitsB[is_close_idx==False])
            raise ValueError(f"Failed to match logits.")
    
    print("Logits matched")
