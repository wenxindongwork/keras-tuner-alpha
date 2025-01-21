"""Full parameter finetune a Gemma2 9B model.

HF_HOME=/dev/shm/temp/hf KERAS_HOME=/dev/shm/temp/keras python examples/singlehost/maxtext_inference.py
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
from kithara import Dataloader, PretrainingPreprocessor, Trainer, Checkpointer
from kithara import MaxTextModel
from examples.example_datasets import example_datasets



from typing import Optional, Any, Dict
import jax
import jax.numpy as jnp
from flax import struct
import numpy as np
from maxtext.MaxText.max_utils import create_device_mesh, get_kv_cache_annotations
import maxtext.MaxText.common_types

# @struct.dataclass
# class InferenceState:
#     """Holds the state needed for model inference"""
#     params: Any
#     cache: Dict
#     next_pos: jnp.ndarray
#     generated_tokens: jnp.ndarray
#     tokens: jnp.ndarray

# class MaxTextInferenceRunner:
#     """Adapter class to run MaxText inference using MaxEngine-style approach"""
    
#     def __init__(self, model, config, mesh=None):
#         """Initialize the inference runner
        
#         Args:
#             model: The MaxText model instance
#             config: Model configuration 
#             mesh: Optional JAX mesh for model parallelism
#         """
#         self.model = model
#         self.config = config
        
#         # Set up mesh if not provided
#         if mesh is None:
#             devices_array = create_device_mesh(config=config)
#             self.mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
#         else:
#             self.mesh = mesh
            
#         # Get cache annotations for KV cache management
#         self.kv_cache_annotations = get_kv_cache_annotations(
#             self.model, 
#             self.config,
#             jax.random.PRNGKey(0),
#             self.mesh
#         )
        
#         # Create sharding specs
#         self.kv_cache_shardings = jax.tree_util.tree_map(
#             lambda x: jax.sharding.NamedSharding(self.mesh, x),
#             self.kv_cache_annotations
#         )
        
#         self.replicated_sharding = jax.sharding.NamedSharding(
#             self.mesh, 
#             jax.sharding.PartitionSpec(None)
#         )

#     @jax.jit
#     def prefill(self, params: Any, input_tokens: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> InferenceState:
#         """Run initial prefill pass to setup KV cache
        
#         Args:
#             params: Model parameters
#             input_tokens: Input token IDs [batch_size, seq_len] 
#             rng: Optional PRNG key
            
#         Returns:
#             InferenceState containing KV cache and generation state
#         """
#         if rng is None:
#             rng = jax.random.PRNGKey(0)
            
#         batch_size = input_tokens.shape[0]
#         seq_len = input_tokens.shape[1]
        
#         # Create position IDs and segment mask
#         positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
#         segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
#         with self.mesh:
#             # Run forward pass to get logits and setup cache
#             logits, vars = self.model.apply(
#                 params,
#                 input_tokens,
#                 positions,
#                 decoder_segment_ids=segment_ids,
#                 enable_dropout=False,
#                 model_mode=common_types.MODEL_MODE_PREFILL,
#                 rngs={"params": rng},
#                 mutable=["cache"]
#             )
            
#             # Get final token logits
#             final_logits = jax.lax.dynamic_slice(
#                 logits,
#                 (0, seq_len - 1, 0),
#                 (logits.shape[0], 1, logits.shape[2])
#             )
            
#             # Sample next token
#             next_token = self._sample_token(final_logits, rng)
            
#             return InferenceState(
#                 params=params,
#                 cache=vars["cache"],
#                 next_pos=jnp.full((batch_size, 1), seq_len, dtype=jnp.int32),
#                 generated_tokens=jnp.zeros((batch_size, 1), dtype=jnp.int32),
#                 tokens=next_token
#             )
            
#     @jax.jit
#     def generate_step(self, state: InferenceState, rng: Optional[jax.random.PRNGKey] = None) -> InferenceState:
#         """Generate one token autoregressively
        
#         Args:
#             state: Current inference state
#             rng: Optional PRNG key
            
#         Returns:
#             Updated inference state with new token
#         """
#         if rng is None:
#             rng = jax.random.PRNGKey(0)
            
#         with self.mesh:
#             # Run one step of generation
#             logits, new_vars = self.model.apply(
#                 state.params | {"cache": state.cache},
#                 state.tokens,
#                 state.next_pos,
#                 enable_dropout=False, 
#                 model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
#                 rngs={"params": rng},
#                 mutable=["cache"]
#             )
            
#             # Sample next token
#             next_token = self._sample_token(logits, rng)
            
#             # Update state
#             return InferenceState(
#                 params=state.params,
#                 cache=new_vars["cache"],
#                 next_pos=state.next_pos + 1,
#                 generated_tokens=state.generated_tokens + 1,
#                 tokens=next_token
#             )
            
#     def _sample_token(self, logits: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
#         """Sample next token from logits"""
#         # Basic greedy sampling for now
#         return jnp.argmax(logits, axis=-1, keepdims=True)
        
#     def generate(self, 
#                 params: Any,
#                 input_tokens: jnp.ndarray,
#                 max_new_tokens: int,
#                 rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
#         """Generate tokens from input prompt
        
#         Args:
#             params: Model parameters
#             input_tokens: Input token IDs [batch_size, seq_len]
#             max_new_tokens: Maximum number of tokens to generate
#             rng: Optional PRNG key
            
#         Returns:
#             Generated token IDs [batch_size, seq_len + max_new_tokens]
#         """
#         if rng is None:
#             rng = jax.random.PRNGKey(0)
            
#         # Initial prefill pass
#         rng, subrng = jax.random.split(rng)
#         state = self.prefill(params, input_tokens, subrng)
        
#         generated = [state.tokens]
        
#         # Generate tokens autoregressively
#         for _ in range(max_new_tokens - 1):
#             rng, subrng = jax.random.split(rng)
#             state = self.generate_step(state, subrng)
#             generated.append(state.tokens)
            
#         # Concatenate all tokens
#         return jnp.concatenate([input_tokens] + generated, axis=1)

config = {
    "model_handle": "hf://google/gemma-2-9b",
    "tokenizer_handle": "hf://google/gemma-2-9b",
    "seq_len": 4096,
    "precision": "mixed_bfloat16",
    "training_steps": 200,
    "eval_steps_interval": 100,
    "log_steps_interval": 1,
    "per_device_batch_size": 1,
    "max_eval_samples": 50,
    "model_output_dir": "gs://bucket_name/ckpt/",
    "learning_rate": 5e-5
}

model = MaxTextModel.from_preset(
    preset_handle=config["model_handle"],
    seq_len=config["seq_len"],
    per_device_batch_size=config["per_device_batch_size"],
    precision=config["precision"],
    scan_layers=True
)

model.generate("hello")