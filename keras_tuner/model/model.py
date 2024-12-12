import keras
import jax
import sys
import jax.numpy as jnp
import numpy as np
import functools
from typing import Optional, Any, List, Union
from abc import ABC, abstractmethod
from keras_nlp.models import CausalLM
from keras.src.backend.common import global_state
from keras.distribution import set_distribution
from keras_tuner.model.sharding import ShardingStrategy
from keras_tuner.model.sharding.utils import (
    print_elements_that_are_unsharded_and_large_in_pytree,
)
from keras_tuner.model.ckpt_compatibility.maxtext.from_huggingface import (
    load_hf_weights_into_maxtext_model,
)
from keras_tuner.model.ckpt_compatibility.maxtext.utils import (
    get_maxtext_model_type_from_hf_handle,
)
from keras import ops

from keras.src.backend.common import global_state
from keras_tuner.model.ckpt_compatibility.maxtext.to_huggingface import save_maxtext_model_in_hf_format


class ModelValidationMixin:
    """Mixin providing common model validation functionality."""

    def validate_model(self, model: Any) -> None:
        """Validates the model by ensuring it is not None and checking sharding status."""
        if model is None:
            raise ValueError("Model has not been successfully created.")
        print_elements_that_are_unsharded_and_large_in_pytree(model)


class Model(ABC, ModelValidationMixin):
    """
    Base class for a Kithara model. Provides common attributes and methods for model management.

    Attributes:
        precision (Optional[str]): Precision of activations, e.g., 'float32', 'mixed_float16'.
        model (keras.Model): The Keras model instance.
    """

    def __init__(self, model, sharding_strategy, model_name = None, 
                 weight_dtype: Optional[str] = None, 
                 activation_dtype: Optional[str] = None,
                 scan_layers=False):
        
        self.sharding_strategy = sharding_strategy
        self.model = model
        self.scan_layers=scan_layers
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype

    def __getattr__(self, name):
        """
        Delegates any unknown attributes/methods to the underlying _model.
        This allows direct access to all of _model's methods without explicit delegation.
        """
        try:
            # Try to get the attribute from the Model class first
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to _model
            model = object.__getattribute__(self, "model")
            return getattr(model, name, None)



class KerasModel(Model):
    """
    A Kithara model wrapper for KerasHub models, providing a Keras-compatible interface.

    Attributes:
        model_handle (str): Model identifier, e.g., "hf://google/gemma-2-2b".
        lora_rank (Optional[int]): Rank for LoRA adaptation (disabled if None).
    """

    @classmethod
    def from_preset(
        cls,
        model_handle: str,
        lora_rank: Optional[int] = None,
        precision: Optional[str] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        **kwargs,
    ) -> "KerasModel":
        """Load a Keras model from a preset and apply LoRA if specified."""
        set_precision(precision)
        weight_dtype, activation_dtype = cls._weight_and_activation_dtype(precision)

        set_global_sharding_strategy(sharding_strategy)

        model = CausalLM.from_preset(model_handle, preprocessor=None, **kwargs)
        if lora_rank:
            model.backbone.enable_lora(rank=lora_rank)

        return cls(model, sharding_strategy, weight_dtype=weight_dtype, activation_dtype=activation_dtype)
    
class MaxTextModel(Model):
    """
    A Kithara wrapper for MaxText models, providing a Keras-compatible interface.

    Attributes:
        model_name (str): MaxText model name.
        seq_len (int): Maximum sequence length.
        global_batch_size (int): Batch size across all devices.
    """

    @classmethod
    def from_random(
        cls,
        model_name: Optional[str] = None,
        seq_len: Optional[int] = None,
        per_device_batch_size: Optional[int] = None,
        weight_dtype: str = "float32",
        activation_dtype: str = "bfloat16",
        scan_layers:bool = False,
        maxtext_config_args: Optional[dict] = None
    ) -> "MaxTextModel":
        """Create a randomly initialized MaxText model with the given configuration."""
        set_precision(weight_dtype=weight_dtype, activation_dtype=activation_dtype)

        if maxtext_config_args == None: 
            maxtext_config_args = {}
        assert "weight_dtype" not in maxtext_config_args
        maxtext_config_args["weight_dtype"] = weight_dtype
        assert "dtype" not in maxtext_config_args
        maxtext_config_args["dtype"] = activation_dtype
        assert "scan_layers" not in maxtext_config_args
        maxtext_config_args["scan_layers"] = scan_layers
        
        maxtext_config_args = " ".join([f"{key}={value}" for key, value in maxtext_config_args.items()])

        sharding_strategy, model = cls._initialize_random_model(
            model_name, seq_len, per_device_batch_size, maxtext_config_args
        )
        return cls(model, sharding_strategy, model_name = model_name, scan_layers=scan_layers, weight_dtype=weight_dtype, activation_dtype=activation_dtype)

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        seq_len: int,
        per_device_batch_size: int,
        weight_dtype: str = "float32",
        activation_dtype: str = "bfloat16",
        scan_layers:bool = False,
        maxtext_config_args: Optional[dict] = None,
        **kwargs,
    ) -> "MaxTextModel":
        """Create a MaxText model initialized with weights from HuggingFace Hub."""        
        model_name = get_maxtext_model_type_from_hf_handle(preset_handle)
        set_precision(weight_dtype=weight_dtype, activation_dtype=activation_dtype)
        if maxtext_config_args == None: 
            maxtext_config_args = {}
        
        assert "weight_dtype" not in maxtext_config_args
        maxtext_config_args["weight_dtype"] = weight_dtype
        assert "dtype" not in maxtext_config_args
        maxtext_config_args["dtype"] = activation_dtype
        assert "scan_layers" not in maxtext_config_args
        maxtext_config_args["scan_layers"] = scan_layers
        
        maxtext_config_args = " ".join([f"{key}={value}" for key, value in maxtext_config_args.items()])
        sharding_strategy, model = cls._initialize_random_model(
            model_name, seq_len, per_device_batch_size, maxtext_config_args
        )
        model = load_hf_weights_into_maxtext_model(preset_handle, model, scan_layers)

        return cls(model, sharding_strategy, model_name = model_name, weight_dtype = weight_dtype, activation_dtype = activation_dtype, scan_layers=scan_layers)

    @staticmethod
    def _initialize_random_model(
        model_name: str,
        seq_len: int,
        per_device_batch_size: int,
        maxtext_config_args: Optional[str] = None
    ) -> tuple[ShardingStrategy, keras.Model]:
        """Initialize a random MaxText model with sharding configuration."""
        print("-> Initializing a MaxText {model_name} model...")

        from keras_tuner.model.converter.maxtext import (
            convert_maxtext_model_to_keras_model,
            get_maxtext_config,
        )
        from keras_tuner.model.sharding.maxtext import MaxTextSharding
        from maxtext.MaxText.train import setup_mesh_and_model
        from maxtext.MaxText.max_utils import (
            get_abstract_state,
            unbox_logicallypartioned,
        )

        maxtext_config = get_maxtext_config(
            model_name, maxtext_config_args)
        global_batch_size = per_device_batch_size * jax.device_count()

        # Initialize the model and mesh configuration
        init_rng, _, _, jax_mesh, model, _, tx = setup_mesh_and_model(
            maxtext_config)

        # Initialize model parameters
        def init_initial_state(model, rng):
            input_shape = (global_batch_size, seq_len)
            return model.init(
                {"params": rng, "dropout": rng, "aqt": rng},
                np.ones(input_shape, dtype=jnp.int32),
                np.ones(input_shape, dtype=jnp.int32),
            )

        init_state_partial = functools.partial(init_initial_state, model)

        # Get the model state and shardings
        _, _, state_shardings = get_abstract_state(
            model, tx, maxtext_config, init_rng, jax_mesh, is_training=True
        )
        state = jax.jit(
            init_state_partial, in_shardings=None, out_shardings=state_shardings.params
        )(init_rng)
        state = unbox_logicallypartioned(state)

        sharding_strategy = MaxTextSharding(jax_mesh, state_shardings, maxtext_config)

        set_global_sharding_strategy(sharding_strategy)

        # Convert to Keras model format
        model = convert_maxtext_model_to_keras_model(
            model, state, seq_len, global_batch_size, jax_mesh, maxtext_config
        )
        # Delete state
        def delete_array(x):
            if isinstance(x, jnp.ndarray):
                x.delete()
        jax.tree_util.tree_map(delete_array, state)
        print("✅ Successfully initialized a MaxText {model_name} model...")
        return sharding_strategy, model

    def make_generate_step(self):
        def fn(trainable_variables, non_trainable_variables, x):
            logits, non_trainable_variables = self.model.stateless_call(
                trainable_variables, non_trainable_variables, x
            )
            return logits
        return jax.jit(fn)

    def generate(
        self,
        inputs,
        max_new_tokens=sys.maxsize,
        stop_token_ids: List[int]=None,
    ):
        if stop_token_ids is None:
            stop_token_ids = []

        jitted_generate_fn = self.make_generate_step()
        batch_size = inputs["tokens"].shape[0]

        def next_token(current_inputs):
            logits = jitted_generate_fn(
                [v.value for v in self.model.trainable_variables],
                [v.value for v in self.model.non_trainable_variables],
                current_inputs
            )
            return logits

        tokens = inputs["tokens"]
        segment_ids = inputs["segment_ids"]
        positions = inputs["positions"]

        # Calculate initial number of tokens (where segment_ids == 1)
        num_tokens = int(np.sum(segment_ids[0] == 1))
        seq_len = segment_ids.shape[1]

        # Calculate how many tokens we can/should generate
        generate_steps = min(seq_len - num_tokens, max_new_tokens)

        # Track which sequences have reached EOS
        reached_eos = [False for _ in range(batch_size)]

        for _ in range(generate_steps):
            current_inputs = {
                "tokens": tokens,
                "segment_ids": segment_ids,
                "positions": positions
            }

            # Get next token predictions
            logits = next_token(current_inputs)
            next_token_logits = logits[:, num_tokens - 1, :]
            next_tokens = keras.ops.argmax(next_token_logits, axis=-1)
            
            # Update the tokens array with predictions
            tokens[:, num_tokens] = next_tokens
            
            # Update attention mask (segment_ids)
            segment_ids = np.roll(segment_ids, 1, axis=1)
            segment_ids[:, 0] = 1
            
            # Increment number of tokens
            num_tokens += 1
            
            # Check for EOS tokens
            for i, token in enumerate(next_tokens):
                if token in stop_token_ids:
                    reached_eos[i] = True
                    
            if all(reached_eos):
                break

        return {
            "token_ids": tokens,
            "predicted_token_ids": tokens[:, tokens.shape[1] - generate_steps:]
        }
    
    def save_in_hf_format(self, output_dir: str, dtype: str = "auto"):
        save_maxtext_model_in_hf_format(self.model_name, self, output_dir, scan_layers=self.scan_layers, dtype= dtype)
        


def set_precision(precision: Optional[str]=None, weight_dtype = Optional[str], activation_dtype= Optional[str]) -> None:
    """Set global mixed-precision policy for model weights and activations."""
    assert (precision is None and (weight_dtype is not None) and (activation_dtype is not None)) or ((precision is not None) and (weight_dtype is None) and (activation_dtype is None)), "Please only specify either weight and activation dtype, or precision, but not both."
    
    if precision == None: 
        if weight_dtype == activation_dtype:
            precision  = weight_dtype
        elif weight_dtype=="float32" and activation_dtype == "float16":
            precision = "mixed_float16"
        elif weight_dtype=="float32" and activation_dtype == "bfloat16":
            precision = "mixed_bfloat16"
        else:
            raise ValueError("Weight dtype and activation dtype combination is not valid.")
    
    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy:
        print(f"Overriding existing policy: {policy}")
    keras.mixed_precision.set_global_policy(precision)


def set_global_sharding_strategy(strategy: Optional[ShardingStrategy]) -> None:
    """Set the global sharding strategy for model and data distribution."""
    if strategy:
        if global_state.get_global_attribute("distribution") is not None:
            print("WARNING: Distribution strategy is being overridden.")
        set_distribution(strategy.distribution)
        global_state.set_global_attribute(
            "DATA_SHARDING", strategy.data_sharding)
