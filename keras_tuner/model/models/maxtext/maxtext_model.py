import keras
import jax
import sys
import jax.numpy as jnp
import numpy as np
import functools
from typing import Optional, List
from keras_tuner.model.sharding import ShardingStrategy
from keras_tuner.model.models.maxtext.ckpt_compatibility import (
    save_maxtext_model_in_hf_format,get_maxtext_model_name_from_hf_handle, load_hf_weights_into_maxtext_model
)
from keras_tuner import Model
from keras_tuner.model import set_precision, set_global_sharding_strategy


class MaxTextModel(Model):
    """
    MaxTextModel is class that represents a MaxText model via the Kithara.Model interface. It is 
    a thin wrapper around the underlying MaxText model instance. 

    Methods
    -------
    from_random: Create a randomly initialized MaxText model with the given configuration.
    from_preset: Create a MaxText model initialized with weights from HuggingFace Hub.
    generate: Generate text based on the input tokens, with an option to stop at specific token IDs.
    save_in_hf_format: Save the MaxText model in HuggingFace format.
    """
    @classmethod
    def from_random(
        cls,
        model_name: str,
        seq_len: int = 2048,
        per_device_batch_size: int = 1,
        precision: str = "mixed_float16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
    ) -> "MaxTextModel":
        """Create a randomly initialized MaxText model with the given configuration.
        
        Args:
            model_name (str): Name of the MaxText model configuration to use. 
            seq_len (int, optional): Maximum sequence length. Defaults to 2048.
            per_device_batch_size (int, optional): Batch size per device. Defaults to 1.
            precision (str, optional): Precision mode for computations. Defaults to "mixed_float16".
            scan_layers (bool, optional): Whether to use scan layers for memory efficiency. Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments. Defaults to None.
            
        Returns:
            MaxTextModel: A new instance of MaxTextModel with random initialization.
        """        
        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)

        sharding_strategy, model = cls._initialize_random_model(
            model_name, seq_len, per_device_batch_size, weight_dtype, activation_dtype, scan_layers, maxtext_config_args
        )
        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
        )

    @classmethod
    def from_preset(
        cls,
        preset_handle: str,
        seq_len: int,
        per_device_batch_size: int,
        precision: str = "mixed_float16",
        scan_layers: bool = False,
        maxtext_config_args: Optional[dict] = None,
        **kwargs,
    ) -> "MaxTextModel":
        """Create a MaxText model initialized with weights from HuggingFace Hub.
        
        Args:
            preset_handle (str): HuggingFace model identifier.
            seq_len (int): Maximum sequence length.
            per_device_batch_size (int): Batch size per device.
            precision (str, optional): Precision mode for computations. Defaults to "mixed_float16".
            scan_layers (bool, optional): Whether to use scan layers. Defaults to False.
            maxtext_config_args (Optional[dict], optional): Additional configuration arguments. Defaults to None.
            **kwargs: Additional keyword arguments.
            
        Returns:
            MaxTextModel: A new instance of MaxTextModel initialized with pretrained weights.
        """
        
        set_precision(precision)
        weight_dtype = cls._weight_dtype(precision)
        activation_dtype = cls._activation_dtype(precision)
        
        model_name = get_maxtext_model_name_from_hf_handle(preset_handle)
        sharding_strategy, model = cls._initialize_random_model(
            model_name, seq_len, per_device_batch_size, weight_dtype, activation_dtype, scan_layers, maxtext_config_args
        )
        model = load_hf_weights_into_maxtext_model(preset_handle, model, scan_layers)

        return cls(
            model,
            sharding_strategy,
            model_name=model_name,
            precision=precision,
            scan_layers=scan_layers,
        )

    @staticmethod
    def _initialize_random_model(
        model_name: str,
        seq_len: int,
        per_device_batch_size: int,
        weight_dtype: str, 
        activation_dtype:str,
        scan_layers: bool,
        maxtext_config_args: Optional[str] = None,
    ) -> tuple[ShardingStrategy, keras.Model]:
        """Initialize a random MaxText model with the input configuration.
        
        This internal method handles the low-level initialization of the model,
        including mesh setup, parameter initialization, and conversion to Keras format.
        
        Args:
            model_name (str): Name of the model configuration.
            seq_len (int): Maximum sequence length.
            per_device_batch_size (int): Batch size per device.
            weight_dtype (str): Data type for model weights.
            activation_dtype (str): Data type for activations.
            scan_layers (bool): Whether to use scan layers.
            maxtext_config_args (Optional[str], optional): Additional configuration arguments. Defaults to None.
            
        Returns:
            tuple[ShardingStrategy, keras.Model]: Tuple containing sharding strategy and initialized model.
        """

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

        if maxtext_config_args == None:
            maxtext_config_args = {}
        
        assert "weight_dtype" not in maxtext_config_args
        maxtext_config_args["weight_dtype"] = weight_dtype
        assert "dtype" not in maxtext_config_args
        maxtext_config_args["dtype"] = activation_dtype
        assert "scan_layers" not in maxtext_config_args
        maxtext_config_args["scan_layers"] = scan_layers

        maxtext_config_args = " ".join(
            [f"{key}={value}" for key, value in maxtext_config_args.items()]
        )

        maxtext_config = get_maxtext_config(model_name, maxtext_config_args)
        global_batch_size = per_device_batch_size * jax.device_count()

        # Initialize the model and mesh configuration
        init_rng, _, _, jax_mesh, model, _, tx = setup_mesh_and_model(maxtext_config)

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
        """Create a JIT-compiled function for single-step token generation.
        
        Returns:
            function: Compiled function that performs one step of token generation.
        """
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
        stop_token_ids: List[int] = None,
    ):
        """Generate text tokens using the model.
        
        Args:
            inputs (dict): Input dictionary containing 'tokens', 'segment_ids', and 'positions'.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to generation until EOS or max sequence length.
            stop_token_ids (List[int], optional): List of token IDs that stop generation. Defaults to None.
            
        Returns:
            dict: Dictionary containing token IDs with and without prompt tokens.
        """
        if stop_token_ids is None:
            stop_token_ids = []

        jitted_generate_fn = self.make_generate_step()
        batch_size = inputs["tokens"].shape[0]

        def next_token(current_inputs):
            logits = jitted_generate_fn(
                [v.value for v in self.model.trainable_variables],
                [v.value for v in self.model.non_trainable_variables],
                current_inputs,
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
                "positions": positions,
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
            "predicted_token_ids": tokens[:, tokens.shape[1] - generate_steps :],
        }

    def save_in_hf_format(self, output_dir: str, dtype: str = "auto"):
        """Save the model in HuggingFace format.
        
        Args:
            output_dir (str): Directory path where the model should be saved.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
        """
        save_maxtext_model_in_hf_format(
            self.model_name, self, output_dir, scan_layers=self.scan_layers, dtype=dtype
        )
