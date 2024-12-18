import keras
import jax
import sys
import numpy as np
from typing import Optional, List
from keras_tuner.model.models.maxtext.ckpt_compatibility import (
    save_maxtext_model_in_hf_format,get_maxtext_model_name_from_hf_handle, load_hf_weights_into_maxtext_model
)
from keras_tuner.model.models.maxtext.conversion_utils import (
    MaxTextConversionMixin
)
from keras_tuner.model import Model, set_precision

class MaxTextModel(Model, MaxTextConversionMixin):
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

        sharding_strategy, model = cls.initialize_random_maxtext_model(
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
        sharding_strategy, model = cls.initialize_random_maxtext_model(
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
        
        # Pad batch to be a multiple of fsdp dimension
        mesh = self.sharding_strategy.data_sharding.mesh
        devices_in_data_fsdp = mesh.shape["fsdp"] * mesh.shape["data"]
        remainder = batch_size % devices_in_data_fsdp
        if remainder != 0:
            pad_size = devices_in_data_fsdp - remainder        
            for key in ["tokens", "segment_ids", "positions"]:
                inputs[key] = np.pad(
                    inputs[key],
                    ((0, pad_size), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
    
        def next_token(current_inputs):
            current_inputs = jax.device_put(
                current_inputs, self.sharding_strategy.data_sharding
                )
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
        
        token_ids = tokens[:batch_size, :]
        predicted_token_ids = tokens[:batch_size, num_tokens - generate_steps: num_tokens]
        
        return {
            "token_ids": token_ids,
            "predicted_token_ids": predicted_token_ids,
        }

    def save_in_hf_format(self, output_dir: str, dtype: str = "auto"):
        """Save the model in HuggingFace format.
        
        Args:
            output_dir (str): Directory path where the model should be saved.
            dtype (str, optional): Data type for saved weights. Defaults to "auto".
        """
        save_maxtext_model_in_hf_format(
            self, output_dir, dtype=dtype
        )
