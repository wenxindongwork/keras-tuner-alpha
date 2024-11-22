import jax
import jax.numpy as jnp
import numpy as np
import keras
import functools
from typing import Optional, Any
from abc import ABC, abstractmethod
from keras_nlp.models import CausalLM
from keras.src.backend.common import global_state
from keras_tuner.model.sharding import set_global_sharding_strategy, ShardingStrategy, FSDPShardingStrategy
from keras_tuner.model.sharding.utils import (
    print_elements_that_are_unsharded_and_large_in_pytree,
)
from keras.src.backend.common import global_state


class Model(ABC):
    """
    Abstract base class for machine learning models.

    Provides a common interface for model checkpoint conversion management.

    Attributes:
        precision (Optional[str]): The precision of activations. Can be any dtype name, such as 'float32' or 'float64',
            which causes both the compute and variable dtypes will be that dtype. Can also be the string 'mixed_float16' or
            'mixed_bfloat16', which causes the compute dtype to be float16 or bfloat16 and the variable dtype to be float32.
        _model (keras.Model): The underlying Keras model instance.
    """

    def __init__(self, precision: Optional[str] = None):
        self._precision = precision
        self._set_precision()
        self._model = self._create_model()

    @abstractmethod
    def _create_model(self) -> keras.Model:
        pass

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
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)

    def _set_precision(self):
        if self._precision is not None:
            policy = global_state.get_global_attribute("dtype_policy", None)
            if policy is not None:
                print(f"Overriding exiting policy: {policy}")
            keras.mixed_precision.set_global_policy(self._precision)


class ModelValidationMixin:
    """Mixin providing common validation functionality."""

    def validate_model(self, model: Any) -> None:
        """Validate model initialization and sharding."""
        if model is None:
            raise ValueError("Model has not been successfully created.")
        self._validate_sharding(model)

    def _validate_sharding(self, model: Any) -> None:
        """Check for unsharded large elements in the model."""
        print_elements_that_are_unsharded_and_large_in_pytree(model)


class KerasModel(Model, ModelValidationMixin):
    """
    A class for KerasNLP models with optional LoRA support.

    Implements the Model interface for a Keras-based causal language model,
    with support for loading pretrained models and optional LoRA fine-tuning.

    Attributes:
        model_handle (str): The model identifier (e.g., "hf://google/gemma-2-2b").
        lora_rank (Optional[int]): The rank for LoRA adaptation. If None, LoRA is disabled.
        _model (keras.Model): The underlying Keras model instance.
    Usage:
        model = KerasModel("hf://google/gemma-2-2b", lora_rank = 4)
        model.generate("what is your name?")
        sftTrainer = Trainer(model=model, ... )
    """

    def __init__(
        self,
        model_handle: str,
        lora_rank: Optional[int] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        **kwargs,
    ):
        self.model_handle = model_handle
        self.lora_rank = lora_rank
        self.sharding_strategy = sharding_strategy
        super(KerasModel, self).__init__(**kwargs)

    def _create_model(self):

        if self.sharding_strategy is not None:
            # Define sharding strategy
            set_global_sharding_strategy(self.sharding_strategy)

        if global_state.get_global_attribute("distribution") is None:
            print("WARNING: Model is not sharded. This could cause HBM OOM.")

        model = CausalLM.from_preset(self.model_handle, preprocessor=None)
        if self.lora_rank:
            model.backbone.enable_lora(rank=4)

        self.validate_model(model)
        return model


class MaxTextModel(Model, ModelValidationMixin):
    """
    A wrapper class for MaxText models that provides a Keras-compatible interface.

    Creates and configures a MaxText model with appropriate sharding strategy and
    converts it to a Keras-compatible format.

    Attributes:
        model_name (str): Name of the MaxText model configuration.
        seq_len (int): Maximum sequence length for the model.
        global_batch_size (int): Batch size across all devices.
        _model (keras.Model): The converted MaxText model in Keras format.

    Note:
        Available model names can be found in:
        https://github.com/AI-Hypercomputer/maxtext/tree/main/MaxText/configs/models

    Usage:
        model = MaxTextModel("gemma2-2b", seq_len = 8192, global_batch_size = 16)
        sftTrainer = Trainer(model=model, ... )
    """

    model_name: str
    seq_len: int
    per_device_batch_size: int

    def __init__(
        self, model_name: str, seq_len: int, per_device_batch_size: int, **kwargs
    ):
        self.model_name = model_name
        self.seq_len = seq_len
        self.per_device_batch_size = per_device_batch_size
        super(MaxTextModel, self).__init__(**kwargs)

    def _create_model(self):
        from keras_tuner.model.converter.maxtext import (
            convert_maxtext_model_to_keras_model,
            get_maxtext_config,
        )
        from keras_tuner.sharding.maxtext import MaxTextSharding
        from keras_tuner.model.sharding.maxtext import MaxTextSharding
        from maxtext.MaxText.train import setup_mesh_and_model
        from maxtext.MaxText.max_utils import (
            get_abstract_state,
            unbox_logicallypartioned,
        )

        maxtext_config = get_maxtext_config(self.model_name)
        global_batch_size = self.per_device_batch_size * jax.device_count()

        (
            init_rng,
            _,
            _,
            jax_mesh,
            model,
            _,
            tx,
        ) = setup_mesh_and_model(maxtext_config)

        def init_initial_state(model, rng):
            """
            Initialize model parameters with model.init and rng
            """
            input_shape = (
                global_batch_size,
                self.seq_len,
            )
            model_vars = model.init(
                {"params": rng, "dropout": rng, "aqt": rng},
                np.ones(input_shape, dtype=jnp.int32),
                np.ones(input_shape, dtype=jnp.int32),
            )
            return model_vars

        # Make model a static input to jax.jit
        init_state_partial = functools.partial(init_initial_state, model)

        _, _, state_shardings = get_abstract_state(
            model, tx, maxtext_config, init_rng, jax_mesh, is_training=True
        )

        state = jax.jit(
            init_state_partial,
            in_shardings=None,
            out_shardings=state_shardings.params,
        )(init_rng)

        state = unbox_logicallypartioned(state)

        set_global_sharding_strategy(MaxTextSharding(jax_mesh, state_shardings))

        model = convert_maxtext_model_to_keras_model(
            model, state, self.seq_len, global_batch_size
        )
        self.validate_model(model)
        return model
