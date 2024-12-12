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
    get_maxtext_model_name_from_hf_handle,
)
from keras import ops

from keras.src.backend.common import global_state
from keras_tuner.model.ckpt_compatibility.maxtext.to_huggingface import save_maxtext_model_in_hf_format


class KerasHubModel(Model):
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
    ) -> "KerasHubModel":
        """Load a Keras model from a preset and apply LoRA if specified."""
        set_precision(precision)
        weight_dtype, activation_dtype = cls._weight_and_activation_dtype(precision)

        set_global_sharding_strategy(sharding_strategy)

        model = CausalLM.from_preset(model_handle, preprocessor=None, **kwargs)
        if lora_rank:
            model.backbone.enable_lora(rank=lora_rank)

        return cls(model, sharding_strategy, weight_dtype=weight_dtype, activation_dtype=activation_dtype)
    