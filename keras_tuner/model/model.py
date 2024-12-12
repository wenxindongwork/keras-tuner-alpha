import keras
from abc import ABC
from typing import Optional, Any
from keras.src.backend.common import global_state
from keras.distribution import set_distribution
from keras_tuner.model.sharding import ShardingStrategy
from keras_tuner.model.sharding.utils import (
    print_elements_that_are_unsharded_and_large_in_pytree,
)
from keras.src.backend.common import global_state

class ModelValidationMixin:
    """Mixin providing common model validation functionality."""

    def validate_sharding(self, model: Any) -> None:
        if model is None:
            raise ValueError("Model has not been successfully created.")
        print_elements_that_are_unsharded_and_large_in_pytree(model)


class Model(ABC, ModelValidationMixin):
    """
    Base class for all models in Kithara. This class serves as a thin 
    wrapper around the underlying model instance, providing a uniform 
    interface for Kithara workloads. Currently supported underlying model 
    implementations include MaxText and KerasHub models.
    
    Attributes:
        sharding_strategy(kithara.ShardingStrategy): Strategy used for sharding the model.
        model(Keras.Model): The underlying Keras model instance.
        model_name(str, optional): Optional name of the model.
        precision(str, optional): Optional mixed-precision policy for model weights and activations. 
            Default is "mixed_bfloat16". Supported policies include "float32", "float16", "bfloat16", 
            "mixed_float16", and "mixed_bfloat16". Mixed precision policies load model weight in float32 
            and casts activations to the specified dtype.
        scan_layers: Boolean indicating whether to scan layers using jax.lax.scan. Currently only 
            MaxText models support this feature.
    Methods:
        __init__(model, sharding_strategy, model_name=None, weight_dtype=None, activation_dtype=None, scan_layers=False):
            Initializes the Model instance with the given parameters.
        __getattr__(name):
            Delegates any unknown attributes/methods to the underlying model.
    """
    def __init__(
        self,
        model,
        sharding_strategy,
        model_name=None,
        precision: str = "mixed_bfloat16",
        scan_layers=False,
    ):

        self.sharding_strategy = sharding_strategy
        self.model = model
        self.scan_layers = scan_layers
        self.model_name = model_name
        self.precision = precision
        self.weight_dtype = self._weight_dtype(precision)
        self.activation_dtype = self._activation_dtype(precision)

    def __getattr__(self, name):
        try:
            # Try to get the attribute from the Model class first
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to _model
            model = object.__getattribute__(self, "model")
            return getattr(model, name, None)
    
    @staticmethod
    def _weight_dtype(precision: Optional[str] = None) -> str:
        if "mixed" in precision:
            return "float32"
        return precision
    
    @staticmethod
    def _activation_dtype(precision: Optional[str] = None) -> str:
        if "mixed" in precision:
            return precision.split("_")[1]
        return precision

def set_precision(precision: Optional[str] = None, weight_dtype=Optional[str], activation_dtype=Optional[str]) -> None:
    """
    Sets the precision policy for mixed precision training. This function overrides the 
    default precision policy and must be called before loading the model.

    This function allows you to specify either a precision policy directly or 
    to specify the weight and activation data types, from which the precision 
    policy will be inferred. 

    Args:
        precision (Optional[str]): The precision policy to set. Can be one of 
            'float32', 'float16', 'mixed_float16', or 'mixed_bfloat16'. If None, the 
            precision will be inferred from `weight_dtype` and `activation_dtype`.
        weight_dtype (Optional[str]): The data type for weights. Used to infer 
            the precision policy if `precision` is None. Must be one of 
            'float32', 'float16', or 'bfloat16'.
        activation_dtype (Optional[str]): The data type for activations. Used to 
            infer the precision policy if `precision` is None. Must be one of 
            'float32', 'float16', or 'bfloat16'.

    Returns:
        precision (str): The precision policy that was set.
    """
    assert (precision is None and (weight_dtype is not None) and (activation_dtype is not None)) or ((precision is not None) and (
        weight_dtype is None) and (activation_dtype is None)), "Please only specify either weight and activation dtype, or precision, but not both."

    if precision == None:
        if weight_dtype == activation_dtype:
            precision = weight_dtype
        elif weight_dtype == "float32" and activation_dtype == "float16":
            precision = "mixed_float16"
        elif weight_dtype == "float32" and activation_dtype == "bfloat16":
            precision = "mixed_bfloat16"
        else:
            raise ValueError(
                "Weight dtype and activation dtype combination is not valid.")

    policy = global_state.get_global_attribute("dtype_policy", None)
    if policy:
        print(f"Overriding existing policy: {policy}")
    keras.mixed_precision.set_global_policy(precision)
    return precision


def set_global_sharding_strategy(strategy: Optional[ShardingStrategy]) -> None:
    """
    Sets the sharding strategy for the model and batch input.This function 
    overrides the existing sharding strategy and must be called before loading 
    the model.

    Args:
        strategy (Optional[kithara.ShardingStrategy]): The sharding strategy to be set
            globally. If None, no changes are made to the global state.
    """
    if strategy:
        if global_state.get_global_attribute("distribution") is not None:
            print("WARNING: Distribution strategy is being overridden.")
        set_distribution(strategy.distribution)
        global_state.set_global_attribute(
            "DATA_SHARDING", strategy.data_sharding)
