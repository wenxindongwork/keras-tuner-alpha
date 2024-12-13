import os

os.environ["KERAS_BACKEND"] = "jax"

from typing import Any, Dict, List, Union, Optional
import flax
from functools import lru_cache
import jax
from maxtext.MaxText import pyconfig
from keras_tuner.utils.tree_utils import named_tree_map
from keras.src.utils import tracking
from keras.src import backend
import jax.numpy as jnp
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.src.utils.jax_layer import FlaxLayer


class MaxTextLayer(FlaxLayer):

    @tracking.no_automatic_dependency_tracking
    def _create_variables(self, values: Dict[str, Any], trainable: bool) -> List[Any]:
        """Overrite FlaxLayer's _create_variables so that a KerasVariable is created
        with the name argument set to be the path prefix of it's parents in the pytree.

        Args:
            values: Dictionary containing the variable values
            trainable: Whether the variables should be trainable

        Returns:
            List of created variables
        """

        def create_variable(name: List[str], value: Any) -> Any:
            name_str = "-".join(name)
            if backend.is_tensor(value) or isinstance(value, np.ndarray):
                variable = self.add_weight(
                    value.shape, initializer=value, trainable=trainable, name=name_str
                )
                return variable
            elif isinstance(value, (np.generic, int, float)):
                variable = self.add_weight(
                    (), initializer=value, trainable=trainable, name=name_str
                )
                return variable
            else:
                return value

        # Use JAX's tree_map as it understands registered classes.
        variables = named_tree_map(create_variable, values)

        if trainable:
            self.params = variables
        else:
            self.state = variables

        flat_variables, _ = jax.tree_util.tree_flatten(variables)
        return flat_variables


def convert_maxtext_model_to_keras_model(
    maxtext_model, variables, seq_len: int, global_batch_size: int, mesh, config
) -> Model:
    """Convert a MaxText model to a Keras model

    Args:
        maxtext_model: The MaxText model to convert
        seq_len: Length of the input sequence
        global_batch_size: Batch size for the model

    Returns:
        A Keras Model instance
    """

    def maxtext_wrapper(
        module: Any, inputs: List[Union[np.ndarray, jnp.ndarray]], training: bool
    ) -> Any:
        tokens, positions, segment_ids = inputs
        model_mode = "train" if training else "autoregressive"
        segment_ids = segment_ids if training else None
        with mesh, flax.linen.partitioning.axis_rules(config.logical_axis_rules):
            return module(
                tokens,
                positions,
                segment_ids,
                enable_dropout=training,
                model_mode=model_mode,
            )

    keras_layer = MaxTextLayer(
        module=maxtext_model, method=maxtext_wrapper, variables=variables
    )

    # Build the Keras model
    tokens = Input(
        shape=(seq_len,), batch_size=global_batch_size, dtype="int32", name="tokens"
    )
    positions = Input(
        shape=(seq_len,), batch_size=global_batch_size, dtype="int32", name="positions"
    )
    segment_ids = Input(
        shape=(seq_len,),
        batch_size=global_batch_size,
        dtype="int32",
        name="segment_ids",
    )
    logits = keras_layer([tokens, positions, segment_ids], training=True)
    keras_model = Model(inputs=[tokens, positions, segment_ids], outputs=logits)

    return keras_model


@lru_cache(maxsize=1)
def get_maxtext_pyconfig(
    model_name: Optional[str] = None, maxtext_config: Optional[str] = None
):
    argv = [
        "",
        "maxtext/MaxText/configs/base.yml",
        "run_name=must_supply_but_not_needed",
        ]
    
    if model_name is not None: 
        argv += [f"model_name={model_name}"]
    if maxtext_config is not None: 
        argv += maxtext_config.split(" ")
    # pyconfig.initialize must be called before
    # any JAX computations are executed.
    pyconfig.initialize(argv)
    config = pyconfig.config
    return config
