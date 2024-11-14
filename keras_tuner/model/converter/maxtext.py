from typing import Any, Dict, List, Union
from keras.src.utils.jax_layer import FlaxLayer
from keras.layers import Input
from keras.models import Model
import numpy as np
import jax.numpy as jnp
from keras.src import backend
from keras.src.utils import tracking
from keras_tuner.common.utils import named_tree_map
from maxtext.MaxText import pyconfig
from jax.sharding import Mesh
from maxtext.MaxText import max_utils
from maxtext.MaxText.layers.models import Transformer
from maxtext.MaxText.layers import quantizations
import jax


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
                    value.shape, initializer="zeros", trainable=trainable, name=name_str
                )
                # Need to first shard the value returned by the MaxText model
                # as variable assign expects the value to be either fully addressable
                # or has the same sharding as the variable's layout.
                value = jax.lax.with_sharding_constraint(value, variable._layout)
                variable.assign(value)
                return variable
            elif isinstance(value, (np.generic, int, float)):
                variable = self.add_weight(
                    (), initializer="zeros", trainable=trainable, name=name_str
                )
                variable.assign(value)
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
    maxtext_model: Any, seq_len: int, global_batch_size: int
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
        return module(
            tokens,
            positions,
            segment_ids,
            enable_dropout=training,
            model_mode=model_mode,
        )

    keras_layer = MaxTextLayer(
        module=maxtext_model,
        method=maxtext_wrapper,
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
    x = keras_layer([tokens, positions, segment_ids], training=True)
    keras_model = Model(inputs=[tokens, positions, segment_ids], outputs=x)

    return keras_model


def get_maxtext_config(model_name="default"):
    argv = [
        "",
        "maxtext/MaxText/configs/base.yml",
        f"model_name={model_name}",
        "run_name=must_supply_but_not_needed",
    ]
    # pyconfig.initialize must be called before 
    # any JAX computations are executed. 
    pyconfig.initialize(argv)
    config = pyconfig.config
    return config


def get_maxtext_model(maxtext_config):

    devices_array = max_utils.create_device_mesh(maxtext_config)
    mesh = Mesh(devices_array, maxtext_config.mesh_axes)
    quant = quantizations.configure_quantization(maxtext_config)

    # Model is not parameterized nor initialized.
    model = Transformer(maxtext_config, mesh, quant)
    return model
