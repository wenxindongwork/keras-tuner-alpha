from jax.sharding import NamedSharding
from jax.tree_util import tree_reduce
from jax.tree_util import tree_map
from keras.distribution import DeviceMesh
from jax.sharding import Mesh
from keras_tuner.common.utils import named_tree_map
import jax
import keras
from typing import Union

"""Util functions for array sharding"""


def is_not_sharded(jax_array):
    if isinstance(jax_array.sharding, NamedSharding):
        return all(x is None for x in jax_array.sharding.spec)
    return False


def is_not_sharded_and_is_large(
    jax_array: Union[jax.Array, keras.Variable], threshold_mb: float = 0
) -> bool:
    """
    Checks if a JAX array is unsharded and larger than a specified size threshold.

    Args:
        jax_array: Either a JAX array or a Keras Variable containing a JAX array
        threshold_mb: Size threshold in megabytes (default: 20 MB)

    Returns:
        bool: True if the array is unsharded and larger than the threshold, False 
        if the array is not a JAX array or a Keras Variable containing a JAX array. 
    """
    threshold_bytes = int(threshold_mb * 1024 * 1024)

    actual_array = jax_array
    if isinstance(jax_array, keras.Variable):
        actual_array = jax_array.value

    if not isinstance(actual_array, jax.Array):
        return False

    if not isinstance(actual_array.sharding, NamedSharding):
        return False

    is_unsharded = all(x is None for x in actual_array.sharding.spec)
    is_large_enough = actual_array.size >= threshold_bytes

    return is_unsharded and is_large_enough

def print_elements_that_are_unsharded_and_large_in_pytree(pytree):
    def print_fn(path, x):
        if is_not_sharded_and_is_large(x): 
            print(f"{path} is unsharded and has shape", x.shape)
    named_tree_map(print_fn, pytree)

def entire_tree_is_sharded(pytree):
    return not tree_reduce(lambda x, y: x or y, tree_map(is_not_sharded, pytree))


def get_size_in_mb(jax_array):
    bytes_size = jax_array.nbytes
    return int(bytes_size / (1024 * 1024))


def convert_keras_mesh_to_jax_mesh(keras_mesh: DeviceMesh) -> Mesh:
    """Converts a Keras DeviceMesh to a JAX Mesh.

    This function takes a Keras DeviceMesh instance and converts it to an equivalent
    JAX Mesh instance, preserving the device configuration, shape, and axis names.

    Args:
        keras_mesh (DeviceMesh): The Keras DeviceMesh to convert. Should be a valid
            DeviceMesh instance with properly configured devices, shape, and axis names.

    Returns:
        Mesh: A JAX Mesh instance with equivalent configuration to the input
            Keras DeviceMesh.

    Example:
        ```python
        keras_mesh = DeviceMesh(
            shape=(8,),
            axis_names=("batch",),
            devices=jax.devices()
        )
        jax_mesh = convert_keras_mesh_to_jax_mesh(keras_mesh)
        ```
    """
    return Mesh(
        devices=keras_mesh.devices.reshape(keras_mesh.shape),
        axis_names=keras_mesh.axis_names,
    )


def convert_jax_mesh_to_keras_mesh(jax_mesh: Mesh) -> DeviceMesh:
    """Converts a JAX Mesh to a Keras DeviceMesh.

    This function takes a JAX Mesh instance and converts it to an equivalent
    Keras DeviceMesh instance, preserving the device configuration, shape, and axis names.

    Args:
        jax_mesh (Mesh): The JAX Mesh to convert. Should be a valid Mesh instance
            with properly configured devices, shape, and axis names.

    Returns:
        DeviceMesh: A Keras DeviceMesh instance with equivalent configuration to
            the input JAX Mesh.

    Example:
        ```python
        jax_mesh = Mesh(
            devices=jax.devices(),
            axis_names=("batch",)
        )
        keras_mesh = convert_jax_mesh_to_keras_mesh(jax_mesh)
        ```

    Note:
        The mesh shape is reconstructed from the JAX mesh's shape dictionary using
        the order specified in axis_names to ensure consistent dimension ordering.
    """
    mesh_shape = tuple(jax_mesh.shape[name] for name in jax_mesh.axis_names)
    return DeviceMesh(
        shape=mesh_shape,
        axis_names=jax_mesh.axis_names,
        devices=jax_mesh.devices,
    )
