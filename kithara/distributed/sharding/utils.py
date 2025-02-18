"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from jax.sharding import NamedSharding
from jax.tree_util import tree_reduce
from jax.tree_util import tree_map
from keras.distribution import DeviceMesh
from jax.sharding import Mesh
from kithara.utils.tree_utils import named_tree_map
import jax
import keras
from typing import Union
import numpy as np 

"""Util functions for array sharding"""


def is_not_sharded(jax_array):
    if isinstance(jax_array.sharding, NamedSharding):
        return all(x is None for x in jax_array.sharding.spec)
    return False


def is_not_sharded_and_is_large(
    jax_array: Union[jax.Array, keras.Variable], threshold_mb: float = 20
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
    size_in_bytes = np.prod(jax_array.shape) * jax_array.dtype.itemsize
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


def get_size_in_gb(jax_array):
    size_in_bytes = np.prod(jax_array.shape) * jax_array.dtype.itemsize
    size_in_gb = size_in_bytes / (1024 * 1024 * 1024)
    return size_in_gb


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


def create_fsdp_layout_map(model: keras.Model, threshold_mb=5) -> keras.distribution.LayoutMap:
    """
    Analyzes model weights and creates an FSDP layout map based on weight shapes.
    
    Args:
        model: A Keras model
        threshold_mb: Minimum parameter size in mb for FSDP sharding
        
    Returns:
        dict: Layout map with sharding configuration for each weight
    
    Usage: 
        model = keras_hub.models.CausalLM.from_preset(
        "hf://google/gemma-2-2b",
        preprocessor=None,
        load_weights=False)

        layout_map = create_fsdp_layout_map(model)
    """
    layout_map = {}
            
    for var in model.weights:
        shape = var.shape
        path = var.path        
        params_mb = np.prod(shape)// (1024 * 1024)
        
        if params_mb > threshold_mb:
            # Find dimension with maximum size for sharding
            max_dim = np.argmax(shape)
            
            # Create sharding configuration
            fsdp_sharding = [None] * len(shape)
            fsdp_sharding[max_dim] = "fsdp"
            layout_map[path] = tuple(fsdp_sharding)
        else:
            # No sharding needed
            layout_map[path] = tuple(None for _ in shape)
    
    return layout_map