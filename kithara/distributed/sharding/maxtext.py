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

from kithara.distributed.sharding.strategy import ShardingStrategy
from dataclasses import dataclass
from keras.distribution import DeviceMesh, LayoutMap
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding, PartitionSpec as P
from typing import Any
from jax.tree_util import tree_flatten_with_path
from keras.src.distribution.distribution_lib import TensorLayout
import keras
from kithara.distributed.sharding.utils import convert_jax_mesh_to_keras_mesh
from jax.sharding import Mesh
import jax

# Hacky monkey patching for now. Keras validate_axes currently does not accept nested tuples as the ParitionSpec value.
# TODO: Patch Keras and remove this logic
TensorLayout._validate_axes = lambda x: x


@dataclass
class MaxTextSharding(ShardingStrategy):
    """Sharding strategy from MaxText config.
        # TODO: Add docstring
    """

    jax_mesh: Mesh
    state_shardings: Any
    maxtext_config: 'pyconfig.Hyperparameter'

    def __post_init__(self) -> None:
        # """Initialize the sharding strategy configuration."""

        self._jax_mesh = self.jax_mesh
        self.jax_devices = self.jax_mesh.devices
        self.maxtext_config = self.maxtext_config

        self._mesh = self._configure_mesh(self.jax_mesh)
        self._layout_map = self._configure_layout_map(self.state_shardings)
        self._data_sharding = self._configure_data_sharding()
        self._distribution = self._configure_distribution()

    @property
    def mesh(self) -> DeviceMesh:
        return self._mesh

    @property
    def layout_map(self) -> LayoutMap:
        return self._layout_map

    @property
    def data_sharding(self) -> Sharding:
        return self._data_sharding

    @property
    def distribution(self) -> Sharding:
        return self._distribution

    def _configure_mesh(self, jax_mesh) -> DeviceMesh:
        return convert_jax_mesh_to_keras_mesh(jax_mesh)

    def _configure_layout_map(self, state_shardings) -> LayoutMap:
        layout_map = LayoutMap(self._mesh)

        # Maps regex string to tuple
        # E.g. .*params.decoder.decoder.norm.scale.* -> ('tensor',)
        var_path_and_sharding, * \
            _ = tree_flatten_with_path(state_shardings.params)
        mappings = {
            ".*"
            + ".".join(str(k.key) for k in var_path)
            + ".*": tuple(var_sharding.spec)
            for var_path, var_sharding in var_path_and_sharding
        }

        for pattern, layout in mappings.items():
            layout_map[pattern] = layout

        return layout_map

    def _configure_data_sharding(self) -> Sharding:
        data_pspec = P(*self.maxtext_config.data_sharding)
        data_sharding = jax.tree_util.tree_map(
            lambda p: jax.sharding.NamedSharding(self._jax_mesh, p), data_pspec)
        return data_sharding

    def _configure_distribution(self) -> Distribution:
        return keras.distribution.ModelParallel(layout_map=self._layout_map)
