from keras_tuner.trainer.sharding.sharding import ShardingStrategy
from dataclasses import dataclass, field
from keras.distribution import DeviceMesh, LayoutMap
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding, NamedSharding
from jax.sharding import PartitionSpec as P
from typing import Any, Type
from jax.tree_util import tree_flatten_with_path
from keras.src.distribution.distribution_lib import TensorLayout
import keras
from maxtext.MaxText import max_utils
from maxtext.MaxText.train import setup_mesh_and_model

# Hacky monkey patching for now. Keras validate_axes currently does not accept nested tuples as the ParitionSpec value.
TensorLayout._validate_axes = lambda x: x


@dataclass
class MaxTextSharding(ShardingStrategy):
    """Sharding strategy from MaxText config."""

    maxtext_config: Any = field(init=True)

    def __post_init__(self) -> None:
        # """Initialize the sharding strategy configuration."""

        (
            init_rng,
            _,
            _,
            jax_mesh,
            model,
            _,
            tx,
        ) = setup_mesh_and_model(self.maxtext_config)

        self._jax_mesh = jax_mesh
        self.jax_devices = jax_mesh.devices

        _, _, state_shardings = max_utils.get_abstract_state(
            model, tx, self.maxtext_config, init_rng, jax_mesh, is_training=True
        )

        self._mesh = self._configure_mesh(jax_mesh)
        self._layout_map = self._configure_layout_map(state_shardings)

        self._data_sharding = self._configure_data_sharding()

        self._distribution = self._configure_distribution()

        self.validate()

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
        mesh_axis_names = jax_mesh.axis_names
        mesh_shape = tuple(jax_mesh.shape[name] for name in mesh_axis_names)
        return DeviceMesh(
            shape=mesh_shape, axis_names=mesh_axis_names, devices=self.jax_devices
        )

    def _configure_layout_map(self, state_shardings) -> LayoutMap:
        layout_map = LayoutMap(self._mesh)

        # Mapping from regex key to tuple
        # E.g. .*params.decoder.decoder.norm.scale.* -> ('tensor',)
        mappings = {
            ".*"
            + ".".join(str(k.key) for k in var_path)
            + ".*": tuple(var_sharding.spec)
            for var_path, var_sharding in tree_flatten_with_path(
                state_shardings.params
            )[0]
        }
        for pattern, layout in mappings.items():
            layout_map[pattern] = layout

        return layout_map

    def _configure_data_sharding(self) -> Sharding:
        return NamedSharding(
            self._jax_mesh,
            P("data"),
        )

    def _configure_distribution(self) -> Distribution:
        return keras.distribution.ModelParallel(layout_map=self._layout_map)
