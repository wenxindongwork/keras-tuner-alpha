from keras_tuner.trainer.sharding.sharding import ShardingStrategy
from typing import ClassVar
import keras
from keras.distribution import DeviceMesh, LayoutMap, list_devices
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding, NamedSharding, Mesh
from jax.sharding import PartitionSpec as P
import jax
from dataclasses import dataclass

@dataclass
class GemmaFDSP(ShardingStrategy):
    """FSDP sharding strategy for Gemma model."""

    fsdp_dim: ClassVar[str] = "fsdp"

    def __post_init__(self) -> None:
        """Initialize the sharding strategy configuration."""
        self.devices = list_devices()
        num_devices = len(self.devices)
        self._mesh = DeviceMesh(
            shape=(num_devices,),
            axis_names=(self.fsdp_dim,),
            devices=self.devices,
        )

        self._layout_map = self._configure_layout_map()

        self._distribution = self._configure_distribution()

        self._data_sharding = self._configure_data_sharding()

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

    def _configure_layout_map(self) -> LayoutMap:
        layout_map = LayoutMap(self._mesh)
        mappings = {
            ".*token_embedding.embeddings.*": (None, self.fsdp_dim),
            ".*decoder_block.*attention.*(query|key|value).kernel.*": (
                None,
                self.fsdp_dim,
            ),
            ".*decoder_block.*attention_output.kernel.*": (None, None, self.fsdp_dim),
            ".*decoder_block.*ffw_gating.kernel.*": (None, self.fsdp_dim),
            ".*decoder_block.*ffw_gating_2.kernel.*": (None, self.fsdp_dim),
            ".*decoder_block.*ffw_linear.kernel.*": (self.fsdp_dim, None),
            # Lora layers
            ".*decoder_block.*attention.*(query|key|value).lora_kernel.*": (
                None,
                self.fsdp_dim,
            ),
        }

        for pattern, layout in mappings.items():
            layout_map[pattern] = layout

        return layout_map

    def _configure_data_sharding(self) -> Sharding:
        # https://github.com/keras-team/keras/blob/master/keras/src/distribution/distribution_lib.py#L598
        return NamedSharding(
            Mesh(jax.devices(), (self.fsdp_dim,)),
            P("fsdp", None),
        )

    def _configure_distribution(self) -> Distribution:
        return keras.distribution.ModelParallel(layout_map=self._layout_map)
