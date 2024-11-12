from abc import ABC, abstractmethod
from dataclasses import dataclass
from keras.distribution import DeviceMesh, LayoutMap
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding

@dataclass
class ShardingStrategy(ABC):
    """Abstract base class for sharding strategies.

    All concrete implementations must define mesh, layout_map, and data_sharding.
    """

    @property
    @abstractmethod
    def mesh(self) -> DeviceMesh:
        """Device mesh configuration for the sharding strategy."""
        pass

    @property
    @abstractmethod
    def layout_map(self) -> LayoutMap:
        """Layout map configuration for the sharding strategy."""
        pass

    @property
    @abstractmethod
    def data_sharding(self) -> Sharding:
        """Data sharding configuration for the sharding strategy."""
        pass

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        """Data sharding configuration for the sharding strategy."""
        pass

    def validate(self) -> None:
        """Validate the sharding strategy configuration."""
        if not isinstance(self.mesh, DeviceMesh):
            raise ValueError(
                "mesh must be an instance of keras.distribution.DeviceMesh"
            )
        if not isinstance(self.layout_map, LayoutMap):
            raise ValueError(
                "layout_map must be an instance of keras.distribution.LayoutMap"
            )
        if not isinstance(self.data_sharding, Sharding):
            raise ValueError(
                "data_sharding must be an instance of jax.sharding.Sharding"
            )
        if not isinstance(self.distribution, Distribution):
            raise ValueError(
                "distribution must be an instance of keras.distribution.Distribution"
            )
