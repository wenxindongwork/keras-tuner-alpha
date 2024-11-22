from dataclasses import dataclass
from typing import Optional
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, set_distribution
from keras.src.distribution.distribution_lib import Distribution
from jax.sharding import Sharding
from keras_tuner.model.sharding._mesh import Mesh as PredefinedMesh
from keras_tuner.model.sharding._layout import Layout as PredefinedLayout
from keras_tuner.model.sharding._data_sharding import DataSharding
from abc import ABC, abstractmethod
from keras.src.backend.common import global_state

@dataclass
class ShardingStrategy(ABC):
    """Abstract base class for defining model sharding strategies.

    This class provides a framework for implementing different sharding strategies
    for distributed model training. It defines the required properties that all concrete
    sharding strategies must implement.

    Properties:
        mesh: The device mesh configuration for distributed training
        layout_map: The layout mapping for model parameters across devices
        data_sharding: The sharding specification for input data
        distribution: The Keras distribution strategy

    See `PredefinedShardingStrategy` for an example.
    """

    @property
    @abstractmethod
    def mesh(self) -> DeviceMesh:
        pass

    @property
    @abstractmethod
    def layout_map(self) -> LayoutMap:
        pass

    @property
    @abstractmethod
    def data_sharding(self) -> Sharding:
        pass

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        pass


@dataclass
class PredefinedShardingStrategy(ShardingStrategy):
    """This class provides pre-configured sharding strategies optimized for specific
    model architectures and parallelism types.

    Args:
        parallelism (str): The type of parallelism to use. Must be one of ["fsdp", "tp"]
        model (str): The model architecture. Must be one of ["gemma"]

    Example:
        ```python
        strategy = PredefinedShardingStrategy(
            parallelism="fsdp",
            model="gemma"
        )
        set_global_sharding_strategy(strategy)
        ```
    """

    parallelism: str
    model: str

    @property
    def mesh(self) -> DeviceMesh:
        return PredefinedMesh[self.parallelism]

    @property
    def layout_map(self) -> LayoutMap:
        layout_dict = PredefinedLayout[self.model][self.parallelism]
        layout_map = LayoutMap(self.mesh)
        for pattern, layout in layout_dict.items():
            layout_map[pattern] = layout
        return layout_map

    @property
    def data_sharding(self) -> Sharding:
        return DataSharding[self.parallelism]

    @property
    def distribution(self) -> Distribution:
        return ModelParallel(layout_map=self.layout_map)



def set_global_sharding_strategy(strategy: ShardingStrategy) -> None:
    """Sets the global sharding strategy for model training. This function
    must be called before model and optimizer loading.

    Args:
        strategy (ShardingStrategy): The sharding strategy to apply globally
    """
    if global_state.get_global_attribute("distribution") is not None: 
        print(
            "WARNING: distribution is being overriden."
        )
    set_distribution(strategy.distribution)
    global_state.set_global_attribute("DATA_SHARDING", strategy.data_sharding)
