from keras_tuner.model.sharding.models import GEMMA_LAYOUT
from dataclasses import dataclass
from typing import ClassVar


# Layout configurations for different model architectures
@dataclass
class Layout:
    # Class-level dictionary to store mesh types
    _mesh_types: ClassVar[dict] = {
        "gemma": lambda: Layout.gemma(),
    }

    def __class_getitem__(cls, key: str):
        if key not in cls._mesh_types:
            raise KeyError(f"Unknown mesh type: {key}")
        return cls._mesh_types[key]()

    @classmethod
    def gemma(cls):
        return GEMMA_LAYOUT
