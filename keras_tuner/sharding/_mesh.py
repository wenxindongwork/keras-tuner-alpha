from keras.distribution import DeviceMesh
import jax
from enum import Enum
from dataclasses import dataclass
from typing import ClassVar


class Axis(Enum):
    """Enumeration of supported sharding axes in predefined shardings.

    Attributes:
        FSDP: Fully Sharded Data Parallel axis
        TP: Tensor Parallel axis
    """

    FSDP = "fsdp"
    TP = "tp"


@dataclass
class Mesh:
    # Class-level dictionary to store mesh types
    _mesh_types: ClassVar[dict] = {"tp": lambda: Mesh.tp(), "fsdp": lambda: Mesh.fsdp()}

    def __class_getitem__(cls, key: str):
        if key not in cls._mesh_types:
            raise KeyError(f"Unknown mesh type: {key}")
        return cls._mesh_types[key]()

    @classmethod
    def fsdp(cls):
        return DeviceMesh(
            shape=(len(jax.devices()),),
            axis_names=(Axis.FSDP,),
            devices=jax.devices(),
        )

    @classmethod
    def tp(cls):
        return DeviceMesh(
            shape=(len(jax.devices()),),
            axis_names=(Axis.TP,),
            devices=jax.devices(),
        )
