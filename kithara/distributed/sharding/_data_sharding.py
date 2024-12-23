from kithara.distributed.sharding._mesh import Axis
from jax.sharding import NamedSharding, Mesh
import jax
from jax.sharding import PartitionSpec as P
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class DataSharding:

    # Class-level dictionary to store mesh types
    _mesh_types: ClassVar[dict] = {
        "tp": lambda: DataSharding.tp(),
        "fsdp": lambda: DataSharding.fsdp(),
        "fully_replicated": lambda: DataSharding.fully_replicated()
    }

    def __class_getitem__(cls, key: str):
        if key not in cls._mesh_types:
            raise KeyError(f"Unknown mesh type: {key}")
        return cls._mesh_types[key]()

    @classmethod
    def fsdp(cls):
        return NamedSharding(
            Mesh(jax.devices(), (Axis.FSDP,)),
            P(Axis.FSDP, None),
        )

    @classmethod
    def tp(cls):
        return NamedSharding(
            Mesh(jax.devices(), (Axis.TP,)),
            P(None, Axis.TP),
        )

    @classmethod
    def fully_replicated(cls):
        return NamedSharding(
            Mesh(jax.devices(), ("devices",)),
            P(None),
        )
