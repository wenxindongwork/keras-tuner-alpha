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

from keras.distribution import DeviceMesh
import jax
from enum import Enum
from dataclasses import dataclass
from typing import ClassVar


class Axis(str, Enum):
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
