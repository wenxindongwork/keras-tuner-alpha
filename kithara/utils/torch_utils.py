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

import torch
import numpy as np
from jax.experimental import multihost_utils
from typing import Optional


def convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: Optional[str] = None
) -> torch.Tensor:
    expected_dtype = str(weight.dtype) if dtype is None else dtype
    expected_shape = weight.shape
    weight = multihost_utils.process_allgather(weight)
    weight = np.array(weight, dtype="float32")
    torch_dtype = getattr(torch, expected_dtype)
    torch_array = torch.from_numpy(weight).to(torch_dtype).reshape(expected_shape)
    return torch_array
