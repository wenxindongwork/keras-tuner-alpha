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

from typing import Iterator, Any, Iterable, Dict
import jax
from kithara.dataset.dataset import Dataset
from jax.tree_util import tree_map
import numpy as np 

class Dataloader(Iterable):
    """Kithara Dataloader class. This dataloader class supports distributed
    data loading for multi-host training. 

    Attributes:
        dataset: A Kithara Dataset instance, or any Iterable instance 
            that that returns individual, tokenized samples formatted
            as expected by the model.
        per_device_batch_size: Number of samples per batch per device. 
            If you experience HBM OOM errors, try reducing this value.
        dataset_is_sharded_per_host: True if dataset is already sharded
            and each host is provided with a local dataset shard. False if 
            every host is loading from the same dataset. 
    """

    def __init__(
        self,
        dataset: Dataset,
        per_device_batch_size: int,
        dataset_is_sharded_per_host: bool = False,
    ):
        self.dataset = dataset
        self.per_device_batch_size = per_device_batch_size
        self.dataset_is_sharded_per_host = dataset_is_sharded_per_host
        self._iterator = None
        self.num_hosts = jax.process_count()
        self.host_id = jax.process_index()

    def __iter__(self) -> Iterator[Any]:
        """Return iterator over batches in the dataset.

        Yields:
            Per-host batch input
        """
        self._iterator = iter(self.dataset)
        return self

    def __next__(self) -> Dict[str, str]:
        """Get next batch of data from the dataset.

        Returns:
            Per-host batch input
        """
        if self.dataset_is_sharded_per_host:
            # For sharded datasets, each host processes per_host_batch_size samples
            samples  = [next(self._iterator) for _ in range(self.per_host_batch_size)]
        else:
            # For non-sharded datasets:
            # 1. Each host loads global_batch_size samples
            # 2. Only keeps samples corresponding to its host_id            
            samples = []
            for i in range(self.global_batch_size):
                sample = next(self._iterator)
                if i % self.num_hosts == self.host_id:
                    samples.append(sample)
        
        samples = tree_map(
            lambda *arrays: np.concatenate(arrays), *samples
        )
        return samples

    def __len__(self) -> int:
        """Get the total number of batches in the dataset.

        Returns:
            Total number of batches that will be yielded by this dataloader.
            For non-sharded datasets, this is the same across all hosts.
            For sharded datasets, each host has its own length.
        """
        if not hasattr(self.dataset, '__len__'):
            raise TypeError("Dataset must implement __len__ to support length calculation")
        
        total_samples = len(self.dataset)
        if self.dataset_is_sharded_per_host:
            batches = total_samples // self.per_host_batch_size
        else:
            batches = total_samples // (self.num_hosts * self.per_host_batch_size)
        return batches

    @property
    def global_batch_size(self):
        num_devices = jax.device_count()
        return self.per_device_batch_size * num_devices
    @property
    def per_host_batch_size(self):
        num_devices_per_host = jax.local_device_count()
        return self.per_device_batch_size * num_devices_per_host

