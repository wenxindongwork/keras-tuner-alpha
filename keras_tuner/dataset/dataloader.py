from typing import Iterator, Any, Iterable, Dict
import ray
import jax


class Dataloader(Iterable):
    """A dataloader over the provided Ray Dataset.

    Attributes:
        per_device_batch_size (int): Batch size for each device
    """

    def __init__(
        self,
        ray_dataset: ray.data.Dataset,
        per_device_batch_size: int,
        dataset_is_sharded_per_host: bool = False,
    ):
        """Initialize the Dataset wrapper.

        Args:
            ray_dataset: The source Ray dataset to process. Check out
            https://docs.ray.io/en/latest/data/loading-data.html for how
            to load your raw source into a Ray Dataset. Supported formats
            include Huggingface Hub, json, csv, Python list, and more.
            per_device_batch_size: Number of samples per batch per device
            dataset_is_sharded_per_host: True if dataset is already sharded
            and each host should load its own data shard. False if every host
            should load the global batch and keep its own shard and disgard the rest.
        """

        num_devices_per_host = jax.local_device_count()
        num_devices = jax.device_count()
        self.per_device_batch_size = per_device_batch_size
        self.per_host_batch_size = self.per_device_batch_size * num_devices_per_host
        self.global_batch_size = self.per_device_batch_size * num_devices
        self.dataset_is_sharded_per_host = dataset_is_sharded_per_host

        if self.dataset_is_sharded_per_host:
            self._iterator = iter(ray_dataset.iter_batches(
                batch_size=self.per_host_batch_size, drop_last=True
            ))
        else:
            self._iterator = iter(ray_dataset.iter_batches(
                batch_size=self.global_batch_size, drop_last=True
            ))

    def __iter__(self) -> Iterator[Any]:
        """Return iterator over batches in the dataset.

        Yields:
            Batches of data from the dataset
        """
        return self

    def __next__(self) -> Dict[str, str]:
        """Get next batch of data from the dataset.

        Returns:
            Per-host batch input
        """
        batch = next(self._iterator)

        if not self.dataset_is_sharded_per_host:
            # If dataset isn't pre-sharded, we need to extract just this host's portion
            host_id = jax.process_index()
            host_start = host_id * self.per_host_batch_size
            host_end = host_start + self.per_host_batch_size

            # Extract this host's portion from the global batch
            batch = {k: v[host_start:host_end] for k, v in batch.items()}

        return batch
