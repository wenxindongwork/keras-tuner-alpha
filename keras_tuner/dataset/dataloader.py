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

        self.per_device_batch_size = per_device_batch_size
        self.dataset_is_sharded_per_host = dataset_is_sharded_per_host

        if self.dataset_is_sharded_per_host:
            self._iteratable_ds = ray_dataset.iter_batches(
                    batch_size=self.per_host_batch_size, drop_last=True
            )
        else:
            self._iteratable_ds = ray_dataset.iter_batches(
                    batch_size=self.global_batch_size, drop_last=True
                )
        self._iterator = None

    def __iter__(self) -> Iterator[Any]:
        """Return iterator over batches in the dataset.

        Yields:
            Per-host batch input
        """
        self._iterator = iter(self._iteratable_ds)
        return self

    def __next__(self) -> Dict[str, str]:
        """Get next batch of data from the dataset.

        Returns:
            Per-host batch input
        """
        batch = next(self._iterator)

        if not self.dataset_is_sharded_per_host:
            # If dataset isn't pre-sharded, we need to extract just 
            # this host's portion from the global batch
            host_id = jax.process_index()
            host_start = host_id * self.per_host_batch_size
            host_end = host_start + self.per_host_batch_size

            batch = {k: v[host_start:host_end] for k, v in batch.items()}

        return batch

    @property
    def global_batch_size(self):
        num_devices = jax.device_count()
        return self.per_device_batch_size * num_devices
    @property
    def per_host_batch_size(self):
        num_devices_per_host = jax.local_device_count()
        return self.per_device_batch_size * num_devices_per_host

