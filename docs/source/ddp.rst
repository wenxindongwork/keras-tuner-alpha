.. _ddp:

Managing Large Datasets
=======================

Distributed Data Parallel is a data loading technique useful for multi-host training on large datasets.

When to use?
^^^^^^^^^^^^^^^
- When you are running multihost training, and your dataset is too large to fit in memory on a single host. 
- When large batch processing is the bottleneck in your training pipeline.


How it works?
^^^^^^^^^^^^^^^

1. Split your dataset into N shards,  where N is the number of hosts. Each shard having the same number of samples. 
    .. tip:: 
        Use ``ray.data.split()`` on your Ray dataset, or ``ray.data.streaming_split()`` for streamed datasets
2. Each host loads and processes data only from its assigned shard

   For example, with a 32,000-sample dataset split across 4 hosts, each host manages 8,000 samples instead of the full dataset

Example DDP Workflow
^^^^^^^^^^^^^^^^^^
Here's an example of how to implement DDP in Kithara::

    @ray.remote(resources={"TPU": num_chips_per_host})
    def main(train_ds):
        ...
        dataloader = Kithara.Dataloader(
            train_ds, 
            per_device_batch_size=1, # or any other batch size
            dataset_is_sharded_per_host=True, # Enable DDP
        )    
        ...
        trainer.train() 

    train_ds: List[Any] = split_dataset(ray_dataset, num_hosts=num_tpu_hosts)
    ray.get(
        [
            main.remote(train_ds[i]) for i in range(num_tpu_hosts)
        ]
    )