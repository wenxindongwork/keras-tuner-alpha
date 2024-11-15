import ray
from examples.example_datasets import example_datasets
from typing import List, Any
from keras_tuner.dataset.split import split_dataset

ray.init()

num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_devices = int(ray.cluster_resources()["TPU"])
num_tpu_hosts = num_tpu_devices // num_chips_per_host

print(f"{num_tpu_devices=}")
print(f"{num_tpu_hosts=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def main(train_ds, eval_ds, dataset_is_sharded_per_host):
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    import sys

    # Add the MaxText directory to the Python path
    maxtext_dir = "maxtext/MaxText"
    sys.path.append(maxtext_dir)

    from examples.maxtext_example import run_workload

    run_workload(
        train_dataset=train_ds,
        dataset_is_sharded_per_host=dataset_is_sharded_per_host,
    )


train_ds, eval_ds = example_datasets(option="finetune_toy")

split_data_across_host = False

if split_data_across_host:
    train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_tpu_hosts)
    eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_tpu_hosts)
    ray.get([main.remote(train_ds[i], eval_ds[i], split_data_across_host)
            for i in range(num_tpu_hosts)])
else:
    ray.get([main.remote(train_ds, eval_ds, split_data_across_host)
            for _ in range(num_tpu_hosts)])

ray.shutdown()
