import ray
from examples.example_datasets import example_datasets
from typing import List, Any
from keras_tuner.dataset.split import split_dataset


ray.init()

num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)

print(f"{num_tpu_hosts=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def main(train_ds=None, eval_ds=None, dataset_processing_fn=None):
    from huggingface_hub import login
    import os
    os.environ["KERAS_BACKEND"] = "jax"

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    from examples.hf_gemma_example import run_workload

    print("Running workload")
    run_workload(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_processing_fn=None,
        dataset_is_sharded_per_host=False,
    )


train_ds, eval_ds = example_datasets(option="finetune_toy")

split_data_across_host = False

if split_data_across_host:
    train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_tpu_hosts)
    eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_tpu_hosts)
    ray.get([main.remote(train_ds[i], eval_ds[i])
            for i in range(num_tpu_hosts)])
else:
    ray.get([main.remote(train_ds, eval_ds) for i in range(num_tpu_hosts)])

ray.shutdown()
