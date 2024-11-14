import ray
from datasets import load_dataset
from typing import List, Any
from keras_tuner.dataset.split import split_ray_dataset, split_files

ray.init()

num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_devices = int(ray.cluster_resources()["TPU"] / num_chips_per_host)

print(f"{num_tpu_devices=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def main(train_ds=None, eval_ds=None, train_files=None, eval_files=None, dataset_processing_fn= None):
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
        train_files=train_files,
        eval_files=eval_files,
        dataset_processing_fn=None,
        data_is_already_split_across_hosts=False,
    )


# option 1. Create file groups
train_data_input: List[str] = split_files(["file1.json", "file2.json"], num_hosts = num_tpu_devices)
eval_data_input: List[str] = split_files(["file3.json", "file4.json"], num_hosts = num_tpu_devices)
dataset_processing_fn = lambda x, y: ray.data.fron_json(train_data_input), ray.data.from_json(eval_data_input)

# option 2. Create Streaming Dataset
train_ds = load_dataset("allenai/c4", "en", split='train', streaming=True)
test_ds = load_dataset("allenai/c4", "en", split='validation', streaming=True)
train_ds = ray.data.from_huggingface(train_ds)
eval_ds = ray.data.from_huggingface(test_ds)

# option 3. Create Materialized Dataset
dataset_items = [
    {"text": f"{i} What is your name? My name is Mary."} for i in range(1000)
]
dataset = ray.data.from_items(dataset_items)
train_ds, eval_ds = dataset.train_test_split(test_size=500)

split =  False

if split: 
    train_ds: List[Any] = split_ray_dataset(train_ds, num_hosts=num_tpu_devices)
    eval_ds: List[Any] = split_ray_dataset(eval_ds, num_hosts=num_tpu_devices)
    ray.get([main.remote(train_ds[i], eval_ds[i]) for i in range(num_tpu_devices)])
else:
    ray.get([main.remote(train_ds, eval_ds) for i in range(num_tpu_devices)])

ray.shutdown()
