import ray
from examples.example_datasets import example_datasets
from typing import List, Any
from kithara.dataset.utils import split_dataset

ray.init()

# Verify TPU resources
num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)
print(f"{num_tpu_hosts=}")

@ray.remote(resources={"TPU": num_chips_per_host})
def main(train_ds, eval_ds, split_data_across_host):
    
    # HuggingFace login
    from huggingface_hub import login
    import os
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    
    # Run workload in SPMD mode
    from examples.singlehost.sft_lora_example import run_workload
    run_workload(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_processing_fn=None,
        dataset_is_sharded_per_host=split_data_across_host,
    )

# Create mulit-host datasets
train_ds, eval_ds = example_datasets(option = "sft_toy")
split_data_across_host =  False
if split_data_across_host: 
    train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_tpu_hosts)
    eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_tpu_hosts)
    ray.get([main.remote(train_ds[i], eval_ds[i], split_data_across_host) for i in range(num_tpu_hosts)])
else:
    ray.get([main.remote(train_ds, eval_ds, split_data_across_host) for i in range(num_tpu_hosts)])

ray.shutdown()
