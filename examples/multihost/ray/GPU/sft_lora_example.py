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

import ray
from typing import List, Any
from kithara.distributed.data import split_dataset

ray.init()

num_chips_per_host = 4
num_gpu_devices = int(ray.cluster_resources()["GPU"])
print(f"{num_gpu_devices=}")


@ray.remote(num_gpus=num_chips_per_host)
def main(train_ds, eval_ds, split_data_across_host):
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    from examples.singlehost.full_finetuning_example import run_workload

    run_workload(
        train_source=train_ds,
        eval_source=eval_ds,
        dataset_is_sharded_per_host=split_data_across_host,
    )


# Create mulit-host datasets
dataset_items = [
    {
        "prompt": "What is your name?",
        "answer": "My name is Mary",
    }
    for _ in range(1000)
]
dataset = ray.data.from_items(dataset_items)
train_ds, eval_ds = dataset.train_test_split(test_size=500)

split_data_across_host = False
if split_data_across_host:
    train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_gpu_devices)
    eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_gpu_devices)
    ray.get(
        [
            main.remote(train_ds[i], eval_ds[i], split_data_across_host)
            for i in range(num_gpu_devices)
        ]
    )
else:
    ray.get(
        [
            main.remote(train_ds, eval_ds, split_data_across_host)
            for i in range(num_gpu_devices)
        ]
    )

ray.shutdown()
