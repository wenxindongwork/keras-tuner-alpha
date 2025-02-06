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

"""SFT a Gemma2 2B model using LoRA on TPU or GPU.

This script demonstrates how to:
1. Set up a Gemma2 model for LoRA SFT
2. Load HuggingFace Gemma2 checkpoint
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices

This script can be run on both single-host and multi-host. For multi-host set up, please follow `ray/readme.md`.

Singlehost: python examples/singlehost/sft_lora_example.py 
Multihost:  kithara multihost examples/multihost/ray/TPU/sft_lora_example.py --hf-token <TOKEN>
"""

import ray
from typing import List, Any
from kithara.distributed.data import split_dataset
import jax

ray.init()

# Verify TPU resources
num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)
print(f"{num_tpu_hosts=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def main(train_ds, eval_ds, split_data_across_host):

    import subprocess

    subprocess.run(["rm", "-rf", "/tmp/libtpu_lockfile", "/tmp/tpu_logs"])

    # HuggingFace login
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    jax.distributed.initialize()

    # Run workload in SPMD mode
    from examples.singlehost.sft_lora_example import run_workload

    run_workload(
        train_ds,
        eval_ds,
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
    train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_tpu_hosts)
    eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_tpu_hosts)
    ray.get(
        [
            main.remote(train_ds[i], eval_ds[i], split_data_across_host)
            for i in range(num_tpu_hosts)
        ]
    )
else:
    ray.get(
        [
            main.remote(train_ds, eval_ds, split_data_across_host)
            for i in range(num_tpu_hosts)
        ]
    )

ray.shutdown()
