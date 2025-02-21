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

"""Full parameter finetune a Gemma2 9B model.

This script demonstrates how to:
1. Set up a Gemma2 model for full parameter finetuning
2. Load HuggingFace Gemma2 checkpoint
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices
5. Save checkpoint to GCS periodically 
6. Generate text using the trained model
7. Save model in HuggingFace format to GCS

This script should be run on multihost, since gemma2-9b will not fit on a single host. However, 
you can change the model to `gemma2-2b` to run on single host. 

Singlehost: python examples/singlehost/full_finetuning_example.py 
Multihost:  python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/full_finetuning_example.py" --hf-token your_token

"""

import ray
from absl import app
from typing import List, Any
from kithara.config.pyconfig import load_config
from kithara.distributed.data import split_dataset
import jax

ray.init()

# Verify TPU resources
num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_devices = int(ray.cluster_resources()["TPU"])
num_tpu_hosts = num_tpu_devices // num_chips_per_host
print(f"{num_tpu_devices=}")
print(f"{num_tpu_hosts=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def run(config, train_ds, eval_ds, dataset_is_sharded_per_host):

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
    from examples.singlehost.full_finetuning_example import run_workload

    run_workload(
        config=config,
        train_source=train_ds,
        eval_source=eval_ds,
        dataset_is_sharded_per_host=dataset_is_sharded_per_host,
    )


def main(argv):
  config = load_config(argv)

  # Create multi-host datasets
  dataset_items = [
      {"text": f"{i} What is your name? My name is Mary."} for i in range(1000)
  ]
  dataset = ray.data.from_items(dataset_items)
  train_ds, eval_ds = dataset.train_test_split(test_size=500)

  split_data_across_host = False
  if split_data_across_host:
      train_ds: List[Any] = split_dataset(train_ds, num_hosts=num_tpu_hosts)
      eval_ds: List[Any] = split_dataset(eval_ds, num_hosts=num_tpu_hosts)
      ray.get(
         [
             run.remote(config, train_ds[i], eval_ds[i], split_data_across_host)
             for i in range(num_tpu_hosts)
         ]
      )
  else:
     ray.get(
        [
            run.remote(config, train_ds, eval_ds, split_data_across_host)
            for _ in range(num_tpu_hosts)
        ]
     )

  ray.shutdown()


if __name__ == "__main__":
  app.run(main)
