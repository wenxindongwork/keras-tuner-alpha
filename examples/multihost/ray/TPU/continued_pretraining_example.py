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

"""Full parameter continued pretraining a Gemma2 9B model.

This script demonstrates how to

1. Load HuggingFace Gemma2 checkpoint
2. Loading large HuggingFace dataset
3. Configure data loading and preprocessing
4. Run training across TPU/GPU devices
5. Save checkpoint to GCS periodically 
6. Generate text using the trained model
7. Save model in HuggingFace format to GCS

This script can be run on both single-host and multi-host. 
For mulit-host set up, please follow https://kithara.readthedocs.io/en/latest/scaling_with_ray.html.

Singlehost: python examples/singlehost/continued_pretraining_example.py 
Multihost:  python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/continued_pretraining_example.py" --hf-token your_token
"""

import ray
import jax

ray.init()

# Verify TPU resources
num_chips_per_host = 4  # <-- IMPORTANT: Use 4 for v4 and v5, 8 for v4e and v5e
num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)
print(f"{num_tpu_hosts=}")


@ray.remote(resources={"TPU": num_chips_per_host})
def main():

    import subprocess

    subprocess.run(["rm", "-rf", "/tmp/libtpu_lockfile", "/tmp/tpu_logs"])

    # HuggingFace login
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    jax.distributed.initialize()

    from examples.singlehost.continued_pretraining_example import run_workload

    # Save model outputs to GCS
    run_workload(
        gemma2_model_size="9b", model_output_dir="gs://your_gs_bucket/model_output"
    )


ray.get([main.remote() for i in range(num_tpu_hosts)])

ray.shutdown()
