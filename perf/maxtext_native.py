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

"""
Benchmark for training MaxText models via MaxText.  

This benchmark script runs multi-host training with the specified MaxText model. 

Metrics: step time, TFLOP/s/device,  Tokens/s/device
Artifact: Tensorboard, Xplane (Uploaded to BASE_OUTPUT_DIR)

Purpose: Compare native MaxText performance against performance of MaxText via Kithara. 

Launch Script: python ray/submit_job.py "python perf/maxtext_native.py"

TODO: Launch benchmarks via YAML config.
"""

import ray

if __name__ == "__main__":
    ray.init()

    num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
    num_tpu_devices = int(ray.cluster_resources()["TPU"])
    num_tpu_hosts = num_tpu_devices // num_chips_per_host

    print(f"{num_tpu_devices=}")
    print(f"{num_tpu_hosts=}")

    @ray.remote(resources={"TPU": num_chips_per_host})
    def main():
        import sys
        import subprocess

        # Add the MaxText directory to the Python path
        maxtext_dir = "maxtext/MaxText"
        sys.path.append(maxtext_dir)

        # Run parameters
        BASE_OUTPUT_DIR = "GS_BUCKET"  # MODIFY with your GS bucket
        MODEL_NAME = "gemma2-9b"
        SEQ_LEN = 2048
        PER_DEVICE_BATCH_SIZE = 1
        
        subprocess.call(
            [
                "python",
                "maxtext/MaxText/train.py",
                "kithara/model/maxtext/maxtext/MaxText/configs/base.yml",
                f"model_name={MODEL_NAME}",
                "run_name=maxtext_native",
                f"max_target_length={SEQ_LEN}",
                f"per_device_batch_size={PER_DEVICE_BATCH_SIZE}",
                "steps=10",
                "enable_checkpointing=false",
                "dataset_type=synthetic",
                "profiler=xplane",
                "scan_layers=false",
                "skip_first_n_steps_for_profiler=5",
                "gcs_metrics=true",
                "profiler_steps=5",
                "remat_policy=minimal",
                "attention=flash",
                f"base_output_directory={BASE_OUTPUT_DIR}",
                "dataset_path=gs://max-datasets-rogue/",
            ]
        )

    ray.get([main.remote() for _ in range(num_tpu_hosts)])

    ray.shutdown()
