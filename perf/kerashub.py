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
Benchmark for training MaxText models via Kithara.  

This benchmark script runs multi-host training with the specified MaxText model 

Metrics: step time, Tokens/s/device
Artifact: Tensorboard, Xplane (Uploaded to BASE_OUTPUT_DIR)

Purpose: Compare native MaxText performance against performance of MaxText via Kithara. 

Launch Script: python ray/submit_job.py "python perf/kerashub.py"

TODO: Launch benchmarks via YAML config.
"""

import ray

def run_benchmark():
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    from examples.example_datasets import example_datasets
    from kithara import KerasHubModel
    from kithara.dataset import Dataloader, TextCompletionDataset
    from kithara.trainer import Trainer
    from kithara.distributed import PredefinedShardingStrategy
    from kithara.callbacks import Profiler

    # Run parameters
    BASE_OUTPUT_DIR = "GS_BUCKET"  # MODIFY with your GS bucket
    MODEL_HANDLE = "hf://google/gemma-2-9b"
    SEQ_LEN = 2048
    PER_DEVICE_BATCH_SIZE = 1

    keras.config.enable_flash_attention()
    
    train_data, eval_data = example_datasets(option="finetune_toy")

    model = KerasHubModel.from_preset(
        MODEL_HANDLE,
        precision="mixed_bfloat16",
        sharding_strategy=PredefinedShardingStrategy(parallelism="fsdp", model="gemma"),
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Create Dataset
    train_ds = TextCompletionDataset(
        source = train_data, 
        tokenizer_handle=MODEL_HANDLE,
        seq_len=SEQ_LEN,
    )

    # Create Dataloader
    train_dataloader = Dataloader(train_ds, per_device_batch_size=PER_DEVICE_BATCH_SIZE)

    # Create Xprof Profiler
    profiler = Profiler(
        mode="xplane",
        output_path=BASE_OUTPUT_DIR,
        max_profile_steps=5,
        skip_first_n_steps=5,
        optional_postfix="kerashub",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        steps=10,
        log_steps_interval=1,
        tensorboard_dir=BASE_OUTPUT_DIR,
        profiler=profiler,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    ray.init()

    num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
    num_tpu_devices = int(ray.cluster_resources()["TPU"])
    num_tpu_hosts = num_tpu_devices // num_chips_per_host

    @ray.remote(resources={"TPU": num_chips_per_host})
    def main():
        import sys

        # Add the MaxText directory to the Python path
        maxtext_dir = "maxtext/MaxText"
        sys.path.append(maxtext_dir)

        run_benchmark()

    ray.get([main.remote() for _ in range(num_tpu_hosts)])

    ray.shutdown()
