import ray

"""
Benchmark for training MaxText models via Kithara.  

This benchmark script runs multi-host training with the specified MaxText model 

Metrics: step time, Tokens/s/device
Artifact: Tensorboard, Xplane (Uploaded to BASE_OUTPUT_DIR)

Purpose: Compare native MaxText performance against performance of MaxText via Kithara. 

Launch Script: python orchestration/multihost/ray/submit_ray_job.py "python benchmark/kerashub.py"

TODO: Launch benchmarks via YAML config.
"""


def run_benchmark():
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    from examples.example_datasets import example_datasets
    from keras_tuner.model.models.kerashub.keras_hub_model import KerasHubModel
    from keras_tuner.dataset import Dataloader
    from keras_tuner.preprocessor import PretrainingPreprocessor
    from keras_tuner.trainer import Trainer
    from keras_tuner.model.sharding import PredefinedShardingStrategy
    from keras_tuner.observability import Profiler

    # Run parameters
    BASE_OUTPUT_DIR = "GS_BUCKET" #MODIFY with your GS bucket
    MODEL_HANDLE = "hf://google/gemma-2-9b"
    SEQ_LEN = 2048
    PER_DEVICE_BATCH_SIZE = 1


    train_ds, eval_ds = example_datasets(option="finetune_toy")
    
    model = KerasHubModel(
        model_handle=MODEL_HANDLE,
        precision="mixed_bfloat16",
        sharding_strategy=PredefinedShardingStrategy(
            parallelism="fsdp", model="gemma"
        ),
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Create Preprocessor
    preprocessor = PretrainingPreprocessor(
        tokenizer_handle=MODEL_HANDLE,
        seq_len=SEQ_LEN,
        model_type="maxtext",
    )

    # Create Dataloader
    train_dataloader = Dataloader(
        train_ds, per_device_batch_size=PER_DEVICE_BATCH_SIZE
    )

    # Create Xprof Profiler 
    profiler = Profiler(
        mode = "xplane",
        output_path=BASE_OUTPUT_DIR,
        max_profile_steps=5,
        skip_first_n_steps=5,
        optional_postfix="kerashub"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        preprocessor=preprocessor,
        train_dataloader=train_dataloader,
        steps=10,
        log_steps_interval=1,
        tensorboard_dir=BASE_OUTPUT_DIR,
        profiler =profiler
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
