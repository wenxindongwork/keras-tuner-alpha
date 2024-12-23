import ray

"""
Benchmark for training MaxText models via Kithara.  

This benchmark script runs multi-host training with the specified MaxText model 

Metrics: step time, Tokens/s/device
Artifact: Tensorboard, Xplane (Uploaded to BASE_OUTPUT_DIR)

Purpose: Compare native MaxText performance against performance of MaxText via Kithara. 

Launch Script: python ray/submit_job.py "python benchmark/maxtext_via_kithara.py"

TODO: Launch benchmarks via YAML config.
"""

def run_benchmark():

    import os

    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    from examples.example_datasets import example_datasets
    from kithara.model.maxtext import MaxTextModel
    from kithara.dataset import Dataloader
    from kithara.preprocessor import PretrainingPreprocessor
    from kithara.trainer import Trainer
    from kithara.callbacks import Profiler

    # Run parameters
    BASE_OUTPUT_DIR = "GS_BUCKET"  # MODIFY with your GS bucket
    MODEL_NAME = "gemma2-9b"
    SEQ_LEN = 2048
    PER_DEVICE_BATCH_SIZE = 1

    train_ds, eval_ds = example_datasets(option="finetune_toy")

    # Create a randomly initialized MaxText Model
    model = MaxTextModel(
        model_name=MODEL_NAME,
        seq_len=SEQ_LEN,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
        maxtext_config="remat_policy=minimal",
    )

    # Create Keras optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    # Create Preprocessor
    preprocessor = PretrainingPreprocessor(
        tokenizer_handle="hf://google/gemma-2-2b",
        seq_len=SEQ_LEN,
        model_type="maxtext",
    )

    # Create Dataloader
    train_dataloader = Dataloader(train_ds, per_device_batch_size=PER_DEVICE_BATCH_SIZE)

    # Create Xprof Profiler
    profiler = Profiler(
        mode="xplane",
        output_path=BASE_OUTPUT_DIR,
        max_profile_steps=5,
        skip_first_n_steps=5,
        optional_postfix="maxtext_via_kithara",
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
