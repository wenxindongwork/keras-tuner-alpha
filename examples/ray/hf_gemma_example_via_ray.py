import ray

ray.init()

num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e


@ray.remote(resources={"TPU": num_chips_per_host})
def main():
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    from examples.hf_gemma_example import run_workload

    print("Running workload")
    run_workload()


num_tpu_devices = int(ray.cluster_resources()["TPU"] / num_chips_per_host)

ray.get([main.remote() for _ in range(num_tpu_devices)])

ray.shutdown()
