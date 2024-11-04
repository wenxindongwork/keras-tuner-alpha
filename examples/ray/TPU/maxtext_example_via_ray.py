import ray

ray.init()

num_chips_per_host = 4  # 4 for v4 and v5, 8 for v4e and v5e
num_tpu_devices = int(ray.cluster_resources()["TPU"])
num_tpu_hosts = num_tpu_devices // num_chips_per_host

print(f"{num_tpu_devices=}")
print(f"{num_tpu_hosts=}")

@ray.remote(resources={"TPU": num_chips_per_host})
def main():
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    import sys

    # Add the MaxText directory to the Python path
    maxtext_dir = "maxtext/MaxText"
    sys.path.append(maxtext_dir)

    from examples.maxtext_example import run_workload
    
    print("Running workload")
    run_workload()


ray.get([main.remote() for _ in range(num_tpu_hosts)])

ray.shutdown()
