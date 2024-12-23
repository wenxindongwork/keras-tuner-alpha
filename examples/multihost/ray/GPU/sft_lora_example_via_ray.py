import ray

ray.init()

num_chips_per_host = 4 
num_gpu_devices = int(ray.cluster_resources()["GPU"])
print(f"{num_gpu_devices=}")

@ray.remote(num_gpus=num_chips_per_host)
def main():
    from huggingface_hub import login
    import os

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    
    from examples.singlehost.sft_lora_example import run_workload

    print("Running workload")
    run_workload()


ray.get([main.remote() for _ in range(num_gpu_devices)])

ray.shutdown()
