# Test your GPU Ray cluster set up with 
# ray job submit --working-dir . -- python3 ray/GPU/find_devices.py 

import ray

ray.init()

num_chips_per_host = 4 
num_gpu_devices = int(ray.cluster_resources()["GPU"])
print(f"{num_gpu_devices=}")

@ray.remote(num_gpus=num_chips_per_host)
def main():
    import jax 
    print(jax.devices())

ray.get([main.remote() for _ in range(num_gpu_devices)])

ray.shutdown()
