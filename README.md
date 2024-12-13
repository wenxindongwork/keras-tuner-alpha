# Keras Tuner Alpha

This repository contains Keras Tuners prototypes.

# Set up

### 1. Clone this repo with submodules

```
git clone --recursive https://github.com/wenxindongwork/keras-tuner-alpha.git
```

If already cloned, add the submodules

```
git submodule update --init --recursive
```

**Troubleshooting**:

If you don't see the maxtext repository after cloning or updating, try

```
git submodule add --force https://github.com/google/maxtext
```

### 2. Install dependencies

```
pip install -r requirements.txt
pip install libtpu-nightly==0.1.dev20240925+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

# Examples

## Tune a HF model

Example of LoRA finetuning gemma2-2b. This script runs on single-host and multi-host environments, on both TPUs and GPUs. For multi-host set up, we included a Ray guide in the next section. 

```
python keras_tuner/examples/singlehost/hf_gemma_example.py
```

## Tune a MaxText model

Example of training a MaxText model. 

```
python keras_tuner/examples/singlehost/maxtext_example.py
```

## Multi-host examples

Following instructions in `orchestration/README.md` to set up a Ray Cluster for running multi-host workloads. Here are example of how to  run tuning tasks once your cluster has been set up.

```
export RAY_ADDRESS="http://127.0.0.1:8265"
python orchestration/multihost/ray/submit_ray_job.py "python examples/multihost/ray/TPU/hf_gemma_example_via_ray.py" --hf-token your_token
```

Similarly, you can run the MaxText example using the following command

```
export RAY_ADDRESS="http://127.0.0.1:8265"
python orchestration/multihost/ray/submit_ray_job.py "python examples/multihost/ray/TPU/maxtext_example_via_ray.py" --hf-token your_token
```

You can early-stop your job using 

```ray job stop ray_job_id```

# Troubleshooting

1. Disk OOM when loading HF model checkpoint 

Attach a disk to your VM and change HF cache directory using the environment variable `export HF_HOME=<your_new_cache_dir>`. Note that you will have to copy your HF token to this new directory as well using `cp .cache/huggingface/token <your_new_cache_dir>/token`. 
