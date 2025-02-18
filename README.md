# Kithara

A LLM Post-training Library for TPUs and GPUs. 

# Set up

Kithara requires `Python>=3.11`.

### On CPU 

``` 
pip install kithara[cpu] 
```

### On TPU 

``` 
pip install kithara[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url https://download.pytorch.org/whl/cpu 
```
### On GPU 

``` 
pip install kithara[gpu]
```

# Examples

## SFT with LoRA 

Example of LoRA finetuning gemma2-2b. This script runs on single-host and multi-host environments, on both TPUs and GPUs. For multi-host set up, we included a Ray guide in the next section. 

```
python kithara/examples/singlehost/sft_lora_example.py
```

## Full parameter finetuning

Example of training a MaxText model. 

```
python kithara/examples/singlehost/full_finetuning_example.py
```

## Multi-host examples

Following instructions in `ray/README.md` to set up a Ray Cluster for running multi-host workloads. Here are example of how to  run tuning tasks once your cluster has been set up.

First copy the example script in the `examples/multihost` folder to a new folder on your your local machine, let's call it `ray_workdir`.
Then, use the `kithara multihost` CLI` to run the script on your Ray Cluster. 


```
cd ray_workdir
kithara multihost sft_lora_example.py --hf-token your_token
```

Similarly, you can run the full parameter finetuning example using the following command

```
cd ray_workdir
kithara multihost full_finetuning_example.py --hf-token your_token
```

You can early-stop your job using 

```ray job stop ray_job_id```

# Troubleshooting

1. Disk OOM when loading HF model checkpoint 

    First try emptying your cache by running the following code on your Ray Cluster.

    ```
    import shutil
    shutil.rmtree("/home/ubuntu/.cache/huggingface/hub/", ignore_errors=True)
    shutil.rmtree("/home/ubuntu/.keras/models", ignore_errors=True)
   ```

    If you are using a single VM, the path may be different.

    ```
    import shutil
    shutil.rmtree("~/.cache/huggingface/hub/", ignore_errors=True)
    shutil.rmtree("~/.keras/models", ignore_errors=True)
    ```

    If emptying the cache still doesn't help, try attaching a disk to your VM and change HF cache directory using the environment variable `export HF_HOME=<your_new_cache_dir>`. 
    
    You may have to copy your HF token to this new cache directory with `cp .cache/huggingface/token <your_new_cache_dir>/token`. 

2. Permission denied error when uploading checkpoint to GCS 

    First verify your current authentication :

    ```
    gcloud auth list
    gsutil ls gs://your_bucket
    ```

    For your Python code, you likely need to ensure you're using the same credentials.

    ```
    gcloud auth application-default login
    ```

3. jaxlib.xla_extension.XlaRuntimeError errors

    Try uninstall and reinstalling `jax` and `jaxlib`

    ```
    pip uninstall jax jaxlib
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install libtpu-nightly==0.1.dev20250128+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

