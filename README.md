# Kithara - Easy Finetuning on TPUs

[![PyPI](https://img.shields.io/pypi/v/kithara)](https://pypi.org/project/kithara/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/wenxindongwork/keras-tuner-alpha/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/wenxindongwork/keras-tuner-alpha)](https://github.com/wenxindongwork/keras-tuner-alpha/commits/main)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://kithara.readthedocs.io/en/latest/)

<div align="center">

<a href="https://kithara.readthedocs.io/en/latest"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png">
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png">
<img alt="kithara logo" src="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png" height="110" style="max-width: 100%;">
</picture></a>

</div>

## 👋 Overview

Kithara is a lightweight library offering building blocks and recipes for tuning popular open source LLMs on Google TPUs. 

It provides:

- **Frictionless scaling**: Distributed training abstractions intentionally built with simplicity in mind.
- **Multihost training support**: Integration with Ray, GCE and GKE.
- **Async, distributed checkpointing**: Multi-host & Multi-device checkpointing via Orbax.
- **Distributed, streamed dataloading**: Per-process, streamed data loading via Ray.data.
- **GPU/TPU fungibility**: Same code works for both GPU and TPU out of the box. 
- **Native integration with HuggingFace**: Tune and save models in HuggingFace format.

**New to TPUs?**

Using TPUs provides significant advantages in terms of performance, cost-effectiveness, and scalability, enabling faster training times and the ability to work with larger models and datasets. Check out our onboarding guide to TPUs. [TODO: add link]

## 🔗 **Key links and resources**
|                                   |                                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 📚 **Documentation**              | [Read Our Docs](https://kithara.readthedocs.io/en/latest/)                                                                  |
| 💾 **Installation**               | [New to TPUs? We got you covered](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-installation-instructions) |
| ✏️ **Get Started**               | [Quick start examples](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-installation-instructions) |
| 🌟 **Supported Models**           | [List of Models](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-supported-models)                           |
| 🌐 **Supported Algorithms**       | [List of Algorithms](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-supported-models)                       |
| ⌛️ **Performance Optimizations** | [Our Memory and Throughput Optimizations](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-supported-models)  |
| 📈 **Scaling up**                 | [Guide for Tuning Large Models](https://github.com/wenxindongwork/keras-tuner-alpha/tree/main#-installation-instructions)   |
<!-- 
## 💾 Installation Instructions

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

Following instructions in `ray/README.md` to set up a Ray Cluster for running multi-host workloads. Here are examples of how to run the SFT LoRA example once your cluster has been set up.

```
python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
```

Similarly, you can run the full parameter finetuning example using the following command

```
python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/full_finetuning_example.py" --hf-token your_token
```

You can early-stop your job using

`ray job stop ray_job_id`

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
   ``` -->
