# Kithara - Easy Finetuning on TPUs

[![PyPI](https://img.shields.io/pypi/v/kithara)](https://pypi.org/project/kithara/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/wenxindongwork/keras-tuner-alpha/pulls)
[![GitHub last commit](https://img.shields.io/github/last-commit/wenxindongwork/keras-tuner-alpha)](https://github.com/wenxindongwork/keras-tuner-alpha/commits/main)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://kithara.readthedocs.io/en/latest/)

<div align="center">

<a href="https://kithara.readthedocs.io/en/latest"><picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png">
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png">
<img alt="kithara logo" src="https://raw.githubusercontent.com/wenxindongwork/keras-tuner-alpha/documentation-v2/docs/images/kithara_logo_with_green_bg.png" height="150" style="max-width: 100%;">
</picture></a>

</div>

## 👋 Overview

Kithara is a lightweight library offering building blocks and recipes for tuning popular open source LLMs including Gemma2 and Llama3 on Google TPUs. 

It provides:

- **Frictionless scaling**: Distributed training abstractions intentionally built with simplicity in mind.
- **Multihost training support**: Integration with Ray, GCE and GKE.
- **Async, distributed checkpointing**: Multi-host & Multi-device checkpointing via Orbax.
- **Distributed, streamed dataloading**: Per-process, streamed data loading via Ray.data.
- **GPU/TPU fungibility**: Same code works for both GPU and TPU out of the box. 
- **Native integration with HuggingFace**: Tune and save models in HuggingFace format.

**New to TPUs?**

Using TPUs provides significant advantages in terms of performance, cost-effectiveness, and scalability, enabling faster training times and the ability to work with larger models and datasets. Check out our onboarding guide to [getting TPUs](https://kithara.readthedocs.io/en/latest/getting_tpus.html).

## 🔗 **Key links and resources**
|                                   |                                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 📚 **Documentation**              | [Read Our Docs](https://kithara.readthedocs.io/en/latest/)                                                                  |
| 💾 **Installation**               | [Quick Pip Install](https://kithara.readthedocs.io/en/latest/installation.html) |
| ✏️ **Get Started**               | [Intro to Kithara](https://kithara.readthedocs.io/en/latest/quickstart.html) |
| 🌟 **Supported Models**           | [List of Models](https://kithara.readthedocs.io/en/latest/models.html)                           |
| 🌐 **Supported Datasets**       | [List of Data Formats](https://kithara.readthedocs.io/en/latest/datasets.html)                       |
| 🌵 **SFT + LoRA Example**       | [SFT + LoRA Example](https://kithara.readthedocs.io/en/latest/datasets.html)                       |
| 🌵 **Continued Pretraining Example**       | [Continued Pretraining Example](https://kithara.readthedocs.io/en/latest/datasets.html)                       |
| ⌛️ **Performance Optimizations** | [Our Memory and Throughput Optimizations](https://kithara.readthedocs.io/en/latest/optimizations.html)  |
| 📈 **Scaling up**                 | [Guide for Tuning Large Models](https://kithara.readthedocs.io/en/latest/scaling_with_ray.html)   |
