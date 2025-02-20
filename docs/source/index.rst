.. Kithara documentation master file, created by
   sphinx-quickstart on Wed Nov 20 10:35:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

👋 Welcome to Kithara!
===================================
.. note::

   This project is under active development.

**Kithara** is a lightweight library offering building blocks and recipes for tuning popular open source LLMs like Llama 3 and Gemma 2 on Google TPUs.

Get Started
-----------
.. grid:: 2
    :gutter: 3
    :margin: 0
    :class-container: full-width g-0

    .. grid-item-card:: 🛒 Getting TPUs
        :link: getting_tpus
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `New to TPUs? Here is a guide for determining which TPUs to get and how to get them.`

    .. grid-item-card:: ⚒️ Installation
        :link: installation
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `Quick PiP installation guide.`

    .. grid-item-card:: ✏️ Quickstart
        :link: quickstart
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `Fine-tune a Gemma2 2B model using LoRA.`

    .. grid-item-card:: 📍 Finetuning Guide
        :link: finetuning_guide
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `Step-by-step guide for finetuning your model.`

    .. grid-item-card:: 📈 Scaling up with Ray
        :link: scaling_with_ray
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `Guide for running multihost training with Ray.`

    .. grid-item-card:: 📖 API documentation
        :link: model_api
        :link-type: ref
        :columns: 4
        :padding: 2
        :class-item: g-0

        `API documentation for Kithara library components.`
        
.. toctree::
   :caption: Getting Started
   :hidden:


   🛒 Getting TPUs <getting_tpus>
   ⚒️ Installation <installation>
   ✏️ Quickstart <quickstart>
   📎 Supported Models <models>
   📖 Supported Data Formats <datasets>
   📍 Finetuning Guide <finetuning_guide>
   📈 Scaling up with Ray <scaling_with_ray>
   💡 Troubleshooting <troubleshooting>
   💌 Support and Community <support>

.. toctree::
   :caption: Basics
   :hidden:

   🌵 SFT Example <sft>
   🌵 Continued Pretraining Example <pretraining>
   ✨ LoRA <lora>
   📦 Dataset Packing <packing>
   📚 Managing Large Datasets <ddp>
   🔎 Observability <observability>
   🔖 Checkpointing <checkpointing>
   🚀 Performance Optimizations <optimizations>

.. toctree::
   :caption: API
   :hidden:

   Model <api/kithara.model_api>
   Dataset <api/kithara.dataset_api>
   Trainer <api/kithara.trainer_api>

