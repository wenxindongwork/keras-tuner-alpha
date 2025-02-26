.. _finetuning_guide:

Finetuning Guide
===================

Ready to fine-tune with your own data? Let's break it down into simple steps.

1. Pick Your Model 
------------------

Kithara got two options for you:

MaxText
~~~~~~~

For when you want to do full-parameter fine-tuning::

    from kithara import MaxTextModel
    
    model = MaxTextModel.from_preset(
        "hf://google/gemma-2-2b",
        seq_len=8192, # Seq len and batch size need to be specified up front
        per_device_batch_size=1
    )

KerasHub
~~~~~~~~

For when you want to fine-tune with LoRA::

    from kithara import KerasHubModel
    
    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b",
        lora_rank=16  # Applied to q_proj and v_proj
    )

Quick tips:

- Always start model handles with ``hf://`` when loading from HuggingFace - so we know you are not loading from local directory 😀
- The default precision ``mixed_bfloat16`` is your friend - it's memory-friendly! It loads model weights in full precision and casts activations to bfloat16.
- Check out our :doc:`model garden <models>` for supported architectures
- Want to save your model? Simply do ``model.save_in_hf_format(local_dir_or_gs_bucket)``
- Check out :doc:`Model API <api/kithara.model_api>` documentation

2. Prepare Your Data
--------------------

Kithara supports HuggingFace Datasets as well as Ray Datasets.

HuggingFace Dataset::

   from datasets import load_dataset
   hf_dataset = load_dataset("allenai/c4","en",split="train",streaming=True),
   dataset = Kithara.TextCompletionDataset(
      hf_dataset,
      tokenizer_handle="hf://google/gemma-2-2b",
      max_seq_len=8192
   )

Ray Dataset::

   import ray
   ray_dataset = ray.data.read_json("s3://anonymous@ray-example-data/log.json")
   dataset = Kithara.TextCompletionDataset(
      ray_dataset,
      tokenizer_handle="hf://google/gemma-2-2b",
      max_seq_len=8192
   )
      
Now create a Kithara Dataloader to batchify your dataset.::

    # Create dataloader
    dataloader = Kithara.Dataloader(
        dataset,
        per_device_batch_size=1
    )

The expected flow is always Ray/HF Dataset -> Kithara Dataset -> Kithara Dataloader.

Quick Tips:

- Your global batch size is always `per_device_batch_size` * `number of devices (chips)`.
- Check out :doc:`supported data formats <datasets>` (CSV, JSON, etc.)
- Check out :doc:`Dataset API <api/kithara.dataset_api>` documentation.


3. Choose Algorithm
-------------------

- **Continued pretraining**: Train your base model with large datasets to expand its knowledge
- **Supervised finetuning**: Quickly and efficiently tune your model using labeled examples

Check out these examples to get started.

- :doc:`🌵 Continued Pretraining Example <pretraining>`
- :doc:`🌵 SFT+LoRA Example <sft>`

4. Select Hardware
------------------

If you haven't done so, check out :doc:`Getting TPUs <getting_tpus>` to get your TPUs ready.

If your TPU topology has multiple hosts, and you are not familiar with distributed training, 
we recommend you follow the :doc:`Scaling up with Ray <scaling_with_ray>` guide to set up a 
Ray Cluster so that you can run multihost jobs. 

