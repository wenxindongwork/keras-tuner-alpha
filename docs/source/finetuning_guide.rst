.. _finetuning_guide:

Finetuning Guide
===================

Ready to fine-tune your base model? Let's break it down into simple steps.

1. Pick Your Model 
----------------

We've got two options for you:

MaxText
~~~~~~~

For when you want to do full-parameter fine-tuning::

    from kithara import MaxTextModel
    
    model = MaxTextModel.from_preset(
        "hf://google/gemma-2-2b",  # HuggingFace model
        seq_len=8192, # Seq len and batch size need to be specified up front
        per_device_batch_size=1, 
        precision="mixed_bfloat16",  # Default precision
        scan_layers=True  # Set to True for models <9B for performance gain
    )

KerasHub
~~~~~~~~

For when you want to fine-tune with LoRA::

    from kithara import KerasHubModel
    
    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b",  # HuggingFace model
        precision="mixed_bfloat16",  # Default precision
        lora_rank=16  # Applied to q_proj and v_proj
    )

Quick tips:

- Always start model handles with ``hf://`` when loading from HuggingFace - so we know you are not loading from local directory 😀
- ``mixed_bfloat16`` is your friend - it's memory-friendly! It loads model weights in full precision and casts activations to bfloat16.
- Check out our :doc:`model garden <models>` for more options
- Want to save your model back into HuggingFace format? Simply do ``model.save_in_hf_format(local_dir_or_gs_bucket)``.

2. Prepare Your Data
-------------------

Getting your data ready is super easy with Ray::

    from datasets import load_dataset
    import ray
    
    # Load streaming dataset
    hf_dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )
    ray_dataset = ray.data.from_huggingface(hf_dataset)

Now, let's make it Kithara-friendly. Create a Kithara Dataset and Dataloader::

    # Tokenize dataset
    dataset = Kithara.TextCompletionDataset(
        ray_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        max_seq_len=8192
    )
    
    # Create dataloader
    dataloader = Kithara.Dataloader(
        dataset,
        per_device_batch_size=1
    )

The expected flow is always Ray Dataset -> Kithara Dataset -> Kithara Dataloader.

Quick Tips:

- Need a specific column? Use column_mapping={"text": "your_column"}
- Browse our :doc:`supported data formats <datasets>`. 


3. Choose Algorithm
------------------

- Continued pretraining: Train your base model with large datasets to expand its knowledge
- Supervised finetuning + LoRA: Quickly and efficiently tune your model using labeled examples


[Algorithms to be added]

- DPO 

4. Select Hardware
-----------------

If you haven't done so, check out :doc:`Getting TPUs <getting_tpus>` to get your TPUs ready. 

5. Run Workflow
--------------

Follow the examples below to run your workflow. 

