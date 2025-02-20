.. _lora:

LoRA: Low-Rank Adaptation Guide
===============================

What is LoRA?
------------
LoRA (Low-Rank Adaptation) is a technique that makes fine-tuning large language models much more efficient. Instead of updating all model parameters during fine-tuning, LoRA:

- Keeps the original model weights frozen
- Adds small trainable matrices to specific model layers
- Drastically reduces the number of parameters that need to be updated

For example, you can fine-tune a 9B parameter model by training only about 100M parameters (roughly 1% of the original size) or less. 

Simple Usage
------------

To train your model with LoRA, you don't have to worry about changing anything in your training script other specifying the ``lora_rank`` arg.
LoRA will be applied to the q_proj and v_proj layers.::

    from Kithara import KerasHubModel
    
    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b",
        lora_rank=16  # <-- One line toggle
    )

Saving LoRA Models
----------
You have three options for saving models trained with LoRA:

1. Save Only LoRA Adapters
~~~~~~~~~~
Since the base model is left unchanged, you can save just the LoRA Adapters::

    model.save_in_hf_format(
        local_dir_or_gs_bucket,
        only_save_adapters=True
    )

2. Save Base Model and Adapters Separately
~~~~~~~~~~
In case you want to save the base model as well. ::

    model.save_in_hf_format(
        local_dir_or_gs_bucket,
        only_save_adapters=False,
        save_adapters_separately=True
    )

3. Save Merged Model
~~~~~~~~~~
Creates a single model combining base weights and adaptations::

    model.save_in_hf_format(
        local_dir_or_gs_bucket,
        save_adapters_separately=False
    )

Load LoRA Models back into HuggingFace 
----------

To load a model trained with LoRA back into HuggingFace, you can use the following code, where ``lora_dir`` and ``model_dir`` stores the weights saved by Kithara.::

    # Load adapters separately
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_model.load_adapter(lora_dir)
    
    # Load merged model
    hf_model = AutoModelForCausalLM.from_pretrained(model_dir)


Next Steps
----------
For a complete example of using LoRA with supervised fine-tuning, see the :doc:`SFT + LoRA guide <sft>`.