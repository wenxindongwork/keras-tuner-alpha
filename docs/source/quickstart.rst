.. _quickstart:

Quick Start
===========

This guide demonstrates how to fine-tune a Gemma2 2B model using LoRA.


Setup
-----
Import required packages::

    import os
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    import ray
    from transformers import AutoTokenizer
    from kithara import (
        KerasHubModel,
        Dataloader,
        Trainer,
        SFTDataset,
    )

Quick Usage
----------

1. Create the Model::

    model = KerasHubModel.from_preset(
        "hf://google/gemma-2-2b",
        precision="mixed_bfloat16",
        lora_rank=4,
    )

.. tip::
    New to HuggingFace? First `apply access <https://huggingface.co/google/gemma-2-2b>`_ to the HuggignFace model, create an access token, and setting the ``HF_TOKEN`` environment variable.
    
2. Initialize Tokenizer::

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

3. Prepare Dataset::

    dataset_items = [
        {
            "prompt": "What is your name?",
            "answer": "My name is Kithara",
        }
        for _ in range(1000)
    ]
    dataset = ray.data.from_items(dataset_items)
    train_ds, eval_ds = dataset.train_test_split(test_size=500)

4. Create Dataset and Optimizer::

    train_dataset = SFTDataset(
        train_ds,
        tokenizer=tokenizer,
        max_seq_len=4096,
    ).to_packed_dataset()
    
    eval_dataset = SFTDataset(
        eval_ds,
        tokenizer=tokenizer,
        max_seq_len=4096,
    ).to_packed_dataset()
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=0.01
    )

5. Create Dataloaders::

    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=1
    )
    
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=1
    )

6. Initialize and Run Trainer::

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=200, # You can also use epochs instead of steps
        eval_steps_interval=10,
        max_eval_samples=50,
        log_steps_interval=10,
    )
    
    trainer.train()

7. Test the Model::

    pred = model.generate(
        "What is your name?",
        max_length=30,
        tokenizer=tokenizer,
        return_decoded=True
    )
    print("Tuned model generates:", pred)

Running the Script on Single Host or Multi-host
------------------------------------------------

The script can also be found on `Github <https://github.com/wenxindongwork/keras-tuner-alpha/blob/main/examples/singlehost/sft_lora_example.py>`_.

Single host::

    python examples/singlehost/sft_lora_example.py

Multi-host via Ray::

    python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token
