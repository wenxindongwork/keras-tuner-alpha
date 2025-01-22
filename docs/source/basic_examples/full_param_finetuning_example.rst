Full parameter finetuning
========================= 

In this example, we take the ``google/Gemma-2-9b`` pre-trained model from HuggingFace and fine-tune it on a custom dataset. 
Just like any other Kithara workload, this example works on **both TPU and GPU**, and can be run on both single-host and multi-host environments.

Key Kithara features illustrated in this example include

#. **Native HF support** 
#. **Extensive dataset format support** 
#. **Simple Trainer API** 
#. **Fast Model Checkpointing** 


Ready? Let's get started!


**Step 1** 

Create a ``Gemma2-9b`` model and load the pre-trained weights from HuggingFace. 
"mixed_bfloat16" is a good choice for TPU training, it 
refers to using bfloat16 for activations and float32 for weights.

.. code-block:: python

    from kithara import MaxTextModel

    model = MaxTextModel.from_preset(
        preset_handle="hf://google/gemma-2-9b",
        seq_len=4096,
        per_device_batch_size=1,
        precision="mixed_bfloat16",
    )

**Step 2** 

Create the ``Optimizer``. Kithara supports all Keras optimizers, here we use AdamW.  

.. code-block:: python
    
    import keras 

    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )

**Step 3** 

Create toy data. Kithara leverages ``ray.data`` to load and transform datasets. 
Please check out ``ray.data`` for the extensive list of supported dataset formats, 
including JSON, JSONL, CSV, BigQuery and more, as well as the extensive list 
of supported dataset sources including local files, GCS, S3, and Azure. 

.. code-block:: python

    import ray

    dataset_items = [
        {"text": f"{i} What is your name? My name is Kithara."} for i in range(1000)
    ]
    dataset = ray.data.from_items(dataset_items)
    train_data, eval_data = dataset.train_test_split(test_size=500)

**Step 4**

Create the ``Dataset``. Kithara Dataset takes in as source a Ray dataset, and 
returns tokenized, ready-to-train model inputs. ``TextCompletionDataset`` is used for pretraining tasks, 
similarly  ``SFTDataset`` is used for SFT tasks.

.. code-block:: python

    from kithara import TextCompletionDataset

    train_dataset = TextCompletionDataset(
        source = train_data,
        tokenizer_handle="hf://google/gemma-2-9b",
        max_seq_len=4096,
    )

    eval_dataset = TextCompletionDataset(
        source = eval_data,
        tokenizer_handle="hf://google/gemma-2-9b",
        max_seq_len=4096,
    )

**Step 5** 

Create the ``Dataloader``. Kithara ``Dataloader`` offers **streamed** and **multi-host distributed** dataset loading. However, 
in this example, we are showing the undistributed version for simplicity.

.. code-block:: python

    from kithara import Dataloader

    train_dataloader = Dataloader(
        train_dataset,
        per_device_batch_size=1 
    )
    eval_dataloader = Dataloader(
        eval_dataset,
        per_device_batch_size=1 
    )

**Step 6** 

Create a ``Checkpointer``. Kithara ``Checkpointer`` offers fast, distributed, asynchronously model 
saving to either a local or a cloud storage location. Note that for efficiency, checkpoints are not 
saved in HuggingFace format. ``model.save_in_hf_format`` is used for that purpose.

.. code-block:: python

    from kithara import Checkpointer

    checkpointer = Checkpointer(
        "gs://your_bucket/your_model_name/ckpt/",
        model=model,
        save_interval_steps=20,
        max_to_keep=5
    )

**Step 7** 

Initialize ``Trainer``. ``Trainer`` is the main class that orchestrates the training process.
Optionally, you can pass in a tensorboard directory to log training metrics, and access the tensorboard UI via 
``tensorboard --logdir gs://your_bucket/your_model_name/tensorboard/``.

.. code-block:: python

    from kithara import Trainer

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        steps=200,
        eval_steps_interval=100,
        log_steps_interval=1,
        max_eval_samples=50,
        checkpointer=checkpointer,
        tensorboard_dir="gs://your_bucket/your_model_name/tensorboard/",
    )

**Step 8**

Generate text before training.

.. code-block:: python

    pred = trainer.generate("What is your name?", skip_special_tokens=True)
    print(f"Before training, model generated {pred}")

**Step 9**

Kick off training.

.. code-block:: python

    trainer.train()

**Step 10**

Generate text after training.

.. code-block:: python

    pred = trainer.generate("What is your name?", skip_special_tokens=True)
    print(f"Tuned model generated {pred}")

**Step 11**

Save model in HuggingFace format.

.. code-block:: python

    model.save_in_hf_format("gs://your_bucket/your_model_name/final/")

**Step 12**

Restore model from saved, HuggingFace format model back into Kithara. 

.. code-block:: python

    from kithara import MaxTextModel

    model = MaxTextModel.from_preset(
        preset_handle="gs://your_bucket/your_model_name/final/",
        seq_len=4096,
        per_device_batch_size=1,
        precision="mixed_bfloat16",
    )

**Step 13**

Alternatively, restore model back into a HuggingFace ``AutoModelForCausalLM`` model.

.. code-block:: python

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gs://your_bucket/your_model_name/final/")

**Step 14**

Alternatively, restore model from a checkpoint saved during training (see Step 6). 

.. code-block:: python

    model = MaxTextModel.from_random(
        model_name="gemma2-9b",
        seq_len=4096,
        per_device_batch_size=1,
        precision="mixed_bfloat16",
    )

    # E.g. Restore model from checkpoint at step 20
    checkpointer = Checkpointer(
        "gs://your_bucket/your_model_name/ckpt/20",
        model=model
    )

    checkpointer.load()

    trainer = ... 
    trainer.train()


