.. _checkpointing:

Checkpointing
=============

Checkpointing allows you to save and restore model states during training. 
Kithara supports saving checkpoints to local directories or Google Cloud Storage buckets. 

Kithara checkpointing is fast -- it uses `Orbax <https://orbax.readthedocs.io/en/latest/>`_ to save checkpoints asynchronously and in a distributed manner.

.. note::
    For efficiency, checkpoints are saved in a non-huggingface format. Use ``model.save_in_hf_format`` to save models in HuggingFace format.

Basic Usage
----------

During Training
^^^^^^^^^^^^^^^^^^^
Provide the ``checkpointer`` arg to the ``Trainer`` class to save checkpoints during training.::

    from Kithara import Checkpointer, Trainer
    # Keeps the latest 5 checkpoints, saving one every 100 steps
    checkpointer = Checkpointer("gs://...", save_interval_steps=100, max_to_keep=5)
    trainer = Trainer(..., checkpointer=checkpointer)
    trainer.train()

Restoring checkpoint after training:::

    # Initialize a random model
    model = MaxTextModel.from_random(
        model_name="gemma2-9b",
        seq_len=4096,
        per_device_batch_size=1,
        precision="mixed_bfloat16",
    )

    # Restore from specific checkpoint
    checkpointer = Checkpointer(
        "gs://your_bucket/your_model_name/ckpt/20",  # Step 20
        model=model
    )
    checkpointer.load()

    model.generate(...)

As Standalone Utility
^^^^^^^^^^^^^^^^^^^^^

You can also use the ``Checkpointer`` as a standalone utility to save and load checkpoints outside of the ``Trainer`` class. 

.. code-block:: python

    model = kithara.MaxTextModel.from_preset("hf://google/gemma2-2b")

    # Attach the checkpointer to the model
    checkpointer = Checkpointer("gs://...", model)

    # Save checkpoint - checkpoints need to be numbered sequentially
    checkpointer.save(0, blocking=True)

    # Load latest checkpoint back to the model
    checkpointer.load()

