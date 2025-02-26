.. _packing:

Dataset Packing
===============

What is Dataset Packing?
-----------------------

Dataset packing combines multiple short sequences into longer ones during training, reducing overall training time when your data consists of sequences shorter than the model's maximum sequence length.

When to Use?
------------
Use Packing will not hurt performance, so you should use it whenever possible. However, Packing is only supported for MaxText models.

How to Use?
---------

Convert your dataset to a packed format using ``.to_packed_dataset()``:

.. code-block:: python

    dataset = Kithara.TextCompletionDataset(
        ray_dataset,
        tokenizer="hf://google/gemma-2-2b",
        max_seq_len=4096
    ).to_packed_dataset()

Important Notes
-------------

* No other changes to your training script are needed
* Loss calculation remains accurate as Kithara uses per-token loss
* Adjust your learning rate and training steps since packing reduces the effective number of samples in a batch