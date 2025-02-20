.. _dataset_api:

Datasets
========

TextCompletionDataset
--------------------

.. py:class:: TextCompletionDataset

   A dataset class for standard text completion tasks.

   :param source: The source Ray dataset containing the text data
   :type source: ray.data.Dataset
   :param tokenizer: The tokenizer instance to use
   :type tokenizer: Optional[AutoTokenizer]
   :param tokenizer_handle: Handle/name of the HF tokenizer to load if not provided. E.g. "hf://google/gemma-2-2b"
   :type tokenizer_handle: Optional[str]
   :param column_mapping: Mapping of source column name to expected column name ("text")
   :type column_mapping: Optional[Dict[str, str]]
   :param model_type: Type of model implementation to use. Supported types: "KerasHub", "MaxText", "auto"
   :type model_type: ModelImplementationType | "auto"
   :param max_seq_len: Maximum sequence length for tokenization (default: 1024)
   :type max_seq_len: int
   :param custom_formatting_fn: A custom formatting function to apply to the raw sample before any other transformation steps
   :type custom_formatting_fn: Optional[callable]
   :param packing: Whether to enable sequence packing
   :type packing: bool

   .. py:method:: to_packed_dataset()

      Converts the current dataset to a PackedDataset for more efficient processing.

      :return: A new PackedDataset instance
      :rtype: PackedDataset

SFTDataset
----------

.. py:class:: SFTDataset(TextCompletionDataset)

   A dataset class for Supervised Fine-Tuning (SFT) tasks.

   :param source: The source Ray dataset containing the training data
   :type source: ray.data.Dataset
   :param tokenizer: HuggingFace tokenizer instance
   :type tokenizer: Optional[AutoTokenizer]
   :param tokenizer_handle: Handle/name of the HF tokenizer to load if not provided. E.g. "hf://google/gemma-2-2b"
   :type tokenizer_handle: Optional[str]
   :param column_mapping: Mapping of source column names to expected column names ("prompt" and "answer")
   :type column_mapping: Optional[Dict[str, str]]
   :param model_type: Type of model implementation to use. Supported types: "KerasHub", "MaxText", "auto"
   :type model_type: ModelImplementationType | "auto"
   :param max_seq_len: Maximum sequence length for tokenization (default: 1024)
   :type max_seq_len: int
   :param custom_formatting_fn: A custom formatting function to apply to the raw sample before any other transformation steps
   :type custom_formatting_fn: Optional[callable]

   .. py:method:: to_packed_dataset()

      Converts the current dataset to a PackedDataset for more efficient processing.

      :return: A new PackedDataset instance
      :rtype: PackedDataset

PackedDataset
------------

.. py:class:: PackedDataset

   A dataset class that packs multiple sequences together on the fly for more efficient processing.

   :param source_dataset: The source dataset containing unpacked sequences
   :type source_dataset: TextCompletionDataset
   :param pad_value: The value to use for padding (default: 0)
   :type pad_value: int

   .. note::
      - Packing must be used with Flash Attention enabled (which should be enabled by default)
      - Packing currently only works for MaxText models
      - Packing does not currently work for DDP training


Example Usage
------------

Here's a simple example of using the TextCompletionDataset::

    dataset = TextCompletionDataset(
        source=ray_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        max_seq_len=512,
    )

For supervised fine-tuning tasks, use the SFTDataset::

    sft_dataset = SFTDataset(
        source=ray_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        column_mapping={"input": "prompt", "output": "answer"},
        max_seq_len=1024
    )

To enable sequence packing for more efficient processing::

    packed_dataset = dataset.to_packed_dataset()