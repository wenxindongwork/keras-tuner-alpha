.. _datasets:

Supported Data Formats
======================

The expected flow is always Ray Dataset -> Kithara Dataset -> Kithara Dataloader. 

E.g::
    
    hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ray_dataset = ray.data.from_huggingface(hf_dataset)
    
    kithara_dataset = Kithara.TextCompletionDataset(
        ray_dataset,
        tokenizer_handle="hf://google/gemma-2-2b",
        max_seq_len=8192
    )
    dataloader = Kithara.Dataloader(
        kithara_dataset,
        per_device_batch_size=1
    )

.. tip:: 
    You can load data from your local directory, S3, GCS, ABS, and any other data storage platforms supported by Ray Data.

Following are examples of how to load data from various formats using Ray Dataset. 

HuggingFace Dataset
~~~~~~~~~~~~~~~~~~
Load datasets directly from the Hugging Face Hub. Streaming datasets are supported::

    hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ray_dataset = ray.data.from_huggingface(hf_dataset)
    
JSON and JSONL
~~~~~~~~~~~~~

.. code-block:: python

    ray_dataset = ray.data.read_json("s3://anonymous@ray-example-data/log.json")

CSV
~~~

.. code-block:: python

    ray_dataset = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

Python Dictionary
~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset_items = [{
        "prompt": "What is your name?",
        "answer": "My name is Kithara",
    }  for _ in range(1000)]

    ray_dataset = ray.data.from_items(dataset_items)

Pandas
~~~~~~

.. code-block:: python

    df = pd.DataFrame({
    "food": ["spam", "ham", "eggs"],
    "price": [9.34, 5.37, 0.94]
    })
    ray_dataset = ray.data.from_pandas(df)

Lines of Text 
~~~~~~~~~~~~~
.. code-block:: python

    ray_dataset = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

TFRecords
~~~~~~~~~

.. code-block:: python


    ray_dataset = ray.data.read_tfrecords("s3://anonymous@ray-example-data/iris.tfrecords")

Parquet 
~~~~~~~

.. code-block:: python

    ray_dataset = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

Additional Formats
~~~~~~~~~~~~~~~
Kithara supports all Ray Data formats. For more information:

- Complete list of supported formats: `Ray Data Input/Output <https://docs.ray.io/en/latest/data/api/input_output.html>`_
- Ray Dataset transformation guide: `Ray Data API <https://docs.ray.io/en/latest/data/transforming-data.html>`_

Note: You should handle data cleaning, transformation, shuffling, and splitting using Ray Dataset utilities before passing the data to Kithara Dataset.