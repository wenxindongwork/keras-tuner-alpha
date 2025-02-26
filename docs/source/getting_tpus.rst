.. _getting_tpus:

Getting TPUs
============

Prerequisites
-------------

- A Google Cloud Platform (GCP) account with appropriate billing enabled. 
- Basic understanding of machine learning and command-line tools.
- To set up your GCP Project for TPUs, see `How to set up a GCP account for TPUs <https://cloud.google.com/tpu/docs/setup-gcp-account>`_

Supported TPUs
----------------

Kithara supports all GCP generations and sizes of TPUs. For supported TPUs, see `Cloud TPU Pricing <https://cloud.google.com/tpu/pricing?hl=en>`_.

Choosing a TPU Type
--------------------

For most Kithara workloads, we recommend choosing TPU v5e or Trillium since they have the most availability. Additionally, here are some recipes to map your model and workload to an appropriate TPU generation and chips:

1. Identify the total HBM memory required by your model using the table at the bottom of the page.

2. Identify the per-chip HBM memory of the TPU generation of your interest

   * TPU v4: 32GB per chip
   * TPU v5e: 16GB per chip
   * TPU v5p: 96GB per chip
   * TPU v6e (Trillium): 32GB per chip

3. Calculate how many chip you will need

   * Divide total required HBM by per-chip HBM
   * Example: 35GB HBM required / 32GB per TPU V4 = 1.09 → Use 2 chips minimum

4. Consider Topology Constraints

   * TPUs are arranged in pods with specific slice configurations
   * Common topologies: 2×2×1 (4 chips), 2×2×2 (8 chips), 4×4×4 (64 chips)
   * Choose next largest supported topology that meets your memory needs


**How much total TPU HBM do I need for fine tuning my model?**

.. list-table:: Model Size Requirements
   :header-rows: 1
   :widths: 20 40 40

   * - Model size
     - Full Parameter
     - LoRA
   * - 2b
     - 32 GB
     - 10 GB
   * - 9b
     - 144 GB
     - 40 GB
   * - 27b
     - 432 GB
     - 124 GB
   * - 70b
     - 1,120 GB
     - 322 GB
   * - 405b
     - 6,480 GB
     - 1,863 GB

These approximates assume you are training with the default mixed precision strategy (i.e. model weights loaded in full precision, activations casted to bfloat16).

In case you are wondering how we came up with these numbers :) 

.. tip::

    Total HBM required = Model Size + Optimizer Size + Buffer for intermediates

1. Model Parameters

* Required Memory = (Model Size in Billions) × 4GB
* Example: 2B model requires 8GB

2. Optimizer State

* Full Fine-tuning: 3 × Model Parameter Memory
* Partial Fine-tuning: (% trainable parameters) × Full Optimizer Memory
* Example: 2B model

  * Full fine-tuning: 24GB
  * 5% partial fine-tuning: 1.2GB

3. Buffer for intermediates

* Reserve ~10GB extra HBM memory for intermediate tensors
* Memory usage scales linearly with batch size and sequence length
* If experiencing Out-of-Memory (OOM):

  * Reduce batch size
  * Reduce sequence length
