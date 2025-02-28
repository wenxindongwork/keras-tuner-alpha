.. _getting_tpus:

Getting TPUs
============

Prerequisites
-------------

- A Google Cloud Platform (GCP) account with appropriate billing enabled. 
- Basic understanding of machine learning and command-line tools.
- To set up your GCP Project for TPUs, see `Set up a GCP Account <https://cloud.google.com/tpu/docs/setup-gcp-account>`_

Supported TPUs
----------------

Kithara supports all GCP generations and sizes of TPUs. For supported TPUs, see `Cloud TPU Pricing <https://cloud.google.com/tpu/pricing?hl=en>`_.


Requesting Capacity
----------------

Please review the following steps before provisioning capacity (creating VMs):

1. Familiarize yourself with key concepts
2. Choose a TPU type
3. Choose a usage mode
4. Choose a type of capacity
5. Request quota 

After completing these steps, you can provision capacity. Creating VMs without completing these steps first may lead to errors.

.. tip::

    If you want to skip this section, to get started quickly, 64 chips of Trillium (TPU v6e) with :doc:`DWS (Flex Start)<dws_flex_start>` mode in any supported `zone <https://cloud.google.com/tpu/pricing?hl=en>`_ is a good starting point.  To see the size of models you can tune with this capacity, use `this calculator <https://v0-calculator-for-tpu.vercel.app/>`_. Instructions to create VMs are :doc:`here <create_vm>`.

    Later, you can tune these parameters using the rest of the instructions in this section for production workloads.

1.  Familiarize yourself with key concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TPU Chip and Cores**

A TPU chip typically (but not always) consists of two TPU cores which share memory and can be thought of as one large accelerator with twice the compute capacity (FLOPs). Inference-optimized chips like the TPU v5e and Trillium (TPU v6e) only have one TPU core per chip.

.. figure:: https://jax-ml.github.io/scaling-book/assets/img/cores.png
   :align: center
   :width: 50%
   
   A TPU Chip (orange box) with two cores (white boxes)


Where are these terms used typically?

*   **Chips:** To define pricing. E.g: price-per-chip-hour
*   **Cores:** To create VMs. E.g: You may request a v5e-8 which requests 8 cores (for v5e, this is also equal to 8 chips). Or, you may request v5p-8 which creates a VM with 8 cores (4 chips).


.. table::
   :width: 100%

   ==========  ==========
   TPU Type    Number of cores
   ==========  ==========
   TPU v4p     2x per chip
   TPU v5p     2x per chip
   TPU v5e     1x per chip
   Trillium    1x per chip
   ==========  ==========
  
**Slices and Topology**


You may see the terms topology and slice in `Google Cloud TPU documentation <https://cloud.google.com/tpu/docs>`_. Topology refers to TPU networking, i.e., how the chips are connected to each other with high-speed inter-chip interconnects (ICI). A collection of chips connected to each other directly with ICI is called a slice.

The performance-optimized “p-series” (v4, v5p) are interconnected in a 3D topology. The efficient “e-series” (v5e, v6e) are interconnected in a 2D topology.

.. figure:: https://jax-ml.github.io/scaling-book/assets/img/subslices.png
   :align: center
   
   A slice with 3D topology. Left: 2x2x1 topology; Right-top: 2x2x2; Right-bottom: 2x2x4


.. figure:: https://jax-ml.github.io/scaling-book/assets/img/more-subslices.png
   :align: center


   A slice with 2D topology. Right-top: 4x8; Right-bottom: 4x4


.. tip:: 

   For most users, the simplest way to get started is to simply create a VM specifying the number of cores you want to use (e.g: v5e-8) without specifying a specific slice shape. This will automatically create the largest slice possible.

   **Recap**: A v5p-128 slice = 128 cores = 64 chips = 4x4x4 topology = 4x4x4 slice

If you are interested, you can read more about TPU networking `here <https://jax-ml.github.io/scaling-book/tpus/#tpu-networking>`_.


2. Choose a TPU Type
~~~~~~~~~~~~~~~~~~~~~~~~

.. tip:: For most Kithara workloads, we recommend choosing TPU v5e or Trillium since they have the most availability. 

Additionally, here are some recipes to map your model and workload to an appropriate TPU generation and chips:

.. note:: 
   You can use this `calculator <https://v0-calculator-for-tpu.vercel.app/>`_ if you do not want to identify the right TPU type manually using the formulas below.

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



3.  Choose a usage mode (consumption type)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: Cloud TPU Usage Modes 

  *   - Type
      - How it works
      - Supported versions, zones and billing
      - Best fit for:
  *   - Spot/Preemptible
      - You request TPU resources which could be preempted. Spot VMs are available at a much lower price than on-demand
        resources. Spot VMs might be easier to obtain than on-demand resources but can be preempted (shut down) at any
        time. There is no limit on runtime duration.
      - `All versions and zones <https://cloud.google.com/tpu/pricing?hl=en>`_ . Billing: Hourly, based on actual usage.
      - ML users who want to run batch / fault-tolerant workloads. Read more about Spot
        `here <https://www.google.com/url?q=https://cloud.google.com/compute/docs/instances/spot&sa=D&source=editors&ust=1740725171021728&usg=AOvVaw00ayRqd2jJwRPUMohb7O3T>`_.
  *   - On Demand
      - You request TPU resources to be used as soon as possible, for as long as you want. On-demand resources won't be
        preempted, but there's no guarantee that there will be enough available TPU resources to satisfy your request.
        On demand is the default when you create TPU resources.
      - `All versions and zones <https://cloud.google.com/tpu/pricing?hl=en>`_ . Billing: Hourly, based on actual usage.
      - On demand is a good fit for workloads that require a flexible end time, likely longer than 7 days.
  *   - Dynamic Workload Scheduler (DWS) - Flex Start
      - You request TPU resources for a specific amount of time, up to 7 days. DWS resources are delivered from a
        dedicated pool of capacity, so the availability of these resources is higher than on-demand.
      - TPU v5e, Trillium (TPU v6e) [Zones TBD] | Billing: Hourly, based on actual usage
      - ML users who want short-term capacity for jobs that take less than 7 days. More about DWS is described here.
        **Preview starting March 2025**
  *   - Reservation: 3-year
      - You request TPU resources in advance for a specific amount of time. These resources are reserved for your
        exclusive use during that period of time. Reservations provide the highest level of assurance for capacity and
        are cost-effective, with a lower price than on-demand resources. You can only use a reservation for TPUs if you
        have a committed use discount (CUD). For more information, `contact Google Cloud
        sales <https://www.google.com/url?q=https://cloud.google.com/contact&sa=D&source=editors&ust=1740725171024055&usg=AOvVaw1hOtjCp1cBKHkPwVDUmnaZ>`_.
      -  `All versions and zones <https://cloud.google.com/tpu/pricing?hl=en>`_ | Billing: Monthly, based on reserved quota
      - Reservations are ideal for long-running training jobs and inference workloads. These are as they include 3-year committed use discounts (CUDs)
  *   - Reservation: 1-year
      - You request TPU resources in advance for a specific amount of time. These resources are reserved for your
        exclusive use during that period of time. Reservations provide the highest level of assurance for capacity and
        are cost-effective, with a lower price than on-demand resources. You can only use a reservation for TPUs if you
        have a committed use discount (CUD). For more information, `contact Google Cloud
        sales <https://www.google.com/url?q=https://cloud.google.com/contact&sa=D&source=editors&ust=1740725171025093&usg=AOvVaw3AaGiV3yP-6VtFr1azKx-h>`_.
      -  `All versions and zones <https://cloud.google.com/tpu/pricing?hl=en>`_ | Billing: Monthly, based on reserved quota
      - Reservations are ideal for long-running training jobs and inference workloads. These are priced lower than on-demand as they include 1-year committed use discounts (CUDs)






.. note:: We recommend DWS Flex Start, DWS Calendar Mode (coming soon) or Reservations for Kithara.

4. Choose a type of capacity
~~~~~~

Once you have decided on the billing mode, there are three ways you can secure capacity to create VMs:


.. list-table:: Model Size Requirements
   :header-rows: 1
   :widths: 20 40 40

   * - VM Type
     - Supported Usage Modes
     - Recommended For
   * - Queued Resource
     - DWS, Spot, On demand
     - Any non-GKE usage
   * - Google Kubernetes Engine
     - DWS (Coming Soon), Spot, On demand, Reservations
     - Any GKE Usage
   * - Compute Engine
     - DWS, Spot, On demand, Reservations
     - Customers with reservations


Before you can create VMs, you must request quota. Read more in the next section.

5. Quota
~~~~~

Once quota has been granted, you can create as many Dynamic Workload Scheduler (DWS), spot, on-demand, reservation VMs as the quota allows. 

When working with Cloud TPUs, you'll encounter quotas that govern your usage. These limits manage availability.

For example, you might have a quota on:

* **The number of TPUs you can create**: This prevents over-provisioning and ensures resources are available for all users.

* **The type of TPUs you can access**: Quotas are tied to a particular TPU type (e.g: v5e) and a particular VM type (spot or on-demand).

These quotas help ensure fair access, prevent abuse, and maintain the stability of the cloud platform. If your project requires more resources than your current quotas allow, you can typically request increases.

To use TPUs with GKE, a separate quota is required. GKE quota is allocated in terms of number of chips. Non-GKE quota is allocated in terms of number of cores.

Read more about quotas and how to request them `here <https://cloud.google.com/tpu/docs/quota>`_.


6. Provision Capacity
~~~~

Once you have determined the type of TPU, type of capacity and usage mode, you are ready to provision capacity. 


.. list-table:: Steps to create VMs
   :header-rows: 0
   :widths: 40 40 20 40 40

   * - Queued Resource →
     - :doc:`DWS Flex Start<dws_flex_start>`
     - `On demand <https://cloud.google.com/tpu/docs/queued-resources#request_an_on-demand_queued_resource>`_
     - `Spot <https://cloud.google.com/tpu/docs/queued-resources#request-spot-qr>`_
     - `Reservation <https://cloud.google.com/tpu/docs/queued-resources#request_a_queued_resource_using_a_reservation>`_
   * - GKE VM →
     - N/A
     - `On demand <https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#provisioning-tpus-options>`_
     - Spot (Add ``--spot`` to the on-demand command)
     - Reservation (Add ``--reservation`` and ``--reservation-affinity=specific`` flags to the on-demand command)
   * - Compute Engine VM →
     - N/A
     - `On demand <https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api>`_
     - Spot (Add ``--spot`` to the on-demand command)
     - Reservation (Add ``--reserved`` to the on-demand command)



.. include:: dws_flex_start.rst

