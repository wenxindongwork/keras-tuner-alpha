.. _scaling_with_ray:

Running on Multihost?
=====================

Kithara works with any accelerator orchestrator. However, if you are new to distributed training, we provide guide for multihost training with Ray.

.. admonition:: What is Ray?

    Ray is a popular and powerful tool for running distributed TPU and GPU workloads, offering:

    - Dashboard for job queueing and log visualization
    - Streamlined environment setup and file syncing
    - Simple command interface for multihost workloads

Follow the instructions below to set up your Ray cluster.

* :doc:`Set up Ray Cluster with TPU VMs <installation/tpu_vm>`
* :doc:`Set up Ray Cluster with TPU QRs <installation/tpu_qr>` 
* :doc:`Set up Ray Cluster with TPU GKE <installation/tpu_gke>`

.. toctree::
    :maxdepth: 2
    :caption: Ray Cluster Setup
    :hidden:

    Ray Cluster with TPU VMs <installation/tpu_vm>
    Ray Cluster with TPU QRs <installation/tpu_qr>
    Ray Cluster with TPU GKE <installation/tpu_gke>
