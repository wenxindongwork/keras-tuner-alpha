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


After setting up your Ray Cluster, you can run multihost jobs using the following recipe. Run this script 
on your local machine with which you've set up your Ray Cluster.::

    import ray
    import jax

    ray.init()

    num_chips_per_host = 4  # <--IMPORTANT: Use 4 for v4 and v5, 8 for v4e and v5e TPU
    num_tpu_hosts = int(ray.cluster_resources()["TPU"] / num_chips_per_host)
    print(f"{num_tpu_hosts=}")

    # Define a Ray remote function which will be run on all hosts simultinuosly
    @ray.remote(resources={"TPU": num_chips_per_host})
    def main():

        # HuggingFace login
        from huggingface_hub import login
        import os

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        # Let JAX know that we are running a distributed job
        jax.distributed.initialize()

        # No need to change your single host job script, simply use it as it is. 
        from examples.singlehost.quick_start import run_workload

        # Run this workload on all hosts. Don't worry, we are handling 
        # all the model sharding and batch sharding for you. 
        run_workload()


    ray.get([main.remote() for i in range(num_tpu_hosts)])

    ray.shutdown()
