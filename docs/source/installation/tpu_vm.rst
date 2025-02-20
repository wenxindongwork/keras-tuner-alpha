.. _tpu_vm:

Setting up Ray Cluster with TPU VMs
================================================

Prerequisites
-------------
* Clone the Kithara repo. 

.. code-block:: bash

    git clone https://github.com/wenxindongwork/keras-tuner-alpha.git

* Modify ``ray/TPU/cluster.yaml`` with your GCP project, zone, and TPU resource types.

.. tip::
       Search for "MODIFY" in the YAML file to find required changes

Setting up the Ray Cluster
-------------------------

1. Launch the cluster::

       ray up -y ray/TPU/cluster.yaml

3. Monitor setup process::

       ray monitor ray/TPU/cluster.yaml

4. Launch Ray dashboard::

       ray dashboard ray/TPU/cluster.yaml

   The dashboard will be available at ``localhost:8265``

Troubleshooting
~~~~~~~~~~~~~~
* ``update-failed`` errors typically don't affect proper node setup
* Check node status by executing::

      ray attach cluster.yaml
      ray status

Running Multihost Jobs
---------------------

1. Submit job::

       python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token

2. To stop a job early::

       export RAY_ADDRESS="http://127.0.0.1:8265"
       ray job stop ray_job_id

Cleanup
-------

When finished, tear down the cluster::

    ray down cluster.yaml