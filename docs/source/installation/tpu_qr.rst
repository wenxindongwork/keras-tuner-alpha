.. _tpu_vm:


Setting up Ray Cluster with TPU Queued Resources
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

1. Create Ray Head Node
----------------------

Create the Ray head node and launch the Ray cluster::

    ray up -y ray/TPU/QR/cluster.yaml

2. Monitor Cluster Creation
--------------------------

The ``ray up`` command will take a few minutes. If you lose the terminal, monitor the cluster creation process with::

    ray monitor ray/TPU/QR/cluster.yaml

3. Launch Ray Dashboard
----------------------

Launch Ray Cluster dashboard once the ``ray_head_default`` node is ``Active``. You should see the dashboard on your ``localhost:8265``::

    ray dashboard ray/TPU/QR/cluster.yaml

4. Note Head Node IP
-------------------

In the Ray dashboard, note down the IP of your ray head node from the ``Cluster`` panel.

.. note::
   This should be the internal IP, not the external IP of the head node. It should start with 10.x.x.x

5. Set Resource Variables
------------------------

Specify the following variables with your own TPU resources spec::

    export ZONE="us-central2-b"
    export QR_NAME="my_tpu_qr"
    export NODE_ID="v4-a"
    export PROJECT="gcp_project_name"
    export TPU_TYPE="v4-32"
    export RAY_CLUSTER_IP="your_cluster_ip"

6. Create TPU VMs
----------------

Create TPU VMs via QR::

    gcloud alpha compute tpus queued-resources create $QR_NAME \
        --node-id $NODE_ID \
        --zone $ZONE \
        --project $PROJECT \
        --accelerator-type $TPU_TYPE \
        --runtime-version tpu-ubuntu2204-base \
        --metadata-from-file='startup-script=ray/TPU/QR/qr_worker_startup_script.sh'

7. Monitor QR Status
-------------------

Monitor the status of the QR creation::

    gcloud compute tpus queued-resources describe $QR_NAME --project $PROJECT --zone $ZONE

Once the status becomes ``ACTIVE``, monitor the logs to verify package installation::

    gcloud alpha compute tpus queued-resources ssh $QR_NAME \
        --project $PROJECT \
        --zone $ZONE \
        --command="sudo cat /var/log/syslog | grep startup-script" \
        --worker=0 \
        --node=all

8. Attach TPU VMs to Ray Cluster
-------------------------------

Once QRs are ready, attach the TPU VMs as worker nodes::

    gcloud alpha compute tpus queued-resources ssh $QR_NAME \
        --project $PROJECT \
        --zone $ZONE \
        --command="ray start --address=$RAY_CLUSTER_IP:6379 --resources='{\"tpu_host\": 1}'" \
        --worker=all \
        --node=all

Troubleshooting
~~~~~~~~~~~~~~

If you encounter Python or Ray version inconsistencies, check the worker node logs::

    gcloud alpha compute tpus queued-resources ssh $QR_NAME \
        --project $PROJECT \
        --zone $ZONE \
        --command="sudo cat /var/log/syslog | grep startup-script" \
        --worker=all \
        --node=all

9. Run Examples
--------------

Your Ray Cluster is now ready. Try examples in the ``examples/multihost/TPU`` folder::

    python ray/submit_job.py "python3.11 examples/multihost/ray/TPU/sft_lora_example.py" --hf-token your_token

To early-stop your job::

    ray job stop ray_job_id

10. Remove QRs
-------------

To remove QRs from your Ray Cluster::

    gcloud compute tpus queued-resources delete $QR_NAME --project $PROJECT --zone $ZONE

11. Tear Down Cluster
--------------------

When finished with your ray cluster, tear it down::

    ray down ray/TPU/QR/cluster.yaml