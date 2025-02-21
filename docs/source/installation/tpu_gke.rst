.. _tpu_vm:

Setting up Ray Cluster with TPU GKE
=====================================
Prerequisites
-------------
* Clone the Kithara repo.

.. code-block:: bash

    git clone https://github.com/wenxindongwork/keras-tuner-alpha.git
* Create a GKE Cluster with Ray add-on: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gcp-gke-tpu-cluster.html

Preparing Your GKE Cluster
--------------------------
* Enable GCSFuse.

  This step allows GCS buckets to be mounted on GKE containers as drives. This makes it easier for Kithara to save checkpoints to GCS.

  You can follow the instructions here: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gke-gcs-bucket.html.
* Authenticate to Your GKE Cluster:

.. code-block:: bash

    gcloud container clusters get-credentials $CLUSTER --zone $ZONE --project $YOUR_PROJECT
* Create a Hugging Face token on https://huggingface.co/docs/hub/en/security-tokens
* Save the Hugging Face token to the Kubernetes cluster:

.. code-block:: bash

    kubectl create secret generic hf-secret \
        --from-literal=hf_api_token=HUGGING_FACE_TOKEN 




Setting Up a Ray Cluster
------------------------
1. Edit one of the following manifest files:

   - Single-host: https://github.com/richardsliu/keras-tuner-alpha/blob/main/ray/TPU/GKE/single-host.yaml

   - Multi-host: https://github.com/richardsliu/keras-tuner-alpha/blob/main/ray/TPU/GKE/multi-host.yaml

   Make sure to replace ``YOUR_GCS_BUCKET`` with the name of the GCS bucket created in previous steps.
2. Deploy the Ray cluster:

.. code-block:: bash

    kubectl apply -f $MANIFEST_FILE
3. Check that the cluster is running with:

.. code-block:: bash

    kubectl get pods

If everything works as expected, you should see pods running:

.. code-block:: bash

    NAME                                               READY   STATUS    RESTARTS   AGE
    example-cluster-kuberay-head-kgxkp                 2/2     Running   0          1m
    example-cluster-kuberay-worker-workergroup-bzrz2   2/2     Running   0          1m
    example-cluster-kuberay-worker-workergroup-g7k4t   2/2     Running   0          1m
    example-cluster-kuberay-worker-workergroup-h6zsx   2/2     Running   0          1m
    example-cluster-kuberay-worker-workergroup-pdf8x   2/2     Running   0          1m


Running a Ray Workload
----------------------
1. Set the following environment variable:

.. code-block:: bash

    export RAY_ADDRESS=http://localhost:8265
2. Port-forward to the Ray cluster:

.. code-block:: bash

    kubectl port-forward svc/example-cluster-kuberay-head-svc 8265:8265 &
3. Submit a Ray job, for example:

.. code-block:: bash

    ray job submit  --working-dir . \
        --runtime-env-json='{"excludes": [".git", "kithara/model/maxtext/maxtext/MaxText/test_assets"]}' \
        -- python examples/multihost/ray/TPU/full_finetuning_example.py
4. You can visit ``http://localhost:8265`` in your browser to see the Ray dashboard and monitor job status.


Clean Up
--------
1. When your job is done, you can delete it by running:

.. code-block:: bash

    kubectl delete -f $MANIFEST_FILE

2. The GKE cluster can be deleted with:

.. code-block:: bash

   gcloud clusters delete $CLUSTER --zone $ZONE

