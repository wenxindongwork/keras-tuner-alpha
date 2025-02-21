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
* Enable GCSFuse

  This step allows GCS buckets to be mounted on GKE containers as drives. This makes it easier for Kithara to save checkpoints to GCS.

  You can follow the instructions here: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gke-gcs-bucket.html.
* Authenticate to Your GKE Cluster::
        gcloud container clusters get-credentials $CLUSTER --zone $ZONE --project $YOUR_PROJECT
* Create a Hugging Face token on https://huggingface.co/docs/hub/en/security-tokens
* Save the Hugging Face token to the Kubernetes cluster::
        kubectl create secret generic hf-secret \
                --from-literal=hf_api_token=HUGGING_FACE_TOKEN 




Setting Up a Ray Cluster
------------------------
1. Edit one of the following manifest files:

   - Single-host: https://github.com/richardsliu/keras-tuner-alpha/blob/main/ray/TPU/GKE/single-host.yaml

   - Multi-host: https://github.com/richardsliu/keras-tuner-alpha/blob/main/ray/TPU/GKE/multi-host.yaml

   Make sure to replace ``YOUR_GCS_BUCKET`` with the name of the GCS bucket created in previous steps.
2. Deploy the Ray cluster::
        kubectl apply -f $MANIFEST_FILE
3. Check that the cluster is running with::
        kubectl get pods


Running a Ray Workload
----------------------
1. Set the following environment variable::
        export RAY_ADDRESS=http://localhost:8265
2. Port-forward to the Ray cluster::
        kubectl port-forward svc/example-cluster-kuberay-head-svc 8265:8265 &
3. Submit a Ray job, for example::
        ray job submit  --working-dir . \
                --runtime-env-json='{"excludes": [".git", "kithara/model/maxtext/maxtext/MaxText/test_assets"]}' \
                -- python examples/multihost/ray/TPU/full_finetuning_example.py
4. You can visit ``http://localhost:8265`` in your browser to see the Ray dashboard and monitor job status.
