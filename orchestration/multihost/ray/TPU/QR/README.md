## Instructions for setting up Ray Cluster with QR ##

1. Create the Ray head node and launch the Ray cluster. 
    ```
    ray up -y orchestration/multihost/ray/TPU/QR/cluster.yaml
    ```
2. This will take a while (a few minutes), using this command to monitor the cluster creation process. 
    ```
    ray monitor orchestration/multihost/ray/TPU/QR/cluster.yaml
    ```
3. Launch Ray Cluster dashboard once the Ray Cluster is ready. 
    ```
    ray dashboard orchestration/multihost/ray/TPU/QR/cluster.yaml
    ```
4. In the Ray dashboard, note down the IP of your ray head node, you can find this in the `Cluster` panel.  

5. Specify the following variables with your own TPU resources spec. 

    ```
    export ZONE="us-central2-b"
    export QR_NAME="my_tpu_qr"
    export NODE_ID="v4-a"
    export PROJECT="gcp_project_name"
    export TPU_TYPE="v4-32"
    export RAY_CLUSTER_IP="your_cluster_ip"
    ```

4. Create TPU VMs via QR. 

    ```
    gcloud alpha compute tpus queued-resources create $QR_NAME --node-id $NODE_ID --zone $ZONE  --project $PROJECT --accelerator-type $TPU_TYPE --runtime-version tpu-ubuntu2204-base     --metadata-from-file='startup-script=orchestration/multihost/ray/TPU/QR/qr_worker_startup_script.sh'
    ```

5. Monitor the status of the QR creation with the following command. 
    ```
    gcloud compute tpus queued-resources describe $QR_NAME \
    --project $PROJECT \
    --zone $ZONE
    ```

6. Once the QRs are ready, attach the TPU VMs to the Ray Cluster as worker nodes. 
    ```
    gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="ray start --address=$RAY_CLUSTER_IP:6379 --resources='{\"tpu_host\": 1}'" --worker=all --node=all
    ```

7. Now your Ray Cluster is ready. Try out examples in the `examples/multihost/TPU` folder!

7. Delete your QR 
    ```
    gcloud compute tpus queued-resources delete $QR_NAME \
    --project $PROJECT \
    --zone $ZONE
    ```
