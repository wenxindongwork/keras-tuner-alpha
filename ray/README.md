# Orchestration
You can find instructions for running multihost workloads in this section. 

## Enable Multihost Tuning via Ray Cluster

Ray is a great tool for running distributed TPU and GPU workloads. It offers a dashboard for job queueing and log visualization, and streamlines the environment set up and file syncing. Once you set up a Ray Cluster, you can run multihost workloads with a simple command. We provide instructions for setting up Ray Cluster with GCE TPU/GPU VMs and TPU QRs. 


### Instructions for setting up Ray Cluster with GCE resources ###

1. Assume you have resource capacity and quota in your GCP project and region/zone. Modify `ray/TPU/cluster.yaml` or `ray/GPU/cluster.yaml`template with your configurations. Please take a look at the YAML file and ctrl+F for MODIFY.

2. Run the following commands to bring up your ray cluster. 

    For TPU,  

    ```
    cd ray/TPU
    ```

    For GPU: 

    ```
    cd ray/GPU
    ```

    Run 

    ```
    ray up -y cluster.yaml
    ```

    Monitor the node set up process by running

    ```
    ray monitor cluster.yaml
    ```

    While you monitor the node set up process, launch the ray dashboard.

    ```
    ray dashboard cluster.yaml
    ```

    You should see the dashboard on your `localhost:8265`

    **Troubleshooting**:

    - If you see an `update-failed` error, don't worry. This should not affect the nodes being properly set up.
    - You can exec into the ray head node using `ray attach cluster.yaml` and run `ray status` to see the status of the nodes.


3. Once all nodes in your ray cluster are set up and active, and you have launched the dashboard, run the HuggingFace gemma example using the following commands, in another terminal window. 

    Run a job on TPU: 

    ```
    export RAY_ADDRESS="http://127.0.0.1:8265"
    python ray/submit_ray_job.py "python examples/multihost/ray/TPU/hf_gemma_example_via_ray.py" --hf-token your_token
    ```

    Run a job on GPU: 
    
    ```
    export RAY_ADDRESS="http://127.0.0.1:8265"
    python ray/submit_ray_job.py "python examples/multihost/ray/GPU/hf_gemma_example_via_ray.py" --hf-token your_token
    ```

    You can early-stop your job using 

    ```ray job stop ray_job_id```

4. Once you are done with your ray cluster, tear it down

    `ray down cluster.yaml`


### Instructions for setting up Ray Cluster with TPU QR ###

1. Create the Ray head node and launch the Ray cluster. 
    ```
    ray up -y ray/TPU/QR/cluster.yaml
    ```
2. This will take a while (a few minutes), using this command to monitor the cluster creation process. 
    ```
    ray monitor ray/TPU/QR/cluster.yaml
    ```
3. Launch Ray Cluster dashboard once the Ray Cluster is ready. You should see the dashboard on your `localhost:8265`

    ```
    ray dashboard ray/TPU/QR/cluster.yaml
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

6. Create TPU VMs via QR. 

    ```
    gcloud alpha compute tpus queued-resources create $QR_NAME --node-id $NODE_ID --zone $ZONE  --project $PROJECT --accelerator-type $TPU_TYPE --runtime-version tpu-ubuntu2204-base --metadata-from-file='startup-script=ray/TPU/QR/qr_worker_startup_script.sh'
    ```

7. Monitor the status of the QR creation with the following command. 
    ```
    gcloud compute tpus queued-resources describe $QR_NAME --project $PROJECT --zone $ZONE
    ```

8. Once the QRs are ready, attach the TPU VMs to the Ray Cluster as worker nodes. 
    ```
    gcloud alpha compute tpus queued-resources ssh $QR_NAME --project $PROJECT --zone $ZONE --command="ray start --address=$RAY_CLUSTER_IP:6379 --resources='{\"tpu_host\": 1}'" --worker=all --node=all
    ```

9. Now your Ray Cluster is ready, try out examples in the `examples/multihost/TPU` folder. 

    Run a job on TPU: 

    ```
    export RAY_ADDRESS="http://127.0.0.1:8265"
    python ray/submit_ray_job.py "python examples/multihost/ray/TPU/hf_gemma_example_via_ray.py" --hf-token your_token
    ```

    Run a job on GPU: 
    
    ```
    export RAY_ADDRESS="http://127.0.0.1:8265"
    python ray/submit_ray_job.py "python examples/multihost/ray/GPU/hf_gemma_example_via_ray.py" --hf-token your_token
    ```

10. To remove QRs from your Ray Cluster, run this command. 
    ```
    gcloud compute tpus queued-resources delete $QR_NAME --project $PROJECT --zone $ZONE
    ```

11. Once you are done with your ray cluster, tear it down

    `ray down ray/TPU/cluster.yaml`


## TroubleShooting

1. Error `Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid <PID>.`

    Try removing the following files from TPU VM hosts. 
    ```
    import subprocess
    subprocess.run(["rm", "-rf", "/tmp/libtpu_lockfile", "/tmp/tpu_logs"])
    ```
