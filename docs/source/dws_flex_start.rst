.. _dws_flex_start:

Create DWS (Flex Start) VMs
++++++++++++++++++++++++++

Flex Start uses the TPU queued resources API to request TPU resources in a queued manner. When the requested resource becomes available, it's assigned to your Google Cloud project for your immediate, exclusive use. After the requested run duration, the TPU VMs are deleted and the queued resource moves to the ``SUSPENDED`` state. For more information about queued resources, see `Manage queued resources <https://cloud.google.com/tpu/docs/queued-resources>`_.

To request TPUs using Flex Start, use the `gcloud alpha compute tpus queued-resources create <https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/create>`_ command with the ``--provisioning-model`` flag set to ``FLEX-START`` and the ``--max-run-duration`` flag set to the duration you want your TPUs to run.

.. code:: bash

   gcloud alpha compute tpus queued-resources create \
   <your-queued-resource-id> \
   --zone=<your-zone> \
   --accelerator-type=<your-accelerator-type> \
   --runtime-version=<your-runtime-version> \
   --node-id=<your-node-id> \
   --provisioning-model=FLEX-START \
   --max-run-duration=<run-duration>

Replace the following placeholders:

*   ``<your-queued-resource-id>``: A user-assigned ID for the queued resource request.

*   ``<your-zone>``: The `zone <https://cloud.google.com/tpu/docs/regions-zones>`_ in which to create the
    TPU VM.

*   ``<your-accelerator-type>``: Specifies the version and size of the Cloud TPU to create. For more information about supported accelerator types for each TPU version, see `TPU versions <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions>`_.

*   ``<your-runtime-version>``: The Cloud TPU `software version <https://cloud.google.com/tpu/docs/runtimes>`_.

*   ``<your-node-id>``: A user-assigned ID for the TPU that is created when the queued resource
    request is allocated.

*   ``<run-duration>``: How long the TPUs should run. Format the duration as the number of days, hours, minutes, and seconds followed by ``d``, ``h``, ``m``, and ``s``, respectively. For example, specify ``72h`` for a duration of 72 hours, or specify ``1d2h3m4s`` for a duration of 1 day, 2 hours, 3 minutes, and 4 seconds. The maximum is 7 days.

You can further customize your queued resource request to run at specific times with additional flags:

*   ``--valid-after-duration``: The duration before which the TPU must not be provisioned.

*   ``--valid-after-time``: The time before which the TPU must not be provisioned.

*   ``--valid-until-duration``: The duration for which the request is valid. If the request hasn't been fulfilled by this duration, the request expires and moves to the ``FAILED`` state.

*   ``--valid-until-time``: The time for which the request is valid. If the request hasn't been fulfilled by this time, the request expires and moves to the ``FAILED`` state.

For more information about optional flags, see the `gcloud alpha compute tpus queued-resources create <https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/create>`_ documentation.


.. get_status::

Get the status of a Flex Start request
++++++++++++++++++++++

To monitor the status of your Flex Start request, use the queued resources API to get the status of the queued resource request using the `gcloud alpha compute tpus queued-resources describe <https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/describe>`_ command:

.. code:: bash

   gcloud alpha compute tpus queued-resources describe <your-queued-resource-id> \
   --zone <your-zone>

A queued resource can be in one of the following states:

*   **WAITING_FOR_RESOURCES:** The request has passed initial validation and has been added to the queue.
*   **PROVISIONING:** The request has been selected from the queue and its TPU VMs are being created.
*   **ACTIVE:** The request has been fulfilled, and the VMs are ready.
*   **FAILED:** The request could not be completed. Use the ``describe`` command for more details.
*   **SUSPENDING:** The resources associated with the request are being deleted.
*   **SUSPENDED:** The resources specified in the request have been deleted.

For more information, see `Retrieve state and diagnostic information about a queued resource
request <https://cloud.google.com/tpu/docs/queued-resources#retrieve_state_and_diagnostic_information_about_a_queued_resource_request>`_.

Monitor the run time of Flex Start TPUs
++++++++

You can monitor the run time of Flex Start TPUs by checking the TPU's termination timestamp:

1.  Get the details of your queued resource request using the steps in the previous section, :doc:`Get the status of a Flex Start request<get_status>`.
2.  **If the queued resource is waiting for resources:** In the output, see the ``maxRunDuration`` field. 
    This field specifies how long the TPUs will run once they're created.
    **If the TPUs associated with the queued resource have been created:** In the output, see the
    ``terminationTimestamp`` field listed for each node in the queued resource. This field specifies
    when the TPU will be terminated.

Delete a queued resource
+++++++

**Important:** Queued resources consume quota regardless of their state. Delete queued resources after use to avoid blocking future requests on quota limits.

You can delete a queued resource request and the TPUs associated with the request by deleting the queued resource request and passing the ``--force`` flag to the ``queued-resource delete`` command:

.. code:: bash

   gcloud alpha compute tpus queued-resources delete <your-queued-resource-id> \
   --zone <your-zone> \
   --force

If you delete the TPU directly, you also need to delete the queued resource, as shown in the following example. When you delete the TPU, the queued resource request transitions to the ``SUSPENDED`` state, after which you can delete the queued resource request.

To delete a TPU, use the `gcloud alpha compute tpus tpu-vm delete <https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/delete>`_ command:

.. code:: bash

   gcloud compute tpus tpu-vm delete <your-node-id> \
   --zone <your-zone>

Then, to delete the queued resource, use the `gcloud alpha compute tpus queued-resources delete <https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/queued-resources/delete>`_ command:

.. code:: bash

 gcloud compute tpus queued-resources delete <your-queued-resource-id> \
  --zone <your-zone>

For more information see `Delete a queued resource request <https://cloud.google.com/tpu/docs/queued-resources#delete_a_queued_resource_request>`_.
