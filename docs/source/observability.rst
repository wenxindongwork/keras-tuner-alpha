.. _observability:

Observability
=============

Kithara supports Tensorboard and (soon) Weights and Biases for observability.

Tensorboard
-----------

To use Tensorboard, simply specify the ``tensorboard_dir`` arg in the ``Trainer`` class to a local directory or a Google Cloud Storage bucket.

To track training and evaluation performance, launch the tensorboard server with::

    tensorboard --logdir=your_tensorboard_dir
