.. _installation:

Installation
============

.. note::

    Kithara requires ``Python>=3.11``. 

We recommend using a virtual environment.

1. With `conda`:

.. code-block:: bash

   conda create -n kithara_env python=3.11
   conda activate kithara_env

.. dropdown:: How to install Conda on GCP VM
    :open:

    .. code-block:: bash
    
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh    
        source ~/miniconda3/bin/activate
        conda init --all


2. With `venv`:

.. code-block:: bash

   python3.11 -m venv kithara_env
   source kithara_env/bin/activate

Installation on TPU 
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install kithara[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url https://download.pytorch.org/whl/cpu

Installation on GPU 
~~~~~~~~~~~~~~~~~~~

.. warning:: 

    Our GPU support is still in beta. Please report any issues you encounter.

.. code-block:: bash

    pip install kithara[gpu]
