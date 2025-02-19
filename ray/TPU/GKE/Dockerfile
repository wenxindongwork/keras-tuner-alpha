FROM rayproject/ray:2.42.0-py311

RUN python -m pip install --upgrade pip

RUN pip install -U kithara[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url  https://download.pytorch.org/whl/cpu

USER root
