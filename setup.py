from setuptools import setup, find_packages

setup(
    name="kithara",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flax>=0.7.0",
        "datasets>=3.0.1",
        "huggingface-hub>=0.25.1",
        "jax>=0.4.34",
        "keras>=3.5.0",
        "transformers==4.45.1",
        "scalax==0.2.5",
        "keras-nlp",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "absl-py",
        "fabric==2.7.1",
        "patchwork",
        "ray[default]",
        "pyOpenSSL==23.0.0",
        "ray_tpu",
        "torch"
        "peft"
    ],
)
