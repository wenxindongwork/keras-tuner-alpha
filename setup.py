from setuptools import setup, find_packages

setup(
    name="keras_tuner",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.13",
        "flax>=0.7.0",
        "datasets>=3.0.1",
        "huggingface-hub>=0.25.1", 
        "jax>=0.4.34",
        "keras>=3.5.0",
        "transformers==4.45.1",
        "scalax==0.2.5",
        "keras-nlp"
    ]
)