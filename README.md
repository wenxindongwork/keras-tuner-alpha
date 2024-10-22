
# Keras Tuner Alpha

This repository contains Keras Tuners prototypes. 

# Set up

### 1. Clone this repo with submodules
```
git clone --recursive https://github.com/wenxindongwork/keras-tuner-alpha.git
```
#### Or if already cloned:
```
git submodule update --init --recursive
```

#### Troubleshooting

If you don't see the maxtext repository, try 
```
git submodule add --force https://github.com/google/maxtext
```


### 2. Install dependencies

```
pip install -r requirements.txt
pip install libtpu-nightly==0.1.dev20240925+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

```

# Examples

## Tune a HF model

Example of LoRA finetuning gemma2-2b.
```
python keras_tuner/examples/hf_gemma_example.py
```

## Tune a MaxText model

Example of training a MaxText model. 

```
python keras_tuner/examples/maxtext_default_example.py
```
