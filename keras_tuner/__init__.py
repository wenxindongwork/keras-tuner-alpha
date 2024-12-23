# from keras_tuner.model import MaxTextModel, KerasHubModel
from keras_tuner.dataset import Dataloader
from keras_tuner.preprocessor import Preprocessor, PretrainingPreprocessor, SFTPreprocessor
from keras_tuner.trainer import Trainer
from keras_tuner.callbacks import Checkpointer, Profiler
import jax 
from keras_tuner.utils.gcs_utils import find_cache_root_dir

# Cache JAX compilation to speed up future runs. You should notice
# speedup of training step up on the second run of this script.
import os
jax_cache_dir = os.path.join(find_cache_root_dir(), "jax_cache")
jax.config.update("jax_compilation_cache_dir", os.path.join(find_cache_root_dir(), "jax_cache"))
print(f"Initialized jax_compilation_cache_dir = {jax_cache_dir}")
