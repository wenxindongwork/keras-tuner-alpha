# Unique source of truth of package version
__version__="0.1.0"

import os
# Allows faster HF download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["KERAS_BACKEND"] = "jax"

import sys
maxtext_dir = "maxtext/MaxText"
sys.path.append(maxtext_dir)

from kithara.dataset import Dataloader, SFTDataset, TextCompletionDataset
from kithara.trainer import Trainer
from kithara.callbacks import Checkpointer, Profiler
import jax 
from kithara.utils.gcs_utils import find_cache_root_dir
from kithara.model import KerasHubModel, MaxTextModel, Model
from kithara.distributed import ShardingStrategy, PredefinedShardingStrategy
# Cache JAX compilation to speed up future runs. You should notice
# speedup of training step up on the second run of this script.
jax_cache_dir = os.path.join(find_cache_root_dir(), "jax_cache")
jax.config.update("jax_compilation_cache_dir", os.path.join(find_cache_root_dir(), "jax_cache"))
print(f"Initialized jax_compilation_cache_dir = {jax_cache_dir}")


