import os
# Allows faster HF download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["KERAS_BACKEND"] = "jax"
# Cache with mounted memory
os.environ["HF_HOME"] = "/dev/shm/temp/hf"
os.environ["KERAS_HOME"] = "/dev/shm/temp/keras"

from pathlib import Path
import subprocess
import importlib.metadata
import sys
def _install_maxtext():
    try:
        importlib.metadata.version('maxtext')
    except importlib.metadata.PackageNotFoundError:
        try:
            maxtext_path = Path(os.path.join(os.path.dirname(Path(__file__)), "model/maxtext/maxtext"))
            if maxtext_path.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(maxtext_path), "--no-deps"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pathwaysutils@git+https://github.com/google/pathways-utils.git"])
                print("MaxText installed successfully")
        except Exception as e:
            print(f"Failed to install maxtext: {e}")

_install_maxtext()

import sys
maxtext_dir = Path(os.path.join(os.path.dirname(Path(__file__)), "model/maxtext/maxtext/MaxText"))
sys.path.append(str(maxtext_dir))

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


