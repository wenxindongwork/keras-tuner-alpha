import os

# Allows faster HF download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["KERAS_BACKEND"] = "jax"


from pathlib import Path
import subprocess
from subprocess import DEVNULL
import importlib.metadata
import sys

import time

def _install_maxtext():
    try:
        importlib.metadata.version("maxtext")
    except importlib.metadata.PackageNotFoundError:
        try:
            print(
                "Installing MaxText... This should only happen once when Kithara is first initiated."
            )
            maxtext_path = Path(
                os.path.join(os.path.dirname(Path(__file__)), "model/maxtext/maxtext")
            )
            if maxtext_path.exists():
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-e",
                        str(maxtext_path),
                        "--no-deps",
                    ],
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "pathwaysutils@git+https://github.com/google/pathways-utils.git",
                    ],
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
                print("MaxText installed successfully")
        except Exception as e:
            print(f"Failed to install maxtext: {e}")

    maxtext_dir = Path(
        os.path.join(os.path.dirname(Path(__file__)), "model/maxtext/maxtext/MaxText")
    )
    sys.path.append(str(maxtext_dir))

_install_maxtext()

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
os.environ["JAX_COMPILATION_CACHE_DIR"] = jax_cache_dir
