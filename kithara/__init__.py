"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    str(maxtext_path),
                    "--no-deps",
                ]
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "pathwaysutils@git+https://github.com/google/pathways-utils.git",
                ]
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
jax.config.update("jax_compilation_cache_dir", jax_cache_dir)

print(f"JAX compilation cached at {jax_cache_dir}")
# Cache with mounted memory
os.environ["HF_HOME"] = "/dev/shm/temp/hf"
os.environ["KERAS_HOME"] = "/dev/shm/temp/keras"

from kithara.utils.logging_utils import print_kithara_logo_and_platform_info

try:
    print_kithara_logo_and_platform_info()
except Exception as e:
    pass
