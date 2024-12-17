from typing import Any, Optional, Union
from absl import flags
from etils import epath
from flax.training import orbax_utils, train_state
import grain.python as grain
import jax
import max_logging
from multihost_dataloading import MultiHostDataLoadIterator
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

# pylint: disable=too-many-positional-arguments

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions

abstract_logger = ocp.logging.abstract_logger
cloud_logger = ocp.logging.cloud_logger


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: Optional[str] = "ray",
    orbax_logger: Optional[abstract_logger.AbstractLogger] = None,
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
):
    max_logging.log("Creating checkpoint manager...")
    p = epath.Path(checkpoint_dir)

    if dataset_type == "grain":
        item_names = ("items", "iter")
    else:
        item_names = ("items",)

    # local storage checkpoint needs parent directory created
    p.mkdir(exist_ok=True, parents=True)
    # we need to use ocdbt and zarr3 to control max file size in the checkpoint
    # omitting `iter` uses default handler for `iter`
    item_handlers = {"items": PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}
    mngr = CheckpointManager(
        p,
        item_names=item_names,
        item_handlers=item_handlers,
        options=CheckpointManagerOptions(
            create=True,
            save_interval_steps=save_interval_steps,
            enable_async_checkpointing=use_async,
        ),
        logger=orbax_logger,
    )
    max_logging.log("Checkpoint manager created!")
    return mngr


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
    orbax_logger: Optional[abstract_logger.AbstractLogger] = None,
):
    """Returns an emergency checkpoint."""
    flags.FLAGS.experimental_orbax_use_distributed_process_id = True
    max_logging.log("Creating emergency checkpoint manager...")

    options = emergency_checkpoint_manager.CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
        persistent=PersistentCheckpointOptions(save_interval_steps=persistent_save_interval_steps),
    )
    emergency_mngr = emergency_checkpoint_manager.CheckpointManager(
        local_checkpoint_dir,
        epath.Path(persistent_checkpoint_dir),
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
        logger=orbax_logger,
    )

    max_logging.log("Emergency checkpoint manager created!")
    return emergency_mngr
