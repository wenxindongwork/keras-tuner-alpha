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


def load_state_if_possible(
    checkpoint_manager: Union[CheckpointManager, None],
    data_iterator: Union[MultiHostDataLoadIterator, None],
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_unboxed_pre_state: train_state.TrainState,
    enable_single_replica_ckpt_restoring: Optional[bool] = False,
    dataset_type: Optional[str] = "tfds",
    step: int = -1,  # -1 means latest
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint manager,
      load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint manager,
      load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
      with SingleReplicaArrayHandler

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = checkpoint_manager.latest_step() if step < 0 else step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      def map_to_pspec(data):
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        if not enable_single_replica_ckpt_restoring:
          return ocp.type_handlers.ArrayRestoreArgs(mesh=mesh, mesh_axes=pspec)
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      if enable_single_replica_ckpt_restoring:
        array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=0,
            broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
        )
        ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

      restore_args = jax.tree_util.tree_map(
          map_to_pspec,
          abstract_unboxed_pre_state,
      )

    return (
        checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                items=ocp.args.PyTreeRestore(
                    item=abstract_unboxed_pre_state,
                    restore_args=restore_args,
                )
            ),
        ),
        None,
    )

  if load_parameters_from_path != "":
    restored_params = load_params_from_path(load_parameters_from_path, abstract_unboxed_pre_state.params)
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"restoring full state from {load_full_state_from_path=}")
    p = epath.Path(load_full_state_from_path)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(p, abstract_unboxed_pre_state)
    return {"items": restored}, None

  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None
