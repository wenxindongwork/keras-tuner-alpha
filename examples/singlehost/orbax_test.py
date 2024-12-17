
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
import ray
import jax 
from typing import Optional
from keras_tuner import Dataloader, PretrainingPreprocessor, Trainer
from keras_tuner.model.models.maxtext.maxtext_model import MaxTextModel
from examples.example_datasets import example_datasets

import numpy as np
import orbax.checkpoint as ocp
import jax

import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint



model = MaxTextModel.from_preset(
    preset_handle=config["hf_handle"],
    seq_len=config["seq_len"],
    per_device_batch_size=config["per_device_batch_size"],
    precision=config["precision"],
    scan_layers=True
)


if config.enable_emergency_checkpoint:
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
        config.local_checkpoint_directory,
        config.checkpoint_dir,
        mesh,
        abstract_state,
        config.local_checkpoint_period,
        config.checkpoint_period,
        logger,
    )

#

checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    config.checkpoint_dir,
    config.enable_checkpointing,
    config.async_checkpointing,
    config.checkpoint_period,
    config.dataset_type,
    logger,
    use_ocdbt,
    use_zarr3,
)


checkpointer = ocp.StandardCheckpointer()


checkpointer.save(path / 'checkpoint_name', my_tree)

checkpointer.restore(
    path / 'checkpoint_name/',
    abstract_my_tree
)


def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="c4",
    data_iterator=None,
    config: Optional[pyconfig.config] = None,
) -> bool:
"""Wrapper for saving checkpoint."""
if config and config.enable_checkpointing:
    if (step % config.checkpoint_period == 0) or (
        config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0
    ):
    blocking_until_ready_start = time.time()
    max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
    # We block here on the step finishing so that our checkpointing metrics
    # measure only checkpointing time, not training time.
    jax.block_until_ready(state)
    max_logging.log(
        f"Waited {time.time() - blocking_until_ready_start} seconds for step "
        f"{step} to finish before starting checkpointing."
    )

    # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
    chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
    if config:
        chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
    save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

    if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
        return checkpoint_manager.save(
            step,
            args=orbax.checkpoint.args.PyTreeSave(item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size),
        )

    if dataset_type == "grain":
        return checkpoint_manager.save(
            step,
            args=orbax.checkpoint.args.Composite(
                items=orbax.checkpoint.args.PyTreeSave(
                    item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
                ),
                iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
            ),
        )
    else:
        return checkpoint_manager.save(
            step,
            args=orbax.checkpoint.args.Composite(
                items=orbax.checkpoint.args.PyTreeSave(
                    item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
                )
            ),
        )


if checkpoint_manager is not None:
    state_to_save = state if not config.use_dpo else _split_dpo_state(state)[0]
    if save_checkpoint(checkpoint_manager, int(step), state_to_save, config.dataset_type, data_iterator, config):
        max_logging.log(f"saved a checkpoint at step {step}")

    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()
