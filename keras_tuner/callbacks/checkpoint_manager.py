

from keras.src.callbacks.callback import Callback
import jax
import orbax
import time
from etils import epath
import orbax.checkpoint as ocp
from typing import Optional

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
abstract_logger = ocp.logging.abstract_logger


class Checkpointer(Callback):
    """
    """

    def __init__(
        self,
        checkpoint_dir: str,
        use_async: bool = True,
        save_interval_steps: int = -1,
        orbax_logger: Optional[abstract_logger.AbstractLogger] = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = True, 
        by_batch: bool = True, 
        by_epoch: bool = False, 
        chunk_byte_size:int = 2 * 1024**3,
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.use_async = use_async
        self.save_interval_steps = save_interval_steps
        self.orbax_logger = orbax_logger
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3
        self.by_batch = by_batch
        self.by_epoch = by_epoch
        self.mngr = self._set_up_checkpoint_manager()
        self._num_train_batch = 0
        self._num_train_epochs = 0
        self.chunk_byte_size =chunk_byte_size

    def on_train_batch_begin(self, batch, logs=None):
        
        self._num_train_batch += 1
        
        if self.by_batch:
            if self._num_train_batch % self.save_interval_steps:
                state = self.model.variables
                self.save(self._num_train_batch, state)

    def on_epoch_begin(self, epoch, logs=None):
        
        self._num_train_epoch += 1
        
        if self.by_epoch:
            if self._num_train_epoch % self.save_interval_steps:
                state = self.model.variables
                self.save(self._num_train_epoch, state)

    def save(self, step, state, blocking = False):
        
        print(f"-> Saving checkpoint at step {step}")
        blocking_until_ready_start = time.time()
        print(f"Waiting for step {step} to finish before checkpoint...")
        jax.block_until_ready(state)
        print(
            f"Waited {time.time() - blocking_until_ready_start} seconds for step "
            f"{step} to finish before starting checkpointing."
        )
        
        save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=self.chunk_byte_size), state)

        self.mngr.save(step,
            args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=self.chunk_byte_size
            )
        )
)
        # If we are at a preemption step, we must wait for the 
        # checkpoint to finish writing before exiting.
        if self.mngr.reached_preemption(step):
            self.mngr.wait_until_finished()
            print(f"✅ Successfully saved checkpoint to {self.checkpoint_dir}/{step}")
            exit()
        
        if blocking: 
            self.mngr.wait_until_finished()
            print(f"✅ Successfully saved checkpoint to {self.checkpoint_dir}/{step}")
        
            
    def _set_up_checkpoint_manager(self) -> 'CheckpointManager':
        p = epath.Path(self.checkpoint_dir)

        item_names = ("items",)

        # local storage checkpoint needs parent directory created
        p.mkdir(exist_ok=True, parents=True)
        # we need to use ocdbt and zarr3 to control max file size in the checkpoint
        # omitting `iter` uses default handler for `iter`
        item_handlers = {"items": PyTreeCheckpointHandler(use_ocdbt=self.use_ocdbt, use_zarr3=self.use_zarr3)}
        mngr = CheckpointManager(
            p,
            item_names=item_names,
            item_handlers=item_handlers,
            options=CheckpointManagerOptions(
                create=True,
                save_interval_steps=self.save_interval_steps,
                enable_async_checkpointing=self.use_async,
            ),
            logger=self.orbax_logger,
        )
        return mngr