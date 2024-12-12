import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import jax.tree_util as jtu
from typing import Any, Union, List, Tuple
from keras.src.backend.common import global_state
from keras_tuner.model.sharding._data_sharding import DataSharding
from keras_tuner.dataset import Dataloader
from keras_tuner.model import Model
from keras_tuner.preprocessor import Preprocessor
from keras_tuner.model.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
)
from keras_tuner.preprocessor import Preprocessor
from keras_tuner.model import Model
from keras_tuner.dataset import Dataloader
from keras.src.backend.common import global_state
from typing import Any, Union, List, Tuple

import jax
import sys
import time


class Trainer:
    def __init__(
        self,
        model: Model,
        optimizer: keras.Optimizer,
        train_dataloader: Dataloader,
        preprocessor: Preprocessor = None,
        eval_dataloader: Dataloader = None,
        steps=None,
        log_steps_interval=1,
        eval_steps_interval=sys.maxsize,
        max_eval_samples=sys.maxsize,  # entire batch
        tensorboard_dir=None,
        profiler=None,
    ):
        # Initialize variables
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.preprocessor = preprocessor
        self.steps = steps
        self.tensorboard_dir = tensorboard_dir
        self.step_count = 0
        self.epoch_count = 0
        self.eval_steps_interval = eval_steps_interval
        self.max_eval_samples = max_eval_samples
        self.log_steps_interval = log_steps_interval
        self.global_batch_size = train_dataloader.global_batch_size
        self.profiler = profiler

        self.optimizer.build(self.model.trainable_variables)
        self.callbacks = self._create_callbacks()
        self.train_step = self._make_train_step()
        self.eval_step = self._make_eval_step()

        self.data_sharding = global_state.get_global_attribute(
            "DATA_SHARDING", DataSharding["fully_replicated"]
        )

        self._print_run_summary()
        self._validate_setup()
        self._validate_memory_usage()

    @property
    def loss_fn(self):
        return keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            ignore_class=self.preprocessor.tokenizer.pad_token_id,
        )

    def compute_loss(self, trainable_variables, non_trainable_variables, x, y):
        """This method is stateless and is intended for use with jax.grad."""
        logits, non_trainable_variables = self.model.stateless_call(
            trainable_variables, non_trainable_variables, x
        )
        loss = self.loss_fn(y, logits)

        return loss, non_trainable_variables

    @property
    def grad_fn(self):
        return jax.value_and_grad(self.compute_loss, has_aux=True)

    def _train_step(self, state: Tuple[List[jax.Array]], data):
        """This is the training step function"""
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
        ) = state
        x, y = data["x"], data["y"]
        (loss, non_trainable_variables), grads = self.grad_fn(
            trainable_variables, non_trainable_variables, x, y
        )
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        return (
            loss,
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
            ),
        )

    def _eval_step(self, state: Tuple[List[jax.Array]], data):
        """This is the eval function"""
        (trainable_variables, non_trainable_variables, _) = state
        x, y = data["x"], data["y"]
        logits, non_trainable_variables = self.model.stateless_call(
            trainable_variables, non_trainable_variables, x, training=False
        )
        loss = self.loss_fn(y, logits)
        return logits, loss

    def _make_train_step(self):
        return jax.jit(self._train_step, donate_argnums=(0,))

    def _make_eval_step(self):
        return jax.jit(self._eval_step, donate_argnums=(0,))

    def _get_jax_state(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.model.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.model.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizer.variables])
        return tuple(state)
    
    
    def _form_global_array(self, path, array: np.ndarray) -> jax.Array:
        """Put local sharded array into local devices"""
        seq_len = array.shape[1]
        global_shape = (self.global_batch_size, seq_len)

        try:
            local_device_arrays = np.split(array, len(
                self.data_sharding.mesh.local_devices), axis=0)
        except ValueError as array_split_error:
            raise ValueError(
                f"Unable to put to devices shape {array.shape} with "
                f"local device count {len(self.data_sharding.mesh.local_devices)} "
                f"at {jtu.keystr(path)}"
            ) from array_split_error

        local_device_buffers = jax.device_put(
            local_device_arrays, self.data_sharding.mesh.local_devices)
        return jax.make_array_from_single_device_arrays(global_shape, self.data_sharding, local_device_buffers)

    def train(self):
        """Training loop"""

        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
        )

        # Callbacks
        self.callbacks.on_train_begin()

        while self.step_count < self.steps:
            self.epoch_count += 1
            self.callbacks.on_epoch_begin(self.epoch_count)

            epoch_loss = 0
            train_set_size = 0
            for batch_input in self.train_dataloader:
                if self.step_count >= self.steps:
                    break
                self.step_count += 1

                start_time = time.time()

                # Callbacks
                self.callbacks.on_train_batch_begin(self.step_count)

                # Prepare and shard input if needed
                batch_input = self._prepare_batch_input_for_training(
                    batch_input)
                self._validate_sharding_correctness(batch_input, state)

                # Training step
                loss, state = self.train_step(state, batch_input)
                epoch_loss += loss
                train_set_size += self.global_batch_size

                # Callbacks
                self.callbacks.on_train_batch_end(
                    self.step_count, {"loss": loss})

                jax.block_until_ready(loss)
                # Logging
                if self.step_count % self.log_steps_interval == 0:
                    step_time = time.time() - start_time
                    print(f"Training loss at step {self.step_count}: {loss}")
                    print(f"Step {self.step_count} took {step_time:.3f}s")
                    print(f"Tokens/s/device:", self.global_batch_size *
                          self.preprocessor.seq_len / (step_time * jax.device_count()))

                # Eval
                if self.step_count % self.eval_steps_interval == 0:
                    self.evaluate(state)

            epoch_loss = epoch_loss / train_set_size

            self.callbacks.on_epoch_end(
                self.epoch_count, {"epoch_loss": epoch_loss})
            print(f"Train epoch {self.epoch_count} loss : {epoch_loss}")

        self.callbacks.on_train_end()
        self._update_model_with_state(state)

    def evaluate(self, state=None):
        """Eval loop"""

        if state is None:
            state = self._get_jax_state(
                trainable_variables=True,
                non_trainable_variables=True,
                optimizer_variables=True,
            )
        # Callbacks
        self.callbacks.on_test_begin()

        eval_loss = 0
        eval_set_size = 0
        for step_i, batch_input in enumerate(self.eval_dataloader):
            if eval_set_size + self.global_batch_size > self.max_eval_samples:
                break
            
            start_time = time.time()
            # Prepare and shard input
            batch_input = self._prepare_batch_input_for_training(batch_input)
            self._validate_sharding_correctness(batch_input, state)

            # Eval step
            logits, loss = self.eval_step(state, batch_input)
            eval_loss += loss
            eval_set_size += self.global_batch_size

            # Logging
            if (step_i + 1) % self.log_steps_interval == 0:
                step_time = time.time() - start_time
                start_time = time.time()
                print(f"Eval step {step_i+1} took {step_time:.3f}s")
                print(f"Eval step {step_i+1} loss {loss}")

        # Callbacks
        eval_loss = eval_loss / eval_set_size
        self.callbacks.on_test_end({"loss": eval_loss})
        print("Eval loss: ", eval_loss)

    def _update_model_with_state(self, state):
        """Update model internal parameters with the provided state"""
        trainable_variables, non_trainable_variables, *_ = state
        for variable, value in zip(self.model.trainable_variables, trainable_variables):
            value = jax.lax.with_sharding_constraint(value, variable._layout)
            variable.assign(value)
        for variable, value in zip(
            self.model.non_trainable_variables, non_trainable_variables
        ):
            value = jax.lax.with_sharding_constraint(value, variable._layout)
            variable.assign(value)

    def _prepare_batch_input_for_training(self, batch: List[str]):
        """Convert raw text to model input for training."""
        per_host_bach_input = self.preprocessor.prepare_training_input(batch)

        return jtu.tree_map_with_path(self._form_global_array, per_host_bach_input)

    def _prepare_input_for_inference(self, prompt: str):
        """Convert raw text to model input for inference."""
        return self.preprocessor.prepare_inference_input(prompt)

    def generate(self, prompt: str, stop_token_ids: List[int]|str = "auto"):
        """Generate response in inference mode."""
        input = self._prepare_input_for_inference(prompt)

        if stop_token_ids == "auto":
            stop_token_ids = []
            if hasattr(preprocessor.tokenizer, "end_token_id"):
                stop_token_ids.append(preprocessor.tokenizer.end_token_id)
            if hasattr(preprocessor.tokenizer, "eos_token_id"):
                stop_token_ids.append(preprocessor.tokenizer.eos_token_id)
            # Some models like Llama3 use two end tokens: <|eot_id|> in
            # "instruct" versions and <|end_of_text|> in others.
            if hasattr(preprocessor.tokenizer, "end_token2_id"):
                stop_token_ids.append(preprocessor.tokenizer.end_token2_id)
            if hasattr(preprocessor.tokenizer, "eos_token2_id"):
                stop_token_ids.append(preprocessor.tokenizer.eos_token2_id)

        pred_ids = self.model.generate(
            input,
            stop_token_ids=stop_token_ids,
        )
        return self.preprocessor.tokenizer.decode(pred_ids["token_ids"][0])

    def save_model(self, filepath):
        """Save model weights in .h5 format"""
        self.model.save_weights(filepath)

    def _print_run_summary(self):
        # TODO: Implement more structured logging
        for attr_name, attr_value in vars(self).items():
            if attr_name == "preprocessor":
                continue
            print(attr_name, attr_value)

    def _create_callbacks(self):
        callbacks = []
        if self.tensorboard_dir:
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=self.tensorboard_dir, update_freq="batch"
                )
            )
        if self.profiler:
            callbacks.append(self.profiler)
        return keras.callbacks.CallbackList(callbacks, model=self.model)

    def _validate_sharding_correctness(self, data, state):
        try:

            assert (
                data["y"].shape[0] == self.global_batch_size
            ), f"Input batch dimension does not match global batch size: {data['y'].shape}"

            if not entire_tree_is_sharded(data):
                print(
                    "Warning: data is not sharded",
                    data["y"].shape,
                    data["y"].sharding,
                )
            for variable, value in zip(self.model.trainable_variables, state[0]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: trainable variable is not sharded",
                        get_size_in_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
            for variable, value in zip(self.model.non_trainable_variables, state[1]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: nontrainable variable is not sharded",
                        get_size_in_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
            for variable, value in zip(self.optimizer.variables, state[2]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: optimizer variable is not sharded",
                        get_size_in_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
        except Exception as e:
            print(e)

    def _validate_memory_usage(self):
        total_size = 0
        for v in self.model.trainable_variables:
            total_size += get_size_in_mb(v.value)

        for v in self.optimizer.variables:
            total_size += get_size_in_mb(v.value)

        live_arrays = jax.live_arrays()
        live_arrays_size = 0
        for v in live_arrays:
            live_arrays_size += get_size_in_mb(v)

        if (total_size != live_arrays_size):
            print(
                f"WARNING: Potential memory leakage. HBM usage is {live_arrays_size} MB but model and optimizer are only {total_size} MB in size.")
        else:
            print(
                f"No memory leakage detected. HBM usage ({live_arrays_size} MB) matches model and optimizer size ({total_size} MB).")

    def _validate_setup(self):
        assert (
            self.max_eval_samples >= self.global_batch_size
        ), "Number of eval examples must be greater or equal to global batch size"
