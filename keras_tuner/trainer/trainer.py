import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
import keras
from keras_tuner.preprocessor import Preprocessor
from typing import Any, Union
from keras_tuner.sharding.utils import (
    entire_tree_is_sharded,
    is_not_sharded_and_is_large,
    get_size_in_mb,
)
from typing import List, Tuple
from keras_tuner.model import Model
import time
import sys
from keras.src.backend.common import global_state
from keras_tuner.sharding._data_sharding import DataSharding
from keras_tuner.dataset import Dataloader


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
        max_eval_samples=None,
        tensorboard_dir=None,
        global_batch_size=None,
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
        self.global_batch_size = global_batch_size

        # Instantiates modules
        self.optimizer.build(self.model.trainable_variables)
        self.callbacks = self._create_callbacks()
        self.train_step = self._make_train_step()
        self.eval_step = self._make_eval_step()
        self.data_sharding = global_state.get_global_attribute(
            "DATA_SHARDING", DataSharding["fully_replicated"]
        )

        self._print_run_summary()

        assert (
            self.max_eval_samples >= self.global_batch_size
        ), "Number of eval examples must be greater or equal to global batch size"

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
                self.step_count += 1
                if self.step_count > self.steps:
                    break

                start_time = time.time()

                # Callbacks
                self.callbacks.on_train_batch_begin(self.step_count)

                # Prepare and shard input if needed
                batch_input = self._prepare_batch_input_for_training(batch_input)
                # Converts host local batch inputs to a globally sharded jax.Array.
                batch_input = (
                    jax.experimental.multihost_utils.host_local_array_to_global_array(
                        batch_input, self.data_sharding.mesh, self.data_sharding.spec
                    )
                )
                self._validate_sharding_correctness(batch_input, state)

                # Training step
                loss, state = self.train_step(state, batch_input)
                epoch_loss += loss
                train_set_size += len(batch_input)

                # Eval
                if self.step_count % self.eval_steps_interval == 0:
                    self.evaluate(state)

                # Logging
                if self.step_count % self.log_steps_interval == 0:
                    step_time = time.time() - start_time
                    print(f"Training loss at step {self.step_count}: {loss}")
                    print(f"Step {self.step_count} took {step_time:.3f}s")

                # Callbacks
                self.callbacks.on_train_batch_end(self.step_count, {"loss": loss})

            epoch_loss = epoch_loss / train_set_size

            self.callbacks.on_epoch_end(self.epoch_count, {"epoch_loss": epoch_loss})
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
            print("len(batch_input)", len(batch_input), "self.max_eval_samples", self.max_eval_samples)
            print("batch_input", batch_input)
            if eval_set_size + self.global_batch_size > self.max_eval_samples:
                break

            start_time = time.time()
            # Prepare and shard input
            batch_input = self._prepare_batch_input_for_training(batch_input)
            batch_input = (
                jax.experimental.multihost_utils.host_local_array_to_global_array(
                    batch_input, self.data_sharding.mesh, self.data_sharding.spec
                )
            )
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
        return self.preprocessor.prepare_training_input(batch)

    def _prepare_input_for_inference(self, prompt: str):
        """Convert raw text to model input for inference."""
        return self.preprocessor.prepare_inference_input(prompt)

    def generate(self, prompt: str):
        """Generate response in inference mode."""
        input = self._prepare_input_for_inference(prompt)
        pred_ids = self.model.generate(
            input,
            stop_token_ids=[self.preprocessor.tokenizer.eos_token_id],
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
