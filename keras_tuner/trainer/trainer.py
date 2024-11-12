import os

# Use Jax backend
os.environ["KERAS_BACKEND"] = "jax"
import jax
import keras
from functools import partial
from keras_tuner.preprocessor import Preprocessor
from typing import Any, Union
from keras_tuner.trainer.sharding import (
    any_not_sharded_pytree,
    is_not_sharded_and_is_large,
    get_size_mb,
)
from keras_tuner.trainer.sharding import ShardingStrategy
from jax.ad_checkpoint import print_saved_residuals
from typing import List, Tuple
import time
import sys


class Trainer:
    def __init__(
        self,
        model: Union[str, keras.Model],
        optimizer: keras.Optimizer,
        train_dataset: Any,
        preprocessor: Preprocessor = None,
        eval_dataset=None,
        steps=None,
        log_steps=1,
        eval_steps=sys.maxsize,
        sharding_strategy: ShardingStrategy = None,
        tensorboard_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.preprocessor = preprocessor
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.steps = steps
        self.sharding_strategy = sharding_strategy
        self.optimizer.build(self.model.trainable_variables)
        self.train_step = self.make_train_step()
        self.eval_step = self.make_eval_step()
        self.callbacks = []
        if tensorboard_dir:
            self.callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=tensorboard_dir, update_freq="batch"
                )
            )
        self.callbacks = keras.callbacks.CallbackList(self.callbacks, model=self.model)

        self.step_count = 0
        self.epoch_count = 0
        self._print_run_summary()

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

    def make_train_step(self):
        return jax.jit(self._train_step, donate_argnums=(0,))

    def make_eval_step(self):
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
            for batch_input in self.train_dataset:
                self.step_count += 1
                if self.step_count > self.steps:
                    break

                start_time = time.time()

                # Callbacks
                self.callbacks.on_train_batch_begin(self.step_count)

                # Prepare and shard input if needed
                batch_input = self._prepare_batch_input_for_training(batch_input)
                if self.sharding_strategy:
                    batch_input = jax.device_put(
                        batch_input, self.sharding_strategy.data_sharding
                    )
                    self._validate_sharding_correctness(batch_input, state)

                # Training step
                loss, state = self.train_step(state, batch_input)
                epoch_loss += loss
                train_set_size += len(batch_input)
                self.model.optimizer["iterations"] += 1

                # Eval
                if self.step_count % self.eval_steps == 0:
                    self.evaluate(state)

                # Logging
                if self.step_count % self.log_steps == 0:
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
        for step_i, batch_input in enumerate(self.eval_dataset):

            start_time = time.time()

            # Prepare and shard input
            batch_input = self._prepare_batch_input_for_training(batch_input)
            if self.sharding_strategy:
                batch_input = jax.device_put(
                    batch_input, self.sharding_strategy.data_sharding
                )
                self._validate_sharding_correctness(batch_input, state)

            # Eval step
            logits, loss = self.eval_step(state, batch_input)
            eval_loss += loss
            eval_set_size += len(batch_input)

            # Logging
            if (step_i + 1) % self.log_steps == 0:
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

    def _validate_sharding_correctness(self, data, state):
        try:
            if any_not_sharded_pytree(data):
                print(
                    "Warning: data is not sharded",
                    data["y"].shape,
                    data["y"].sharding,
                )
            for variable, value in zip(self.model.trainable_variables, state[0]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: trainable variable is not sharded",
                        get_size_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
            for variable, value in zip(self.model.non_trainable_variables, state[1]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: nontrainable variable is not sharded",
                        get_size_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
            for variable, value in zip(self.optimizer.variables, state[2]):
                if is_not_sharded_and_is_large(value):
                    print(
                        f"Step {self.step_count}: optimizer variable is not sharded",
                        get_size_mb(value) + "mb",
                        variable.path,
                        value.shape,
                        value.sharding,
                    )
        except Exception as e:
            print(e)
