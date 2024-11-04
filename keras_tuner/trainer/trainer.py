import os

# Use Jax backend
os.environ["KERAS_BACKEND"] = "jax"
import jax
import keras
from functools import partial
from keras_tuner.trainer.preprocessing import Preprocessor
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


class Trainer:
    def __init__(
        self,
        model: Union[str, keras.Model],
        optimizer: keras.Optimizer,
        train_dataset: Any,
        preprocessor: Preprocessor = None,
        eval_dataset=None,
        steps=None,
        log_steps=0,
        sharding_strategy: ShardingStrategy =None,
    ):

        self.model = model
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.preprocessor = preprocessor

        self.log_steps = log_steps
        self.steps = steps
        self.step_count = 0

        self.sharding_strategy = sharding_strategy
        
        self._log_run_spec()

        self.optimizer.build(self.model.trainable_variables)
        self.train_step = self.make_train_step()

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

    def make_train_step(self):
        @partial(
            jax.jit,
            donate_argnums=(0,),
        )
        def compiled_train_step(state, data):
            return self._train_step(state, data)

        return compiled_train_step

    def _init_state(self) -> Tuple[List[jax.Array]]:
        trainable_variables = self.model.trainable_variables
        non_trainable_variables = self.model.non_trainable_variables
        optimizer_variables = self.optimizer.variables

        state = (
            [v.value for v in trainable_variables],
            [v.value for v in non_trainable_variables],
            [v.value for v in optimizer_variables],
        )
        return state

    def train(self):
        """ Training loop """
        state = self._init_state()
        start_time = time.time()
        while self.step_count < self.steps:
            for batch_input in self.train_dataset:
                
                self.step_count += 1
                if self.step_count > self.steps:
                    break

                # Prepare and shard input if needed
                batch_input = self._prepare_batch_input_for_training(batch_input)
                if self.sharding_strategy:
                    batch_input = jax.device_put(batch_input, self.sharding_strategy.data_sharding)
                    self._validate_sharding_correctness(batch_input, state)
                
                # Training step
                loss, state = self.train_step(state, batch_input)

                # Logging
                step_time = time.time() - start_time
                start_time = time.time()
                print(f"Step {self.step_count} took {step_time:.3f}s")
                if self.step_count % self.log_steps == 0:
                    print(f"Training loss at step {self.step_count}: {loss}")
                
        self._update_model_with_state(state)

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

    def _log_run_spec(self):
        #TODO: Implement more structured logging
        for attr_name, attr_value in vars(self).items():
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