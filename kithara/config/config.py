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
from absl import flags


FLAGS = flags.FLAGS


flags.DEFINE_string("model_handle", "hf://google/gemma-2-9b", "The HF handle to the model, e.g. 'hf://google/gemma-2-9b'")
flags.DEFINE_string("tokenizer_handle", "hf://google/gemma-2-9b", "The HF handle to the tokenizer, e.g. 'hf://google/gemma-2-9b'")
flags.DEFINE_integer("seq_len", 1024, "The sequence length")
flags.DEFINE_string("precision", "mixed_bfloat16", "Precision mode for computations.")
flags.DEFINE_integer("training_steps", 200, "Number of training steps to run.")
flags.DEFINE_integer("eval_steps_interval", 100, "Number of eval steps per interval.")
flags.DEFINE_integer("log_steps_interval", 1, "Log steps interval.")
flags.DEFINE_integer("per_device_batch_size", 1, "Batch size per device.")
flags.DEFINE_integer("max_eval_samples", 50, "Maximum number of eval samples.")
flags.DEFINE_string("model_output_dir", "/tmp/", "Directory to output model checkpoints.")
flags.DEFINE_float("learning_rate", 5e-5, "Learning rate.")


def parse_config(): 
  config = {
    "model_handle": FLAGS.model_handle,
    "tokenizer_handle": FLAGS.tokenizer_handle,
    "seq_len": FLAGS.seq_len,
    "precision": FLAGS.precision,
    "training_steps": FLAGS.training_steps,
    "eval_steps_interval": FLAGS.eval_steps_interval,
    "log_steps_interval": FLAGS.log_steps_interval,
    "per_device_batch_size": FLAGS.per_device_batch_size,
    "max_eval_samples": FLAGS.max_eval_samples,
    "model_output_dir": FLAGS.model_output_dir,
    "learning_rate": FLAGS.learning_rate,
  }
  return config
