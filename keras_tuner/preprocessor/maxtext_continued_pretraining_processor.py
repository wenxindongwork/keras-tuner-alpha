import numpy as np
from typing import Any, Dict, List, Union
from keras_tuner.preprocessor.preprocessor import Preprocessor, convert_iterable_to_list_of_string
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Union


class MaxTextContinuedPretrainingPreprocessor(Preprocessor):

    def _tokenize(self, text: Union[str, List[str]]) -> Dict:
        return self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            add_special_tokens=False,
            return_tensors="np",
        )

    def prepare_training_input(
        self,
        batch: Any,
    ) -> Dict:
        inputs: List[str] = convert_iterable_to_list_of_string(batch[self.input_field])

        inputs = [f"<bos>{x}<eos>" for x in inputs]

        inputs_tokenized = self._tokenize(inputs)
        input_ids = inputs_tokenized["input_ids"]
        attention_mask = inputs_tokenized["attention_mask"]
        labels = np.roll(input_ids, -1)
        labels[:, -1] = self.tokenizer.pad_token_id
        positions = jnp.stack(
            [jnp.arange(self.seq_len, dtype=jnp.int32) for _ in range(len(inputs))]
        )

        return {
            "x": {
                "tokens": input_ids,
                "segment_ids": attention_mask,
                "positions": positions,
            },
            "y": labels,
        }

    def prepare_inference_input(
        self,
        prompt: str,
    ) -> Dict:

        tokens = self._tokenize(prompt)
        positions = jnp.arange(self.seq_len, dtype=jnp.int32)

        return {
            "tokens": tokens["input_ids"],
            "segment_ids": tokens["attention_mask"],
            "positions": positions,
        }
