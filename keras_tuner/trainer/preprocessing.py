from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional


@dataclass
class Preprocessor(ABC):
    tokenizer: Optional[Any] = None
    seq_len: Optional[int] = None
    input_field: str = None

    @abstractmethod
    def prepare_training_input(self, batch):
        pass

    @abstractmethod
    def prepare_inference_input(self, prompt):
        pass


class ContinuedPretrainingPreprocessor(Preprocessor):

    def prepare_training_input(self, batch: Dict[str, List[str]]) -> Dict:
        def tokenize(
            text: List[str], padding: str = "max_length"
        ) -> Dict[str, np.ndarray]:
            return self.tokenizer(
                text,
                max_length=self.seq_len,
                padding=padding,
                padding_side="right",
                truncation=True,
                add_special_tokens=False,
                return_tensors="np",
            )

        inputs: List[str] = [x for x in batch[self.input_field]]
        inputs = [f"<bos>{x}<eos>" for x in inputs]

        batch_padded: Dict[str, np.ndarray] = tokenize(inputs)

        input_ids: np.ndarray = batch_padded["input_ids"]
        attention_mask: np.ndarray = batch_padded["attention_mask"]
        input_ids = input_ids[:, :]
        attention_mask = attention_mask[:, :]
        labels: np.ndarray = np.roll(input_ids, -1)
        labels[:, -1] = self.tokenizer.pad_token_id

        return {
            "x": {
                "token_ids": input_ids,
                "padding_mask": attention_mask,
            },
            "y": labels,
        }

    def prepare_inference_input(self, prompt: str) -> Dict:
        """Convert input to model input for inference."""
        tokens: Dict[str, np.ndarray] = self.tokenizer(
            prompt,
            max_length=self.seq_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="np",
        )
        return {
            "token_ids": tokens["input_ids"],
            "padding_mask": tokens["attention_mask"],
        }


@dataclass
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
        inputs = (
            list(batch[self.input_field])
            if not isinstance(batch[self.input_field], list)
            else batch[self.input_field]
        )
        inputs = [x.decode("utf-8") if isinstance(x, bytes) else x for x in inputs]
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
