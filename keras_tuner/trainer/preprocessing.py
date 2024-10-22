from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp

class DataPreparationStrategy(ABC):
    @abstractmethod
    def prepare_training_input(self, batch, tokenizer, seq_len, input_field):
        pass

    @abstractmethod
    def prepare_inference_input(self, prompt, tokenizer, seq_len):
        pass


class DefaultDataPreparationStrategy(DataPreparationStrategy):
    def prepare_training_input(self, batch, tokenizer, seq_len, input_field):
        def tokenize(text, padding="max_length"):
            return tokenizer(
                text,
                max_length=seq_len,
                padding=padding,
                padding_side="right",
                truncation=True,
                add_special_tokens=False,
                return_tensors="np",
            )

        inputs = [x for x in batch[input_field]]
        inputs = [f"<bos>{x}<eos>" for x in inputs]

        batch_padded = tokenize(inputs)

        input_ids = batch_padded["input_ids"]
        attention_mask = batch_padded["attention_mask"]
        input_ids = input_ids[:, :]
        attention_mask = attention_mask[:, :]
        labels = np.roll(input_ids, -1)
        labels[:, -1] = tokenizer.pad_token_id

        return {
            "x": {
                "token_ids": input_ids,
                "padding_mask": attention_mask,
            },
            "y": labels,
        }

    def prepare_inference_input(self, prompt, tokenizer, seq_len):
        """Convert input to model input for inference."""
        tokens = tokenizer(
            prompt,
            max_length=seq_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="np",
        )
        return {
            "token_ids": tokens["input_ids"],
            "padding_mask": tokens["attention_mask"],
        }

class MaxTextDataPreparationStrategy(DataPreparationStrategy):
    def prepare_training_input(self, batch, tokenizer, seq_len, input_field):
        def tokenize(text, padding="max_length"):
            return tokenizer(
                text,
                max_length=seq_len,
                padding=padding,
                padding_side="right",
                truncation=True,
                add_special_tokens=False,
                return_tensors="np",
            )

        inputs = [x.decode("utf-8") for x in batch[input_field].tolist()]
        inputs = [f"<bos>{x}<eos>" for x in inputs]

        batch_padded = tokenize(inputs)

        input_ids = batch_padded["input_ids"]
        attention_mask = batch_padded["attention_mask"]
        input_ids = input_ids[:, :]
        attention_mask = attention_mask[:, :]
        labels = np.roll(input_ids, -1)
        labels[:, -1] = tokenizer.pad_token_id
        positions = jnp.stack(
            [jnp.arange(seq_len, dtype=jnp.int32) for _ in range(len(inputs))]
        )

        return {
            "x": {
                "tokens": input_ids,
                "segment_ids": attention_mask,
                "positions": positions,
            },
            "y": labels,
        }

    def prepare_inference_input(self, prompt, tokenizer, seq_len):
        """Convert input to model input for inference."""
        tokens = tokenizer(
            prompt,
            max_length=seq_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="np",
        )
        positions = jnp.arange(seq_len, dtype=jnp.int32)

        return {
            "tokens": tokens["input_ids"],
            "segment_ids": tokens["attention_mask"],
            "positions": positions,
        }
