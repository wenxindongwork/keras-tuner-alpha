import numpy as np
from typing import Any, Dict, List
from keras_tuner.preprocessor.preprocessor import Preprocessor, convert_iterable_to_list_of_string

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
        inputs: List[str] = convert_iterable_to_list_of_string(batch[self.input_field])
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

