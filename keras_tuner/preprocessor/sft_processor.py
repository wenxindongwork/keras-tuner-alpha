import numpy as np
from typing import Any, Dict, List
from keras_tuner.preprocessor.preprocessor import Preprocessor, convert_iterable_to_list_of_string
from dataclasses import dataclass

@dataclass
class SFTPreprocessor(Preprocessor):

    prompt_field:str = None
    target_field:str = None

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

        prompts: List[str] = convert_iterable_to_list_of_string(batch[self.prompt_field])
        answers: List[str] = convert_iterable_to_list_of_string(batch[self.target_field])

        x_token_ids, x_padding_masks, y_token_ids = [], [], []

        for prompt, answer in zip(prompts, answers):
            full_seq = tokenize(f"<bos>{prompt}{answer}<eos>")
            input_seq = tokenize(f"<bos>{prompt}", padding="do_not_pad")

            input_ids = full_seq['input_ids'][0]
            attention_mask = full_seq['attention_mask'][0]

            labels = np.roll(input_ids, -1)
            labels[-1] = self.tokenizer.pad_token_id
            labels[:len(input_seq['input_ids'][0])-1] = self.tokenizer.pad_token_id

            x_token_ids.append(input_ids[None, :])
            x_padding_masks.append(attention_mask[None, :])
            y_token_ids.append(labels[None, :])

        return {
            "x": {
                "token_ids": np.concatenate(x_token_ids),
                "padding_mask": np.concatenate(x_padding_masks),
            },
            "y": np.concatenate(y_token_ids)
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

