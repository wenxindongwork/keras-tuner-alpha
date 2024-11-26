from dataclasses import dataclass
from typing import Union
from numpy.typing import NDArray
from jax.typing import ArrayLike

import jax.numpy as jnp
from typing import Dict, List
import numpy as np
from transformers import AutoTokenizer as HFTokenizer
from keras_tuner.preprocessor.utils import (
    convert_iterable_to_list_of_string,
    HFtokenize,
)
from jax.tree_util import tree_map

"""
This module provides data processing utilities for Supervised Fine-Tuning (SFT) and
pretraining of language models. It includes classes for handling text inputs and
converting them into model-ready formats.
"""


@dataclass
class SFTTextInput:
    """A class for processing supervised fine-tuning text data consisting of prompt-answer pairs.

    Attributes:
        prompts (List[str]): List of input prompts for the model
        answers (List[str]): List of corresponding target answers
    """

    prompts: List[str] = None
    answers: List[str] = None

    @classmethod
    def from_dict(
        cls, input_data: Dict[str, List[str]], column_mapping: Dict[str, str]
    ):
        """Creates an instance from a dictionary of input data using specified column mappings.

        Args:
            input_data (Dict[str, List[str]]): Dictionary containing the input data
            column_mapping (Dict[str, str]): Mapping of required fields to column names. 
            Column_mapping must has the "prompt" and "answer" keys.

        Returns:
            SFTTextInput: New instance with populated data
        """
        instance = cls()
        try:
            prompts = input_data[column_mapping["prompt"]]
            answers = input_data[column_mapping["answer"]]
        except KeyError as e:
            missing_key = str(e)
            raise KeyError(f"Missing required field: {missing_key}")

        assert len(prompts) == len(
            answers
        ), "Number of prompts must equal number of answers"

        prompts = convert_iterable_to_list_of_string(prompts)
        answers = convert_iterable_to_list_of_string(answers)

        instance.prompts = prompts
        instance.answers = answers

        return instance

    def to_model_input(
        self, tokenizer: HFTokenizer, seq_len: int
    ) -> "PretrainingTrainingInput":
        """Converts text data into tokenized format suitable for model training.

        Args:
            tokenizer (HFTokenizer): HuggingFace tokenizer instance
            seq_len (int): Maximum sequence length for tokenization

        Returns:
            PretrainingTrainingInput: Tokenized data with input_ids, label_ids, and attention_mask
        """

        def _single_sample_to_model_input(prompt, answer):
            full_seq = HFtokenize(
                f"<bos>{prompt}{answer}<eos>", tokenizer, seq_len=seq_len
            )
            prompt_seq = HFtokenize(
                f"<bos>{prompt}",
                tokenizer,
                seq_len=seq_len,
                padding="do_not_pad",
            )
            num_prompt_tokens = len(prompt_seq["input_ids"][0])

            input_ids = full_seq["input_ids"]  # [1, S]
            attention_mask = full_seq["attention_mask"]

            label_ids = np.roll(input_ids, -1)
            label_ids[:, -1] = tokenizer.pad_token_id
            label_ids[:, : num_prompt_tokens - 1] = tokenizer.pad_token_id
            return input_ids, label_ids, attention_mask

        samples = [
            _single_sample_to_model_input(prompt, answer)
            for prompt, answer in zip(self.prompts, self.answers)
        ]
        input_ids, label_ids, attention_mask = tree_map(
            lambda *arrays: np.concatenate(arrays), *samples
        )  # Concatenate list of [1,S] into [B, S]

        return PretrainingTrainingInput(
            input_ids=input_ids, label_ids=label_ids, attention_mask=attention_mask
        )


@dataclass
class PretrainingTrainingInput:
    """A class representing tokenized input data for model training.

    Attributes:
        input_ids (Union[NDArray, ArrayLike]): Token IDs for input sequences
        label_ids (Union[NDArray, ArrayLike]): Token IDs for target sequences
        attention_mask (Union[NDArray, ArrayLike]): Attention mask for input sequences

    Methods:
        for_keras_model: Formats the input data for KerasNLP models.
        for_maxtext_model: Formats the input data for MaxText models
    """

    input_ids: Union[NDArray, ArrayLike]
    label_ids: Union[NDArray, ArrayLike]
    attention_mask: Union[NDArray, ArrayLike]

    def for_keras_model(self):
        return {
            "x": {
                "token_ids": self.input_ids,
                "padding_mask": self.attention_mask,
            },
            "y": self.label_ids,
        }

    def for_maxtext_model(self):
        B, S = self.input_ids.shape
        positions = jnp.arange(S, dtype=jnp.int32)[None, :]
        positions = jnp.repeat(positions, B, axis=0)
        return {
            "x": {
                "tokens": self.input_ids,
                "segment_ids": self.attention_mask,
                "positions": positions,
            },
            "y": self.label_ids,
        }


@dataclass
class PretrainingInferenceInput:
    """A class representing tokenized input data for model inference.

    Attributes:
        input_ids (Union[NDArray, ArrayLike]): Token IDs for input sequences
        attention_mask (Union[NDArray, ArrayLike]): Attention mask for input sequences
    Methods:
        for_keras_model: Formats the input data for KerasNLP models.
        for_maxtext_model: Formats the input data for MaxText models
    """

    input_ids: Union[NDArray, ArrayLike]
    attention_mask: Union[NDArray, ArrayLike]

    def for_keras_model(self):
        return {
            "token_ids": self.input_ids,
            "padding_mask": self.attention_mask,
        }

    def for_maxtext_model(self):

        B, S = self.input_ids.shape
        positions = jnp.arange(S, dtype=jnp.int32)[None, :]
        positions = jnp.repeat(positions, B, axis=0)

        return {
            "tokens": self.input_ids,
            "segment_ids": self.attention_mask,
            "positions": positions,
        }


"""Pretraining"""


@dataclass
class PretrainingTextInput:
    """A class representing tokenized input data for pretraining.

    Attributes:
        input_ids (Union[NDArray, ArrayLike]): Token IDs for input sequences
        label_ids (Union[NDArray, ArrayLike]): Token IDs for target sequences
        attention_mask (Union[NDArray, ArrayLike]): Attention mask for input sequences
    """

    text: List[str] = None

    @classmethod
    def from_dict(
        cls, input_data: Dict[str, List[str]], column_mapping: Dict[str, str]
    ):
        """Creates an instance from a dictionary of input data using specified column mappings.

        Args:
            input_data (Dict[str, List[str]]): Dictionary containing the input data
            column_mapping (Dict[str, str]): Column_mapping must has the "text" key.

        Returns:
            PretrainingTextInput: New instance with populated data
        """

        instance = cls()
        try:
            text = input_data[column_mapping["text"]]
        except KeyError as e:
            missing_key = str(e)
            raise KeyError(f"Missing required field: {missing_key}")

        text = convert_iterable_to_list_of_string(text)
        instance.text = text
        return instance

    def to_model_input(
        self, tokenizer: HFTokenizer, seq_len: int
    ) -> PretrainingTrainingInput:
        """Converts text data into tokenized format suitable for model training.

        Args:
            tokenizer (HFTokenizer): HuggingFace tokenizer instance
            seq_len (int): Maximum sequence length for tokenization

        Returns:
            PretrainingTrainingInput: Tokenized data with input_ids, label_ids, and attention_mask
        """

        def _single_sample_to_model_input(text):
            full_seq = HFtokenize(
                f"<bos>{text}<eos>", tokenizer, seq_len=seq_len)
            input_ids = full_seq["input_ids"]
            attention_mask = full_seq["attention_mask"]

            label_ids = np.roll(input_ids, -1)
            label_ids[:, -1] = tokenizer.pad_token_id
            return input_ids, label_ids, attention_mask

        samples = [_single_sample_to_model_input(x) for x in self.text]
        input_ids, label_ids, attention_mask = tree_map(
            lambda *arrays: np.concatenate(arrays), *samples
        )
        return PretrainingTrainingInput(
            input_ids=input_ids, label_ids=label_ids, attention_mask=attention_mask
        )
