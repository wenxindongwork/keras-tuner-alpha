from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Union, Optional
from transformers import AutoTokenizer as HFTokenizer
from dataclasses import dataclass
from keras_tuner.preprocessor.protos import (
    SFTTextInput,
    PretrainingTextInput,
    PretrainingTrainingInput,
    PretrainingInferenceInput,
)


@dataclass
class Preprocessor(ABC):
    """Abstract base class for text preprocessing operations.

    This class defines the interface for preprocessors that convert raw text inputs
    into formats suitable for model training and inference.

    Attributes:
        tokenizer (Optional[Any]): Tokenizer instance for text tokenization
        seq_len (Optional[int]): Maximum sequence length for tokenization
        column_mapping (dict[str, str]): Mapping between source data columns and expected fields
        model_type (str): Type of model to preprocess for ("keras" or "maxtext")

    Note:
        Implementing classes must override prepare_training_input and prepare_inference_input
    """

    tokenizer: HFTokenizer
    seq_len: int
    column_mapping: Optional[Dict[str, List[str]]] = None
    model_type: str = "keras"

    @abstractmethod
    def prepare_training_input(self, batch: Dict[str, List[str]]):
        """Prepare raw input data for model training. The output of this function will be fed into
        `model.stateless_call`.

        Args:
            batch: Raw text to be preprocessed

        Returns:
            Processed data in format expected by `model.stateless_call`.
        """
        pass

    @abstractmethod
    def prepare_inference_input(self, prompt: List[str]):
        """
        Prepare raw input data for model inference.

        Args:
            prompt: Raw text to be preprocessed

        Returns:
            Processed data in format expected by `model.generate_step`
        """
        pass


@dataclass
class SFTPreprocessor(Preprocessor):
    """Preprocessor for Supervised Fine-Tuning (SFT) tasks.

    This preprocessor handles the conversion of prompt-response pairs into
    formats suitable for supervised fine-tuning of language models.

    Note:
        Expects input data to contain prompt-response pairs according to column_mapping.
        Column mapping must have "prompt" and "answer" keys
    """

    def __post_init__(self) -> None:
        if self.column_mapping is None:
            self.column_mapping = {"prompt": "prompt", "answer": "answer"}

    def prepare_training_input(self, batch: Dict[str, List[str]]) -> Dict:
        batch: SFTTextInput = SFTTextInput.from_dict(batch, self.column_mapping)
        batch: PretrainingTrainingInput = batch.to_model_input(
            self.tokenizer, self.seq_len
        )
        if self.model_type == "keras":
            return batch.for_keras_model()
        elif self.model_type == "maxtext":
            return batch.for_maxtext_model()
        else:
            raise ValueError("model_type must be either keras or maxtext")

    def prepare_inference_input(self, prompts: List[str]) -> Dict:
        tokens: Dict[str, np.ndarray] = self.tokenizer(
            prompts,
            max_length=self.seq_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="np",
        )
        batch = PretrainingInferenceInput(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        if self.model_type == "keras":
            return batch.for_keras_model()
        elif self.model_type == "maxtext":
            return batch.for_maxtext_model()
        else:
            raise ValueError("model_type must be either keras or maxtext")


@dataclass
class PretrainingPreprocessor(Preprocessor):
    """Preprocessor for language model pretraining tasks.

    This preprocessor handles the conversion of raw text into formats suitable
    for language model pretraining, where the model learns to predict the next token.

    Note:
        When input is a dictionary, column_mapping must has the "text" key.
    """

    def __post_init__(self):
        if self.column_mapping is None:
            self.column_mapping = {"text": "text"}

    def prepare_training_input(
        self, batch: Union[List[str], Dict[str, List[str]]]
    ) -> Dict:

        if isinstance(batch, list):
            batch: PretrainingTextInput = PretrainingTextInput(text=batch)
        elif isinstance(batch, dict):
            batch: PretrainingTextInput = PretrainingTextInput.from_dict(
                batch, self.column_mapping
            )
        batch: PretrainingTrainingInput = batch.to_model_input(
            self.tokenizer, self.seq_len
        )
        if self.model_type == "keras":
            return batch.for_keras_model()
        elif self.model_type == "maxtext":
            return batch.for_maxtext_model()
        else:
            raise ValueError("model_type must be either keras or maxtext")

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

        batch = PretrainingInferenceInput(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )

        if self.model_type == "keras":
            return batch.for_keras_model()
        elif self.model_type == "maxtext":
            return batch.for_maxtext_model()
        else:
            raise ValueError("model_type must be either keras or maxtext")
