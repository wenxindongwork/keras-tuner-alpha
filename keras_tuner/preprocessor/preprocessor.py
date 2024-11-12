from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


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


def convert_iterable_to_list_of_string(batch):
    batch = (
        list(batch)
        if not isinstance(batch, list)
        else batch
    )
    
    batch = [x.decode("utf-8") if isinstance(x, bytes) else x for x in batch]
    return batch
