from typing import List, Dict
from transformers import AutoTokenizer as HFTokenizer
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=5) 
def initialize_tokenizer(tokenizer_handle):
    """Creates an HuggingFace AutoTokenizer with the tokenizer_handle."""
    if tokenizer_handle.startswith("hf://"):
        tokenizer_handle = tokenizer_handle.removeprefix("hf://")
    try:
        tokenizer = HFTokenizer.from_pretrained(tokenizer_handle)
    except ValueError as e:
        print("Tokenizer handle is not a valid HuggingFace tokenizer handle.")
    return tokenizer


def HFtokenize(
    text: List[str],
    tokenizer: HFTokenizer,
    seq_len: int,
    padding: str = "max_length",
) -> Dict[str, np.ndarray]:
    """Tokenizes text using a HuggingFace tokenizer with specific parameters.

    This function handles the tokenization of input text, applying padding and
    truncation as needed. It converts text into a format suitable for model input.

    Args:
        text (Union[str, List[str]]): Input text or list of texts to tokenize
        tokenizer (HFTokenizer): HuggingFace tokenizer instance
        seq_len (int): Maximum sequence length for tokenization
        padding (str, optional): Padding strategy. Defaults to "max_length".
            Options include:
            - "max_length": Pad to seq_len
            - "longest": Pad to longest sequence in batch
            - "do_not_pad": No padding

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'input_ids': Token IDs (shape: [batch_size, seq_len])
            - 'attention_mask': Mask indicating real vs padded tokens
                              (shape: [batch_size, seq_len])

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = HFtokenize(["hello world"], tokenizer, seq_len=10)
        >>> print(result['input_ids'].shape)
        (1, 10)
    """

    return tokenizer(
        text,
        max_length=seq_len,
        padding=padding,
        padding_side="right",
        truncation=True,
        add_special_tokens=False,
        return_tensors="np",
    )
