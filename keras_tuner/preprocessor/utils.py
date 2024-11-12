from typing import List, Dict
from transformers import AutoTokenizer as HFTokenizer
import numpy as np


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


def convert_iterable_to_list_of_string(batch):
    """Converts an iterable of mixed string/bytes to a list of strings.

    This utility function is useful when dealing with data that might come from different
    sources and might be encoded as bytes.

    Args:
        batch (Iterable): Input iterable containing strings or bytes objects.
            Can be any iterable (list, tuple, generator, etc.)

    Returns:
        List[str]: List where all elements have been converted to strings.
            Byte strings are decoded using UTF-8 encoding.

    Example:
        >>> result = convert_iterable_to_list_of_string(['hello', b'world'])
        >>> print(result)
        ['hello', 'world']
    """

    batch = list(batch) if not isinstance(batch, list) else batch
    batch = [x.decode("utf-8") if isinstance(x, bytes) else x for x in batch]
    return batch
