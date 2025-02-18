import numpy as np
from typing import Dict, List, Union, Optional
from kithara.dataset.dataset import Dataset
import ray


class PackedDataset(Dataset):
    """A dataset class that packs multiple sequences together on the fly.

    Example:
        The source dataset output samples of the following format, which
        are padded to the target sequence length.
        ```
        Sample 1:
            tokens: [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0]
            segment_ids: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            positions: [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0]

        Sample 2:
            tokens: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            segment_ids: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            positions: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        Sample 3:
            tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
            segment_ids: [1, 1, 1, 1, 1, 1,,1, 1, 1, 0, 0, 0]
            positions: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
        ```

        The packed dataset output samples of the following format:
        ```
        Sample 1:
            tokens: [1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0]
            segment_ids: [1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0]
            positions: [1, 2, 3, 4, 5, 1, 2, 3, 0, 0, 0, 0]

        Sample 2:
            tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
            segment_ids: [1, 1, 1, 1, 1, 1,,1, 1, 1, 0, 0, 0]
            positions: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
        ```

        Loss calculation remains the same as standard text completion tasks,
        as the loss mask will ignore padded tokens. During the forward pass,
        flash attention will use segment_ids to only calculate attention for
        tokens in the same segment.

    Notes:
        - Packing must be used with Flash Attention enabled (which should be enabled by default).
        - Packing currently only works for MaxText models.
        - Packing does not currently work for DDP training, as DDP training requires every
            host to have the same number of data samples in the local dataset shard.

    Args:
        source_dataset (TextCompletionDataset): The source dataset containing unpacked sequences. The original
            dataset must be a TextCompletionDataset.
        pad_value (int, optional): The value to use for padding. Defaults to 0.

    Attributes:
        source_dataset (Dataset): The original unpacked dataset.
        pad_value (int): Value used for padding incomplete sequences.
        _buffer (Dict[str, np.ndarray]): Temporary storage for sequence packing.
        _buffer_is_full (bool): Flag indicating if current buffer is ready for output.
        _segment_id (int): segment_id for the next sequence to be added to the buffer.
        _current_position (int): Current sequence length, bounded by max_sequence_length.
    """

    def __init__(
        self,
        source_dataset: "TextCompletionDataset",
        pad_value: int = 0,
    ):
        super().__init__(source_dataset.source)

        self.source_dataset = source_dataset
        self.pad_value = pad_value

        # Initialize buffers for packing
        self._buffer = None
        self._buffer_is_full = False
        self._segment_id = 1
        self._current_position = 0

    def process_sample(self, input: Dict[str, any]) -> Dict[str, np.ndarray]:
        """Pack multiple sequences into a single fixed-length sequence
        with segmentation and position information.
        """
        
        from kithara.model import ModelImplementationType
        assert (
            self.model_type == ModelImplementationType.MAXTEXT
        ), f"Packing only works for MaxText models, and doesn not works for {self.model_type}"

        input_tokens = input["x"]["tokens"]
        input_segment_ids = input["x"]["segment_ids"]
        input_positions = input["x"]["positions"]
        targets = input["y"]
        target_length = targets.shape[-1]

        if self._buffer is None or self._buffer_is_full:
            self._buffer = input
            self._segment_id = 1
            self._current_position = 0
            self._buffer_is_full = False

        sequence_length = np.sum(input_segment_ids)

        # If we can't fit this sequence, return current buffer and start new one with this sequence
        if self._current_position + sequence_length > target_length:
            old_buffer = self._buffer
            # Start new buffer with current sequence
            self._buffer = input
            self._segment_id = 2
            self._current_position = sequence_length
            self._buffer_is_full = False
            return old_buffer

        # Add the sequence
        self._buffer["x"]["tokens"][
            :, self._current_position : self._current_position + sequence_length
        ] = input_tokens[:, :sequence_length]
        self._buffer["x"]["segment_ids"][
            :, self._current_position : self._current_position + sequence_length
        ] = (input_segment_ids[:, :sequence_length] * self._segment_id)
        self._buffer["x"]["positions"][
            :, self._current_position : self._current_position + sequence_length
        ] = input_positions[:, :sequence_length]
        self._buffer["y"][
            :, self._current_position : self._current_position + sequence_length
        ] = targets[:, :sequence_length]
        self._segment_id += 1
        self._current_position += sequence_length

        return None

    def __iter__(self):
        """Return an iterator over the dataset."""
        for sample in self.source_dataset:
            processed = self.process_sample(sample)
            if processed is not None:
                yield processed

        # Return the last buffer if it contains any data
        if self._buffer is not None and self._current_position > 0:
            yield self._buffer

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, delegate to source_dataset
            ds = object.__getattribute__(self, "source_dataset")
            return getattr(ds, name, None)
