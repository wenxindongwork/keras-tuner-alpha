import numpy as np
from typing import Dict, List, Union, Optional
from kithara.dataset.dataset import Dataset
import ray


class PackedDataset(Dataset):
    """A dataset class that packs multiple sequences together for more efficient processing.

    This implementation packs multiple sequences into fixed-length segments, similar to how
    T5 and other transformer implementations handle packing. For each key in the input,
    it creates two additional fields: {key}_segmentation and {key}_position to track
    the original sequences.

    Args:
        source_dataset (Dataset): The source Kithara dataset to pack
        batch_size (int): Number of sequences to batch together before packing
        pad_value (int): Value to use for padding (default: -1)
    """

    def __init__(
        self,
        source_dataset: Dataset,
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
        """Pack multiple sequences into a single fixed-length sequence with segmentation and position information."""
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
        # If we can't fit this sequence, break
        if self._current_position + sequence_length > target_length:
            self._buffer_is_full = True
            return self._buffer

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
