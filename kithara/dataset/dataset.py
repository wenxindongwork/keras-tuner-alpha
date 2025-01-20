import ray
from typing import Any, Iterator, Iterable
from abc import ABC
class Dataset(Iterable, ABC):
    """A base dataset class that serves as a base for other dataset 
    implementations, including SFTDataset and TextCompletionDataset. 
    It supports both regular and streaming Ray datasets.

    Args:
        source (ray.data.Dataset): The underlying Ray dataset to wrap.
    
    """
    def __init__(self, source: ray.data.Dataset):
        self.source = source
        self._length = None
        self._iterator = None

    def process_sample(self, sample):
        return sample

    def __len__(self) -> int:
        if self._length is None:
            try:
                print(
                    ("Warning: If your dataset is a streaming dataset, "
                    "this operation might trigger its lazy executation.")
                )
                self._length = self.source.count()
            except:
                # For streamed datasets where count() might not work
                self._length = -1
        return self._length

    def __next__(self) -> Any:
        """Returns the next element in the dataset."""
        if self._iterator is None:
            self._iterator = iter(self.source.iter_rows())
        sample = next(self._iterator)
        return self.process_sample(sample)

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the dataset."""
        self._iterator = iter(self.source.iter_rows())
        return self
