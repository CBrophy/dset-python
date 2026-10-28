import numpy as np
import polars as pl

from .encoded_column import EncodedColumn


class ConstantColumn(EncodedColumn):
    """Column with all same elements: a single-value vocabulary."""

    def __init__(self, length: int, vocabulary: pl.DataFrame, name: str):
        super().__init__(name, vocabulary, None, length)

    def _get_indexes(self) -> np.ndarray:
        if self._indexes is None:
            self._indexes = np.zeros(self.length, dtype=np.uint8)
        return self._indexes
