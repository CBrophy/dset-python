import numpy as np
import polars as pl

from .encoded_column import EncodedColumn
from .vector import unpack_vector


class CompressedColumn(EncodedColumn):
    """Compressed column with unique vocabulary and vector with references to values from vocabulary."""

    def __init__(self, vector_bytes: bytes, vocabulary: pl.DataFrame, bit_width: int, length: int, name: str):
        super().__init__(name, vocabulary, None, length)
        # Bit-compressed vector array, decoded on first use
        self._vector_bytes = vector_bytes
        # Number of bits used to encode one vector index
        self._bit_width = bit_width

    def _get_indexes(self) -> np.ndarray:
        if self._indexes is None:
            self._indexes = unpack_vector(self._vector_bytes, self._bit_width, self.length)
            self._vector_bytes = b""
        return self._indexes
