from decimal import Decimal
from typing import Optional, Tuple

import numpy as np
import polars as pl

from .cell_type import CellType, Nothing
from .column import Cell, Column
from .vocabulary import from_vocabulary_value


class EncodedColumn(Column):
    """Column stored as a vocabulary of unique values plus one vocabulary index per row."""

    def __init__(self, name: str, vocabulary: pl.DataFrame, indexes: Optional[np.ndarray], length: int):
        super().__init__(name, length)
        self._vocabulary = vocabulary
        self._indexes = indexes
        # Vocabulary as Python lists, created on first cell access
        self._types: Optional[list] = None
        self._values: Optional[list] = None

    def _get_indexes(self) -> np.ndarray:
        return self._indexes

    def get_vocabulary(self) -> Tuple[pl.DataFrame, np.ndarray]:
        return self._vocabulary, self._get_indexes()

    def _vocabulary_row(self, row_index: int) -> int:
        if self._types is None:
            self._types = self._vocabulary["type"].to_list()
            self._values = self._vocabulary["value"].to_list()
        if not 0 <= row_index < self.length:
            raise IndexError(f"Row index out of range: {row_index}")
        return int(self._get_indexes()[row_index])

    def _value_if(self, row_index: int, cell_type: CellType) -> Optional[str]:
        i = self._vocabulary_row(row_index)
        return self._values[i] if self._types[i] == cell_type else None

    def get_error(self, row_index: int) -> Optional[str]:
        return self._value_if(row_index, CellType.ERROR)

    def get_text(self, row_index: int) -> Optional[str]:
        return self._value_if(row_index, CellType.TEXT)

    def get_number(self, row_index: int) -> Optional[Decimal]:
        value = self._value_if(row_index, CellType.NUMBER)
        return None if value is None else Decimal(value)

    def get_boolean(self, row_index: int) -> Optional[bool]:
        value = self._value_if(row_index, CellType.BOOLEAN)
        return None if value is None else value == "TRUE"

    def get_nothing(self, row_index: int) -> Optional[Nothing]:
        return Nothing.INSTANCE if self.get_type(row_index) == CellType.NOTHING else None

    def get_type(self, row_index: int) -> CellType:
        i = self._vocabulary_row(row_index)
        return CellType(self._types[i])

    def get_cell(self, row_index: int) -> Cell:
        i = self._vocabulary_row(row_index)
        return from_vocabulary_value(self._types[i], self._values[i])
