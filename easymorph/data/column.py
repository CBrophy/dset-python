from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl

from .cell_type import CellType, Nothing
from .vocabulary import make_vocabulary, to_vocabulary_value

# A cell value with its type. The value of a Nothing cell is None.
Cell = Tuple[CellType, Any]


class Column(ABC):
    """Column with number of elements, where elements can have different types."""

    def __init__(self, name: str, length: int):
        self._name = name
        self._length = length

    @property
    def name(self) -> str:
        """Column name."""
        return self._name

    @property
    def length(self) -> int:
        """Column length."""
        return self._length

    def __len__(self):
        return self._length

    @abstractmethod
    def get_error(self, row_index: int) -> Optional[str]:
        """Element at row_index, None if element type is not Error."""

    @abstractmethod
    def get_text(self, row_index: int) -> Optional[str]:
        """Element at row_index, None if element type is not Text."""

    @abstractmethod
    def get_number(self, row_index: int) -> Optional[Decimal]:
        """Element at row_index, None if element type is not Number."""

    @abstractmethod
    def get_boolean(self, row_index: int) -> Optional[bool]:
        """Element at row_index, None if element type is not Boolean."""

    @abstractmethod
    def get_nothing(self, row_index: int) -> Optional[Nothing]:
        """Element at row_index, None if element type is not Nothing."""

    @abstractmethod
    def get_type(self, row_index: int) -> CellType:
        """Type of element at row_index."""

    def get_cell(self, row_index: int) -> Cell:
        """Type and value of element at row_index."""
        cell_type = self.get_type(row_index)
        if cell_type == CellType.NUMBER:
            return cell_type, self.get_number(row_index)
        if cell_type == CellType.TEXT:
            return cell_type, self.get_text(row_index)
        if cell_type == CellType.BOOLEAN:
            return cell_type, self.get_boolean(row_index)
        if cell_type == CellType.ERROR:
            return cell_type, self.get_error(row_index)
        return CellType.NOTHING, None

    def get_vocabulary(self) -> Tuple[pl.DataFrame, np.ndarray]:
        """Returns the column as (vocabulary, indexes): row i holds vocabulary row indexes[i].

        See easymorph.data.vocabulary for the vocabulary layout. The vocabulary may contain
        duplicates and rows no index refers to. Subclasses that already hold their values in
        this form override this cell-by-cell fallback.
        """
        types = []
        values = []
        for row in range(self._length):
            cell_type, value = self.get_cell(row)
            types.append(cell_type)
            values.append(to_vocabulary_value(cell_type, value))
        return make_vocabulary(types, values), np.arange(self._length, dtype=np.uint32)
