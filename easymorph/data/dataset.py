from decimal import Decimal
from typing import List, Optional

from .cell_type import CellType, Nothing
from .column import Column


class Dataset:
    """Dataset which represents table with columns."""

    def __init__(self, columns: List[Column], name: str):
        self._columns = columns
        self._name = name

    @property
    def columns(self) -> List[Column]:
        """Dataset columns."""
        return self._columns

    @property
    def columns_count(self) -> int:
        """Number of columns in dataset."""
        return len(self._columns)

    @property
    def name(self) -> str:
        """Table name."""
        return self._name

    @property
    def rows_count(self) -> int:
        """Number of rows in dataset."""
        return self._columns[0].length if self._columns else 0

    def get_error(self, column: int, row_index: int) -> Optional[str]:
        """Table element, None if element type is not Error."""
        return self._columns[column].get_error(row_index)

    def get_text(self, column: int, row_index: int) -> Optional[str]:
        """Table element, None if element type is not Text."""
        return self._columns[column].get_text(row_index)

    def get_number(self, column: int, row_index: int) -> Optional[Decimal]:
        """Table element, None if element type is not Number."""
        return self._columns[column].get_number(row_index)

    def get_boolean(self, column: int, row_index: int) -> Optional[bool]:
        """Table element, None if element type is not Boolean."""
        return self._columns[column].get_boolean(row_index)

    def get_nothing(self, column: int, row_index: int) -> Optional[Nothing]:
        """Table element, None if element type is not Nothing."""
        return self._columns[column].get_nothing(row_index)

    def get_type(self, column: int, row_index: int) -> CellType:
        """Type of table element."""
        return self._columns[column].get_type(row_index)
