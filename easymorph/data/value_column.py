from decimal import Decimal
from typing import Any, Iterable, Optional

from .cell_type import CellType, Nothing
from .column import Column


class ValueColumn(Column):
    """In-memory column built from plain Python values.

    Supported values: None or Nothing (Nothing), bool (Boolean), int or Decimal (Number), str (Text).
    """

    def __init__(self, name: str, values: Iterable[Any]):
        self._values = [self._normalize(v) for v in values]
        super().__init__(name, len(self._values))

    @staticmethod
    def _normalize(value: Any) -> Any:
        if value is None or isinstance(value, Nothing):
            return Nothing.INSTANCE
        # bool is checked before int because bool is a subclass of int
        if isinstance(value, (bool, str, Decimal)):
            return value
        if isinstance(value, int):
            return Decimal(value)
        raise TypeError(f"Unsupported value type: {type(value).__name__}")

    def _value_if(self, row_index: int, cls: type) -> Any:
        value = self._values[row_index]
        return value if isinstance(value, cls) else None

    def get_error(self, row_index: int) -> Optional[str]:
        return None

    def get_text(self, row_index: int) -> Optional[str]:
        return self._value_if(row_index, str)

    def get_number(self, row_index: int) -> Optional[Decimal]:
        return self._value_if(row_index, Decimal)

    def get_boolean(self, row_index: int) -> Optional[bool]:
        return self._value_if(row_index, bool)

    def get_nothing(self, row_index: int) -> Optional[Nothing]:
        return self._value_if(row_index, Nothing)

    def get_type(self, row_index: int) -> CellType:
        value = self._values[row_index]
        if isinstance(value, bool):
            return CellType.BOOLEAN
        if isinstance(value, str):
            return CellType.TEXT
        if isinstance(value, Decimal):
            return CellType.NUMBER
        return CellType.NOTHING
