from enum import IntEnum


class CellType(IntEnum):
    """Type of cell in dataset."""

    NUMBER = 0
    TEXT = 1
    BOOLEAN = 2
    NOTHING = 3
    ERROR = 4


class Nothing:
    """Nothing value (an empty cell). Use the ``Nothing.INSTANCE`` singleton."""

    INSTANCE: "Nothing"

    def __eq__(self, other):
        return isinstance(other, Nothing)

    def __hash__(self):
        return 0

    def __repr__(self):
        return "Nothing"


Nothing.INSTANCE = Nothing()
