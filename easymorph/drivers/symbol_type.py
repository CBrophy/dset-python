from enum import IntEnum


class SymbolType(IntEnum):
    """Type of symbols encoded in dset file."""

    NOTHING = 0
    INT8 = 1
    INT16 = 2
    INT32 = 4
    DECIMAL = 16
    TEXT = 32
    BOOL_TRUE = 64
    BOOL_FALSE = 65
    ERROR = 128
