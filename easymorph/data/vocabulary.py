"""Columnar vocabulary: the unique values of a column, held as a polars DataFrame.

A vocabulary has two columns:
  type  (UInt8)  - the CellType of each value
  value (String) - TEXT/ERROR: the text; NUMBER: the exact number in fixed-point notation
                   keeping its scale (e.g. "32.70"); BOOLEAN: "TRUE" or "FALSE"; NOTHING: null
Numbers are kept as text so that their scale survives and no Python object is needed per value.
"""

from decimal import Decimal
from typing import Any, Iterable, Optional, Tuple

import polars as pl

from .cell_type import CellType

VOCABULARY_SCHEMA = {"type": pl.UInt8, "value": pl.String}

# The largest .NET System.Decimal mantissa (2^96 - 1) and scale.
MAX_DECIMAL_MANTISSA = "79228162514264337593543950335"
MAX_DECIMAL_SCALE = 28


def make_vocabulary(types: Iterable[int], values: Iterable[Optional[str]]) -> pl.DataFrame:
    return pl.DataFrame({"type": types, "value": values}, schema=VOCABULARY_SCHEMA)


def format_number(value: Decimal) -> str:
    """Fixed-point notation that keeps the scale (e.g. 32.70), like .NET's decimal.ToString()."""
    if value == 0:
        value = abs(value)
    return format(value, "f")


def to_vocabulary_value(cell_type: CellType, value: Any) -> Optional[str]:
    """Converts a typed Python value to its vocabulary text."""
    if cell_type == CellType.NUMBER:
        return format_number(value)
    if cell_type == CellType.BOOLEAN:
        return "TRUE" if value else "FALSE"
    if cell_type in (CellType.TEXT, CellType.ERROR):
        return value
    return None


def from_vocabulary_value(cell_type: int, value: Optional[str]) -> Tuple[CellType, Any]:
    """Converts vocabulary text to a typed Python value (Decimal, bool, str or None)."""
    cell_type = CellType(cell_type)
    if cell_type == CellType.NUMBER:
        return cell_type, Decimal(value)
    if cell_type == CellType.BOOLEAN:
        return cell_type, value == "TRUE"
    return cell_type, value


def number_parts(text: pl.Expr) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Splits number text like "-012.50" into (is_negative, scale, mantissa digits without
    leading zeros, so "" for zero)."""
    body = text.str.strip_prefix("-")
    scale = (
        pl.when(body.str.contains(".", literal=True))
        .then(body.str.len_bytes() - body.str.find(".", literal=True) - 1)
        .otherwise(0)
        .cast(pl.Int64)
    )
    digits = body.str.replace(".", "", literal=True).str.strip_chars_start("0")
    return text.str.starts_with("-"), scale, digits


def fits_net_decimal(scale: pl.Expr, digits: pl.Expr) -> pl.Expr:
    """True where a number with this scale and mantissa digits fits a .NET decimal."""
    length = digits.str.len_bytes()
    return (scale <= MAX_DECIMAL_SCALE) & (
        (length < len(MAX_DECIMAL_MANTISSA))
        | ((length == len(MAX_DECIMAL_MANTISSA)) & (digits <= MAX_DECIMAL_MANTISSA))
    )
