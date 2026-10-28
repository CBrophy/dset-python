import os
from typing import Optional, TextIO

import polars as pl

from easymorph.data import CellType, Dataset
from easymorph.drivers import CancellationToken

_DOUBLE_QUOTE = '"'
# Default line terminator for write_data; streams opened in text mode translate it.
_NEW_LINE = "\n"
# Rows converted to text at a time.
_CHUNK_ROWS = 100_000


def _quote(value: str) -> str:
    return _DOUBLE_QUOTE + value.replace(_DOUBLE_QUOTE, _DOUBLE_QUOTE + _DOUBLE_QUOTE) + _DOUBLE_QUOTE


def write_data_to_file(
    dataset: Dataset, file_name: str, separator: str, token: Optional[CancellationToken] = None
) -> None:
    """Writes CSV data to specified file (UTF-8 with BOM). Lines end with the platform newline;
    line breaks inside values are written unchanged."""
    with open(file_name, "w", encoding="utf-8-sig", newline="") as file:
        write_data(dataset, file, separator, token, os.linesep)


def write_data(
    dataset: Dataset,
    file: TextIO,
    separator: str,
    token: Optional[CancellationToken] = None,
    line_terminator: str = _NEW_LINE,
) -> None:
    """Writes CSV data to specified text stream. Every value is quoted; there is no trailing newline."""
    token = token or CancellationToken()
    token.throw_if_cancellation_requested()

    width = dataset.columns_count
    if width == 0:
        return

    length = dataset.rows_count

    file.write(separator.join(_quote(column.name) for column in dataset.columns))
    if length > 0:
        file.write(line_terminator)

    # Vocabulary values are already in CSV form (numbers keep their scale, booleans are
    # TRUE/FALSE, Nothing is empty), so each unique value is quoted once and rows are
    # assembled by index.
    quoted = (
        pl.lit(_DOUBLE_QUOTE)
        + pl.col("value").fill_null("").str.replace_all(_DOUBLE_QUOTE, _DOUBLE_QUOTE * 2, literal=True)
        + pl.lit(_DOUBLE_QUOTE)
    )
    formatted = []
    for i, column in enumerate(dataset.columns):
        token.throw_if_cancellation_requested()
        vocabulary, indexes = column.get_vocabulary()
        unknown = ~vocabulary["type"].is_in(list(CellType))
        if unknown.any():
            raise ValueError("Unknown cell type")
        values = vocabulary.select(quoted).to_series()
        formatted.append(values.gather(indexes).alias(f"c{i}"))
    frame = pl.DataFrame(formatted)
    line = pl.concat_str(pl.all(), separator=separator)

    # Values.
    for start in range(0, length, _CHUNK_ROWS):
        token.throw_if_cancellation_requested()

        if start > 0:
            file.write(line_terminator)
        lines = frame.slice(start, _CHUNK_ROWS).select(line).to_series()
        file.write(lines.str.join(line_terminator).item())
