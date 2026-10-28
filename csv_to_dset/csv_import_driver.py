import codecs
import io
import os
from typing import Any, BinaryIO, Optional, TextIO, Union

import numpy as np
import polars as pl

from easymorph.data import CellType, Dataset, EncodedColumn, Nothing, make_vocabulary
from easymorph.data.vocabulary import VOCABULARY_SCHEMA, fits_net_decimal, from_vocabulary_value, number_parts
from easymorph.drivers import CancellationToken

# Plain decimal numbers as written by dset_to_csv. Exponents, a leading '+' and
# leading zeros (e.g. zip codes like 007) are not treated as numbers.
_NUMBER_PATTERN = r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
# Bytes read at a time when scanning the end of the input for blank lines.
_TAIL_BYTES = 4096


def _infer_vocabulary(texts: pl.Series) -> pl.DataFrame:
    """Converts CSV texts to a vocabulary: empty values become Nothing, TRUE/FALSE (any case)
    become Boolean, plain decimal numbers that fit a .NET decimal become Number, the rest Text."""
    text = pl.col("text")
    upper = text.str.to_uppercase()
    _, scale, digits = number_parts(text)
    is_number = text.str.contains(_NUMBER_PATTERN) & fits_net_decimal(scale, digits)
    # Negative zero ("-0.00") is stored as zero.
    number = pl.when(digits == "").then(text.str.strip_prefix("-")).otherwise(text)

    return (
        pl.DataFrame({"text": texts})
        .select(
            type=pl.when(text == "")
            .then(CellType.NOTHING)
            .when(upper.is_in(["TRUE", "FALSE"]))
            .then(CellType.BOOLEAN)
            .when(is_number)
            .then(CellType.NUMBER)
            .otherwise(CellType.TEXT),
            value=pl.when(text == "")
            .then(None)
            .when(upper.is_in(["TRUE", "FALSE"]))
            .then(upper)
            .when(is_number)
            .then(number)
            .otherwise(text),
        )
        .cast(VOCABULARY_SCHEMA)
    )


def _trailing_blank_lines(stream: BinaryIO) -> int:
    """Counts the blank lines at the end of the stream, which polars reads as rows of empty values."""
    end = stream.seek(0, os.SEEK_END)
    newlines = 0
    while end > 0:
        start = max(0, end - _TAIL_BYTES)
        stream.seek(start)
        chunk = stream.read(end - start)
        stripped = chunk.rstrip(b"\r\n")
        newlines += chunk.count(b"\n", len(stripped))
        if stripped:
            break
        end = start
    stream.seek(0)
    # The first line break ends the last record; each further one is a blank line.
    return max(newlines - 1, 0)


def infer_value(text: str) -> Any:
    """Converts CSV text to a typed cell value: Nothing, bool, Decimal or str."""
    vocabulary = _infer_vocabulary(pl.Series([text], dtype=pl.String))
    cell_type, value = from_vocabulary_value(*vocabulary.row(0))
    return Nothing.INSTANCE if cell_type == CellType.NOTHING else value


def read_data_from_file(
    file_name: str,
    separator: str = ",",
    infer_types: bool = True,
    encoding: str = "utf-8-sig",
    token: Optional[CancellationToken] = None,
) -> Dataset:
    """Reads CSV data from specified file. A UTF-8 BOM is skipped if present."""
    if codecs.lookup(encoding).name in ("utf-8", "utf-8-sig"):
        return _read(file_name, separator, infer_types, token)
    # Polars only reads UTF-8, so other encodings are converted first.
    with open(file_name, encoding=encoding, newline="") as file:
        return read_data(file, separator, infer_types, token)


def read_data(
    file: TextIO, separator: str = ",", infer_types: bool = True, token: Optional[CancellationToken] = None
) -> Dataset:
    """Reads CSV data from specified text stream. The first record holds the column names.

    With infer_types, empty values become Nothing, TRUE/FALSE become Boolean and plain
    decimal numbers become Number; everything else is Text. Without it every value is Text.
    """
    return _read(io.BytesIO(file.read().encode("utf-8")), separator, infer_types, token)


def _read(
    source: Union[str, BinaryIO], separator: str, infer_types: bool, token: Optional[CancellationToken]
) -> Dataset:
    token = token or CancellationToken()
    token.throw_if_cancellation_requested()

    if len(separator.encode("utf-8")) != 1:
        raise ValueError("Separator should be a single ASCII character")

    if isinstance(source, str):
        with open(source, "rb") as f:
            blank_lines = _trailing_blank_lines(f)
    else:
        blank_lines = _trailing_blank_lines(source)

    try:
        # Without a header every column is read as text, and duplicate names are kept as data.
        df = pl.read_csv(
            source,
            has_header=False,
            infer_schema=False,
            separator=separator,
            empty_string_is_null=False,
            raise_if_empty=False,
            # File names are paths, not glob patterns (e.g. "data[1].csv").
            glob=False,
        )
    except pl.exceptions.ComputeError as ex:
        message = str(ex).split("\n")[0]
        if "more fields" in message:
            message = "a row has more values than there are columns"
        raise ValueError(f"Malformed CSV: {message}") from None

    if df.height == 0:
        return Dataset([], "")

    df = df.fill_null("")
    column_names = list(df.row(0))
    df = df.slice(1)

    # Blank lines at the end of the file are not records.
    if 0 < blank_lines <= df.height:
        tail = df.tail(blank_lines)
        if tail.select(pl.all_horizontal(pl.all() == "")).to_series().all():
            df = df.head(df.height - blank_lines)

    duplicates = sorted({name for name in column_names if column_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate column names: {', '.join(duplicates)}")

    columns = []
    for name, series in zip(column_names, df.get_columns()):
        token.throw_if_cancellation_requested()

        if series.len() == 0:
            # replace_strict ignores return_dtype for empty input
            columns.append(EncodedColumn(name, make_vocabulary([], []), np.zeros(0, dtype=np.uint32), 0))
            continue

        # Types are inferred once per unique value, not once per row.
        uniques = series.unique(maintain_order=True)
        indexes = series.replace_strict(
            uniques, pl.int_range(0, len(uniques), eager=True), return_dtype=pl.UInt32
        ).to_numpy()
        if infer_types:
            vocabulary = _infer_vocabulary(uniques)
        else:
            vocabulary = make_vocabulary(pl.repeat(CellType.TEXT, len(uniques), eager=True), uniques)

        columns.append(EncodedColumn(name, vocabulary, indexes, len(indexes)))

    return Dataset(columns, "")
