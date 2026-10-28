import json
from typing import List, Optional, Tuple

import numpy as np
import polars as pl

from easymorph.data import CellType, Column, Dataset
from easymorph.data.vocabulary import number_parts
from easymorph.drivers import CancellationToken

# The largest precision of a Parquet (and polars) decimal.
_MAX_DECIMAL_PRECISION = 38
# Parquet key-value metadata listing, as a JSON array, the columns written as text because
# their values have mixed types. parquet_to_dset detects the cell types of these columns again.
MIXED_COLUMNS_KEY = "easymorph.mixed_columns"
# Compression codecs supported by write_data_to_file.
COMPRESSIONS = ("zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed")
DEFAULT_COMPRESSION = "zstd"


def _number_dtype(numbers: pl.Series) -> Optional[pl.DataType]:
    """Decimal type with the largest scale used that holds every number exactly, or None if
    that needs more digits than a decimal allows."""
    _, scale, digits = number_parts(pl.col("value"))
    integer = (digits.str.len_bytes().cast(pl.Int64) - scale).clip(lower_bound=0)
    sizes = pl.DataFrame({"value": numbers}).select(scale=scale.max(), integer=integer.max()).row(0)
    scale, integer = (size or 0 for size in sizes)
    if integer + scale > _MAX_DECIMAL_PRECISION:
        return None
    return pl.Decimal(_MAX_DECIMAL_PRECISION, scale)


def _to_series(column: Column) -> Tuple[pl.Series, bool]:
    """Converts a column to a typed series. A column whose values are all numbers, all booleans
    or all text (ignoring Nothing) keeps that type, and a column of errors becomes text. A column
    that mixes types, or whose numbers need more digits than a decimal allows, becomes text and
    is reported as mixed."""
    vocabulary, indexes = column.get_vocabulary()
    unknown = ~vocabulary["type"].is_in(list(CellType))
    if unknown.any():
        raise ValueError("Unknown cell type")

    # Only vocabulary rows that some row refers to decide the type.
    used = np.bincount(indexes, minlength=vocabulary.height)[: vocabulary.height] > 0
    types = set(vocabulary["type"].filter(pl.Series(used)).unique().to_list()) - {CellType.NOTHING}

    values = vocabulary["value"]
    mixed = len(types) > 1
    if types == {CellType.NUMBER}:
        numbers = vocabulary.select(pl.when(pl.col("type") == CellType.NUMBER).then(pl.col("value"))).to_series()
        dtype = _number_dtype(numbers)
        if dtype is not None:
            values = numbers.cast(dtype)
        else:
            mixed = True
    elif types == {CellType.BOOLEAN}:
        booleans = pl.when(pl.col("type") == CellType.BOOLEAN).then(pl.col("value") == "TRUE")
        values = vocabulary.select(booleans).to_series()
    return values.gather(indexes).alias(column.name), mixed


def write_data(dataset: Dataset, token: Optional[CancellationToken] = None) -> Tuple[pl.DataFrame, List[str]]:
    """Converts a dataset into a polars DataFrame, as write_data_to_file does. Also returns the
    names of the columns written as text because their values have mixed types."""
    token = token or CancellationToken()
    series = []
    mixed_columns = []
    for column in dataset.columns:
        token.throw_if_cancellation_requested()
        values, mixed = _to_series(column)
        series.append(values)
        if mixed:
            mixed_columns.append(column.name)
    return pl.DataFrame(series), mixed_columns


def write_data_to_file(
    dataset: Dataset,
    file_name: str,
    token: Optional[CancellationToken] = None,
    compression: str = DEFAULT_COMPRESSION,
) -> None:
    """Writes a dataset to a Parquet file, compressed with one of COMPRESSIONS.

    A column whose values are all numbers becomes a decimal column with the largest scale used
    (or text if that needs more than 38 digits), all booleans a boolean column, and all text a
    text column; Nothing becomes null. Columns that mix types or contain errors are written as
    text, in the same form as dset_to_csv, and listed in the MIXED_COLUMNS_KEY metadata.
    """
    if compression not in COMPRESSIONS:
        raise ValueError(f"Unknown compression: {compression}. Expected one of: {', '.join(COMPRESSIONS)}")
    token = token or CancellationToken()
    token.throw_if_cancellation_requested()
    df, mixed_columns = write_data(dataset, token)
    metadata = {MIXED_COLUMNS_KEY: json.dumps(mixed_columns, ensure_ascii=False)} if mixed_columns else None
    df.write_parquet(file_name, compression=compression, metadata=metadata)
