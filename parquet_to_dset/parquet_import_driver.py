import json
from typing import Collection, Optional

import numpy as np
import polars as pl

from easymorph.data import CellType, Dataset, EncodedColumn
from easymorph.data.vocabulary import (
    MAX_DECIMAL_SCALE,
    VOCABULARY_SCHEMA,
    fits_net_decimal,
    infer_vocabulary,
    number_parts,
)
from easymorph.drivers import CancellationToken

# Written by dset_to_parquet: the text columns whose cell types are detected on import.
MIXED_COLUMNS_KEY = "easymorph.mixed_columns"

# Float text as produced by polars: shortest round-trip form, possibly with an exponent
# (e.g. "1.5", "1e-7", "1.2345e+22"). NaN and infinities do not match.
_FLOAT_PATTERN = r"^(-?)([0-9]+)(?:\.([0-9]+))?(?:e([+-]?[0-9]+))?$"


def _fixed_point(texts: pl.Series) -> pl.Series:
    """Rewrites float text in fixed-point notation without trailing zeros, keeping at most
    MAX_DECIMAL_SCALE decimal places. Text that is not a finite float becomes null."""
    # Each step is materialized once; a single expression would repeat the regex for every use.
    digits, point, length = pl.col("digits"), pl.col("point"), pl.col("length")
    # Every branch is evaluated for every row, so widths and positions are kept non-negative.
    split = point.clip(0, length)
    fraction = pl.col("field_1")
    return (
        pl.DataFrame({"text": texts})
        .lazy()
        .select(pl.col("text").str.extract_groups(_FLOAT_PATTERN).alias("parts"))
        .unnest("parts")
        .select(
            sign=pl.col("1"),
            digits=pl.col("2") + pl.col("3").fill_null(""),
            # Position of the decimal point within digits.
            point=pl.col("2").str.len_bytes().cast(pl.Int64) + pl.col("4").cast(pl.Int64).fill_null(0),
        )
        .with_columns(length=digits.str.len_bytes().cast(pl.Int64))
        .select(
            "sign",
            fixed=pl.when(point <= 0)
            .then(pl.lit("0.") + digits.str.pad_start(length - split.clip(upper_bound=point), "0"))
            .when(point >= length)
            .then(digits.str.pad_end(point.clip(lower_bound=0), "0"))
            .otherwise(digits.str.slice(0, split) + pl.lit(".") + digits.str.slice(split)),
        )
        .select("sign", pl.col("fixed").str.split_exact(".", 1))
        .unnest("fixed")
        .with_columns(fraction.str.slice(0, MAX_DECIMAL_SCALE).str.strip_chars_end("0").fill_null(""))
        .select(
            pl.col("sign")
            + pl.col("field_0").str.strip_chars_start("0").str.pad_start(1, "0")
            + pl.when(fraction != "").then(pl.lit(".") + fraction).otherwise(pl.lit(""))
        )
        .collect()
        .to_series()
    )


def _number_vocabulary(texts: pl.Series, numbers: pl.Series) -> pl.DataFrame:
    """Vocabulary of numbers given as fixed-point text: those that fit a .NET decimal become
    Number, anything else (NaN, infinities, out of range) keeps its original text as Text."""
    number = pl.col("number")
    _, scale, digits = number_parts(number)
    is_number = pl.col("is_number")
    return (
        pl.DataFrame({"text": texts, "number": numbers})
        .lazy()
        .with_columns(digits=digits, is_number=number.is_not_null() & fits_net_decimal(scale, digits))
        .select(
            type=pl.when(is_number).then(CellType.NUMBER).otherwise(CellType.TEXT),
            # Negative zero ("-0.00") is stored as zero.
            value=pl.when(~is_number)
            .then(pl.col("text"))
            .when(pl.col("digits") == "")
            .then(number.str.strip_prefix("-"))
            .otherwise(number),
        )
        .collect()
    )


def _to_vocabulary(name: str, uniques: pl.Series) -> pl.DataFrame:
    """Converts the unique values of a Parquet column to a vocabulary. Nulls become Nothing."""
    dtype = uniques.dtype

    def of_type(cell_type: CellType, values: pl.Series) -> pl.DataFrame:
        return pl.DataFrame({"type": pl.repeat(cell_type, len(values), eager=True), "value": values})

    if dtype == pl.Null:
        vocabulary = of_type(CellType.NOTHING, pl.Series(uniques, dtype=pl.String))
    elif dtype == pl.Boolean:
        texts = uniques.replace_strict({True: "TRUE", False: "FALSE"}, return_dtype=pl.String)
        vocabulary = of_type(CellType.BOOLEAN, texts)
    elif dtype.is_integer() or dtype.is_decimal():
        texts = uniques.cast(pl.String)
        vocabulary = _number_vocabulary(texts, texts)
    elif dtype.is_float():
        texts = uniques.cast(pl.String)
        vocabulary = _number_vocabulary(texts, _fixed_point(texts))
    elif dtype == pl.Duration:
        vocabulary = of_type(CellType.TEXT, uniques.dt.to_string())
    elif dtype in (pl.String, pl.Categorical, pl.Enum) or dtype.is_temporal():
        vocabulary = of_type(CellType.TEXT, uniques.cast(pl.String))
    else:
        raise ValueError(f"Unsupported Parquet type of column {name}: {dtype}")

    is_null = uniques.is_null()
    return vocabulary.select(
        type=pl.when(is_null).then(CellType.NOTHING).otherwise(pl.col("type")),
        value=pl.when(is_null).then(None).otherwise(pl.col("value")),
    ).cast(VOCABULARY_SCHEMA)


def _to_column(name: str, series: pl.Series, mixed: bool) -> EncodedColumn:
    # Types are converted once per unique value, not once per row.
    uniques = series.unique(maintain_order=True).alias("value")
    if mixed and series.dtype == pl.String:
        vocabulary = infer_vocabulary(uniques.fill_null(""))
    else:
        vocabulary = _to_vocabulary(name, uniques)
    index = pl.DataFrame({"value": uniques, "index": pl.int_range(0, len(uniques), dtype=pl.UInt32, eager=True)})
    indexes = (
        pl.DataFrame({"value": series})
        .join(index, on="value", how="left", nulls_equal=True, maintain_order="left")["index"]
        .to_numpy()
    )
    return EncodedColumn(name, vocabulary, np.ascontiguousarray(indexes), len(indexes))


def read_data_from_file(file_name: str, token: Optional[CancellationToken] = None) -> Dataset:
    """Reads a Parquet file into a dataset.

    Nulls become Nothing, booleans Boolean, and integers, decimals and floats Number. Floats are
    stored in fixed-point notation with at most 28 decimal places; NaN, infinities and numbers
    outside the range of a .NET decimal become Text. Strings, dates, times and durations become
    Text (dates and times in ISO 8601 form). Nested and binary columns are not supported.

    Text columns that dset_to_parquet wrote for columns of mixed types have their cell types
    detected as csv_to_dset does: empty values become Nothing, TRUE/FALSE Boolean and plain
    decimal numbers Number.
    """
    token = token or CancellationToken()
    token.throw_if_cancellation_requested()
    metadata = pl.read_parquet_metadata(file_name)
    mixed_columns = json.loads(metadata[MIXED_COLUMNS_KEY]) if MIXED_COLUMNS_KEY in metadata else []
    return read_data(pl.read_parquet(file_name, glob=False), mixed_columns, token)


def read_data(
    df: pl.DataFrame, mixed_columns: Collection[str] = (), token: Optional[CancellationToken] = None
) -> Dataset:
    """Converts a polars DataFrame into a dataset, as read_data_from_file does. The cell types of
    the text columns named in mixed_columns are detected from their values."""
    token = token or CancellationToken()
    columns = []
    for series in df.get_columns():
        token.throw_if_cancellation_requested()
        columns.append(_to_column(series.name, series, series.name in mixed_columns))
    return Dataset(columns, "")
