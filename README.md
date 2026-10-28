# EasyMorph dataset format (Python)
Open-source tools for reading and writing EasyMorph dataset files (.dset) and converting them to and from
CSV and Parquet, ported from the C# version.

[EasyMorph](https://easymorph.com) is a data preparation application designed for non-technical users. The .dset format is a data file format native to EasyMorph projects. Technically, it's a binary data file with columnar compression, where columns can mix values of different types - numbers, text, boolean, etc.

The repository contains:

  * `easymorph` — a Python library for reading and writing EasyMorph dataset files (.dset)
  * `dset_to_csv` — a command-line utility that converts a .dset file into a .csv file
  * `csv_to_dset` — a command-line utility that converts a .csv file into a .dset file
  * `dset_to_parquet` — a command-line utility that converts a .dset file into a Parquet file
  * `parquet_to_dset` — a command-line utility that converts a Parquet file into a .dset file

## Installation

Requires Python 3.10+, [polars](https://pola.rs) and numpy (installed automatically).

```
pip install .
```

To run from a checkout without installing the package, install just the dependencies:

```
pip install -r requirements.txt
```

## Performance

The .dset format stores each column as a vocabulary of unique values plus a bit-packed
index per row. The library works on that form directly: CSV and Parquet reading and
de-duplication are done by polars, bit packing by numpy, and type detection, number encoding
and CSV formatting run once per *unique* value as vectorized polars expressions. Python never
loops over rows.

As a guide, a 2M-row, 8-column CSV (108 MB, including a 100% unique ID column) converts to
.dset in about 3.5 s and back in about 2 s. A 2M-row, 7-column Parquet file (38 MB, with
three 100% unique columns: integers, floats and dates) converts to .dset in about 6 s and
back in about 6 s, of which about 4 s is writing or reading the .dset file. Columns with few
distinct values cost almost nothing; the remaining cost is proportional to the number of
unique values, and unique floats are the most expensive to convert.

The per-cell API (`get_number`, `get_text`, ...) is convenient for random access but is a
Python call per cell. For bulk work use `column.get_vocabulary()`, which returns
`(vocabulary, indexes)`: a polars DataFrame with `type` and `value` columns, and a numpy
array giving the vocabulary row of each data row. Vocabulary values are text: numbers in
exact fixed-point form keeping their scale (e.g. `"32.70"`), booleans as `TRUE`/`FALSE`,
and null for Nothing. For example, to get a column as a polars Series of strings:

```python
vocabulary, indexes = dataset.columns[0].get_vocabulary()
series = vocabulary["value"].gather(indexes)
```

## Library usage

```python
from easymorph import CellType, DriverFactory, ImportConfig, read_dset

dataset = read_dset("data.dset")
# or: DriverFactory.get_import_driver().import_datasets(ImportConfig("data.dset"))[0]

print(dataset.columns_count, dataset.rows_count)
for row in range(dataset.rows_count):
    for col in range(dataset.columns_count):
        cell_type = dataset.get_type(col, row)
        if cell_type == CellType.NUMBER:
            value = dataset.get_number(col, row)   # decimal.Decimal
        elif cell_type == CellType.TEXT:
            value = dataset.get_text(col, row)
        elif cell_type == CellType.BOOLEAN:
            value = dataset.get_boolean(col, row)
        elif cell_type == CellType.ERROR:
            value = dataset.get_error(col, row)
        else:  # CellType.NOTHING
            value = None
```

Each typed getter returns `None` when the cell is of a different type. Numbers are returned as `decimal.Decimal` to preserve the exact .NET decimal value and scale.

Imports can be canceled from another thread by passing a `CancellationToken` in `ImportConfig` and calling `token.cancel()`; the import then raises `OperationCanceledError`.

Encrypted datasets are not supported.

### Writing datasets

```python
from decimal import Decimal
from easymorph import Dataset, ValueColumn, write_dset

dataset = Dataset(
    [
        ValueColumn("Name", ["Alice", "Bob", None]),          # None is an empty cell (Nothing)
        ValueColumn("Amount", [Decimal("12.50"), 7, None]),  # int or Decimal
        ValueColumn("Active", [True, False, True]),
    ],
    "",
)
write_dset(dataset, "out.dset")
```

Any `Dataset` can be written, including one read with `read_dset`. Numbers must fit in a .NET decimal
(up to 28 decimal places, about ±7.9e28); otherwise `ValueError` is raised. The output file is only
replaced once the whole dataset has been written successfully.

### Converting from Python

The converters behind the command-line utilities can also be called directly:

```python
from csv_to_dset import read_data_from_file as read_csv
from dset_to_csv import write_data_to_file as write_csv
from dset_to_parquet import write_data_to_file as write_parquet
from parquet_to_dset import read_data_from_file as read_parquet
from easymorph import read_dset, write_dset

dataset = read_dset("data.dset")
write_csv(dataset, "data.csv", ",")
write_parquet(dataset, "data.parquet", compression="snappy")

write_dset(read_csv("data.csv"), "from_csv.dset")
write_dset(read_parquet("data.parquet"), "from_parquet.dset")
```

To skip the intermediate file, `parquet_to_dset.read_data(df)` converts a polars DataFrame into a
`Dataset`, and `dset_to_parquet.write_data(dataset)` returns `(df, mixed_columns)`: the DataFrame that
would be written and the names of its mixed-type columns (see [Parquet](#parquet)).

## Command-line usage

```
dset-to-csv source.dset destination.csv
csv-to-dset source.csv destination.dset [--separator ";"] [--encoding cp1252] [--no-infer-types]
dset-to-parquet source.dset destination.parquet [--compression zstd]
parquet-to-dset source.parquet destination.dset
# or without installing:
python -m dset_to_csv source.dset destination.csv
python -m csv_to_dset source.csv destination.dset
python -m dset_to_parquet source.dset destination.parquet
python -m parquet_to_dset source.parquet destination.dset
```

Each utility prints a warning when the destination file already exists, and then overwrites it.

### CSV

`dset-to-csv` writes UTF-8 with a BOM, comma separated, with every value quoted.

`csv-to-dset` reads UTF-8 by default (a BOM is optional), and uses the first row as column names. Unless
`--no-infer-types` is given, cell types are detected:

| CSV value | Cell type |
|---|---|
| empty | Nothing |
| `TRUE` / `FALSE` (any case) | Boolean |
| plain decimal number such as `42`, `-3.50` (decimal places are preserved) | Number |
| anything else, including `007`, `1e5`, `+1` | Text |

Converting a .dset to CSV and back preserves values, with two caveats inherent to CSV: text that looks
like a number becomes a number, and error cells become text.

### Parquet

Parquet columns have a single type, while .dset columns can mix types. `dset-to-parquet` writes each
column as follows (Nothing is always written as null):

| .dset column contains | Parquet type |
|---|---|
| only numbers | decimal(38, *s*), where *s* is the largest scale in the column |
| only booleans | boolean |
| only text, only errors, or only Nothing | string |
| a mix of types, or numbers that need more than 38 digits | string, in the same form as `dset-to-csv` |

The names of the mixed columns are stored in the Parquet file metadata (key `easymorph.mixed_columns`).

The output is compressed with zstd by default. `--compression` (or `-c`) selects another codec: `snappy`,
`gzip`, `brotli`, `lz4` or `uncompressed`.

`parquet-to-dset` converts Parquet types as follows (null is always Nothing):

| Parquet type | Cell type |
|---|---|
| boolean | Boolean |
| integer, decimal | Number |
| float | Number in fixed-point form, with at most 28 decimal places; NaN, infinities and values beyond ±7.9e28 become Text |
| string, categorical, enum | Text |
| date, time, datetime, duration | Text in ISO 8601 form, e.g. `2024-01-02`, `2024-01-02 03:04:05.000000`, `P1D` |
| binary, list, struct | not supported (`ValueError`) |

Text columns listed as mixed in the file metadata have their cell types detected the same way as
`csv-to-dset`. A .dset to Parquet round trip therefore keeps the types of single-type columns exactly,
while mixed columns have the same caveats as CSV.

## Running tests

```
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests -t .
```
