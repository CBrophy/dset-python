import contextlib
import datetime
import io
import os
import tempfile
import unittest
from decimal import Decimal

import numpy as np
import polars as pl

from dset_to_parquet import write_data, write_data_to_file
from dset_to_parquet.parquet_export_driver import COMPRESSIONS, MIXED_COLUMNS_KEY
from dset_to_parquet.program import main as dset_to_parquet_main
from easymorph import ValueColumn, read_dset, write_dset
from easymorph.data import CellType, Dataset, EncodedColumn, make_vocabulary
from easymorph.drivers import CancellationToken, OperationCanceledError
from parquet_to_dset import read_data, read_data_from_file
from parquet_to_dset.program import main as parquet_to_dset_main

from .test_import_driver import convert_dataset_to_csv, get_dataset_file_path, import_expected_file


def parquet_round_trip(dataset):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.parquet")
        write_data_to_file(dataset, path)
        return read_data_from_file(path)


def cells(dataset, column=0):
    return [dataset.columns[column].get_cell(row) for row in range(dataset.rows_count)]


class TestParquetExport(unittest.TestCase):
    def test_column_types(self):
        dataset = Dataset(
            [
                ValueColumn("Numbers", [Decimal("32.70"), 7, None, Decimal("-0.5")]),
                ValueColumn("Booleans", [True, None, False, True]),
                ValueColumn("Texts", ["a", "", None, "b"]),
                ValueColumn("Mixed", [1, "a", True, None]),
                ValueColumn("Empty", [None, None, None, None]),
            ],
            "",
        )

        df, mixed_columns = write_data(dataset)

        self.assertEqual(
            {
                "Numbers": pl.Decimal(38, 2),
                "Booleans": pl.Boolean,
                "Texts": pl.String,
                "Mixed": pl.String,
                "Empty": pl.String,
            },
            dict(df.schema),
        )
        self.assertEqual([Decimal("32.70"), Decimal("7.00"), None, Decimal("-0.50")], df["Numbers"].to_list())
        self.assertEqual([True, None, False, True], df["Booleans"].to_list())
        self.assertEqual(["a", "", None, "b"], df["Texts"].to_list())
        self.assertEqual(["1", "a", "TRUE", None], df["Mixed"].to_list())
        self.assertEqual(["Mixed"], mixed_columns)

    def test_errors_are_written_as_text(self):
        vocabulary = make_vocabulary([CellType.ERROR, CellType.NOTHING], ["Division by zero", None])
        column = EncodedColumn("A", vocabulary, np.array([0, 1, 0], dtype=np.uint32), 3)

        df, mixed_columns = write_data(Dataset([column], ""))

        self.assertEqual(["Division by zero", None, "Division by zero"], df["A"].to_list())
        self.assertEqual([], mixed_columns)

    def test_unused_vocabulary_values_do_not_decide_the_type(self):
        vocabulary = make_vocabulary([CellType.NUMBER, CellType.TEXT], ["1.5", "unused"])
        column = EncodedColumn("A", vocabulary, np.array([0, 0], dtype=np.uint32), 2)

        df, _ = write_data(Dataset([column], ""))

        self.assertEqual(pl.Decimal(38, 1), df["A"].dtype)

    def test_numbers_too_wide_for_a_decimal_are_written_as_text(self):
        values = [Decimal("79228162514264337593543950335"), Decimal("0.0000000000000000000000000001")]
        dataset = Dataset([ValueColumn("A", values)], "")

        df, mixed_columns = write_data(dataset)

        self.assertEqual(pl.String, df["A"].dtype)
        self.assertEqual(["A"], mixed_columns)
        self.assertEqual(cells(dataset), cells(parquet_round_trip(dataset)))

    def test_mixed_columns_are_listed_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.parquet")
            write_data_to_file(read_dset(get_dataset_file_path("MixedDataset.dset")), path)

            self.assertEqual('["Дата", "Open", "Low", "Volume"]', pl.read_parquet_metadata(path)[MIXED_COLUMNS_KEY])

    def test_compression(self):
        # Distinct values, so that dictionary encoding alone does not make the file small.
        dataset = Dataset([ValueColumn("A", [f"repeated text value {i}" for i in range(10_000)])], "")
        sizes = {}
        with tempfile.TemporaryDirectory() as tmp:
            for compression in COMPRESSIONS:
                with self.subTest(compression=compression):
                    path = os.path.join(tmp, f"{compression}.parquet")
                    write_data_to_file(dataset, path, compression=compression)

                    self.assertEqual(cells(dataset), cells(read_data_from_file(path)))
                    sizes[compression] = os.path.getsize(path)

        self.assertGreater(sizes["uncompressed"], sizes["zstd"])

    def test_unknown_compression_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.parquet")

            with self.assertRaisesRegex(ValueError, "Unknown compression: lzo"):
                write_data_to_file(Dataset([ValueColumn("A", [1])], ""), path, compression="lzo")

            self.assertFalse(os.path.exists(path))

    def test_command_line_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = os.path.join(tmp, "out.parquet")
            source = get_dataset_file_path("MixedDataset.dset")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(0, dset_to_parquet_main([source, parquet, "--compression", "gzip"]))
                self.assertEqual(0, dset_to_parquet_main([source, parquet, "-c", "uncompressed"]))
                with self.assertRaises(SystemExit):
                    dset_to_parquet_main([source, parquet, "--compression", "lzo"])

            self.assertIn("invalid choice: 'lzo'", errors.getvalue())
            self.assertEqual(
                import_expected_file("ExpectedMixedDataset.csv"), convert_dataset_to_csv(read_data_from_file(parquet))
            )

    def test_cancellation(self):
        token = CancellationToken()
        token.cancel()

        with self.assertRaises(OperationCanceledError):
            write_data(read_dset(get_dataset_file_path("MixedDataset.dset")), token)


class TestParquetImport(unittest.TestCase):
    def test_mixed_dataset_round_trip(self):
        # Number-like text comes back as numbers and errors as text, which looks the same in CSV.
        result = parquet_round_trip(read_dset(get_dataset_file_path("MixedDataset.dset")))

        self.assertEqual(import_expected_file("ExpectedMixedDataset.csv"), convert_dataset_to_csv(result))
        self.assertEqual(CellType.NUMBER, result.get_type(0, 2))
        self.assertEqual(CellType.NOTHING, result.get_type(0, 0))

    def test_typed_columns_round_trip(self):
        dataset = Dataset(
            [
                ValueColumn("Numbers", [Decimal("32.70"), 7, None]),
                ValueColumn("Booleans", [True, None, False]),
                ValueColumn("Texts", ["a", "", None]),
            ],
            "",
        )

        result = parquet_round_trip(dataset)

        self.assertEqual(
            [(CellType.NUMBER, Decimal("32.70")), (CellType.NUMBER, Decimal("7.00")), (CellType.NOTHING, None)],
            cells(result, 0),
        )
        self.assertEqual(cells(dataset, 1), cells(result, 1))
        self.assertEqual(cells(dataset, 2), cells(result, 2))

    def test_empty_and_zero_length_datasets(self):
        self.assertEqual(0, parquet_round_trip(read_dset(get_dataset_file_path("EmptyDataset.dset"))).columns_count)

        result = parquet_round_trip(read_dset(get_dataset_file_path("ZeroLengthDataset.dset")))
        self.assertEqual(["Foo", "Bar", "Baz"], [c.name for c in result.columns])
        self.assertEqual(0, result.rows_count)

    def test_integers_and_decimals(self):
        df = pl.DataFrame(
            {
                "Int8": pl.Series([-5, None, -5], dtype=pl.Int8),
                "UInt64": pl.Series([2**64 - 1, 0, 1], dtype=pl.UInt64),
                "Decimal": pl.Series([Decimal("32.70"), Decimal("-0.00"), None], dtype=pl.Decimal(10, 2)),
            }
        )

        dataset = read_data(df)

        self.assertEqual(
            [(CellType.NUMBER, Decimal(-5)), (CellType.NOTHING, None), (CellType.NUMBER, Decimal(-5))], cells(dataset, 0)
        )
        self.assertEqual(Decimal(2**64 - 1), dataset.get_number(1, 0))
        self.assertEqual(
            [(CellType.NUMBER, Decimal("32.70")), (CellType.NUMBER, Decimal("0.00")), (CellType.NOTHING, None)],
            cells(dataset, 2),
        )

    def test_floats(self):
        cases = [
            (1.0, CellType.NUMBER, "1"),
            (1.5, CellType.NUMBER, "1.5"),
            (-0.0, CellType.NUMBER, "0"),
            (100.0, CellType.NUMBER, "100"),
            (0.1 + 0.2, CellType.NUMBER, "0.30000000000000004"),
            (1e-7, CellType.NUMBER, "0.0000001"),
            (-1.2345e22, CellType.NUMBER, "-12345000000000000000000"),
            (7.9e28, CellType.NUMBER, "79000000000000000000000000000"),
            # At most 28 decimal places are kept.
            (1.2345678901234567e-20, CellType.NUMBER, "0.0000000000000000000123456789"),
            (5e-324, CellType.NUMBER, "0"),
            # Not representable as a .NET decimal.
            (8e28, CellType.TEXT, "8e+28"),
            (float("nan"), CellType.TEXT, "NaN"),
            (float("inf"), CellType.TEXT, "inf"),
            (float("-inf"), CellType.TEXT, "-inf"),
            (None, CellType.NOTHING, None),
        ]
        dataset = read_data(pl.DataFrame({"A": [value for value, _, _ in cases]}, schema={"A": pl.Float64}))

        for row, (value, cell_type, text) in enumerate(cases):
            with self.subTest(value=value):
                expected = Decimal(text) if cell_type == CellType.NUMBER else text
                self.assertEqual((cell_type, expected), dataset.columns[0].get_cell(row))

    def test_float32_uses_its_shortest_form(self):
        dataset = read_data(pl.DataFrame({"A": pl.Series([0.1, 123456789.0], dtype=pl.Float32)}))

        self.assertEqual([Decimal("0.1"), Decimal("123456790")], [dataset.get_number(0, r) for r in range(2)])

    def test_text_and_temporal_columns(self):
        df = pl.DataFrame(
            {
                "String": ["a", "", None],
                "Categorical": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
                "Date": [datetime.date(2024, 1, 2), None, datetime.date(2024, 1, 2)],
                "Datetime": [datetime.datetime(2024, 1, 2, 3, 4, 5), None, None],
                "Time": [datetime.time(1, 2, 3), None, None],
                "Duration": [datetime.timedelta(days=1), None, datetime.timedelta(seconds=1)],
                "Null": [None, None, None],
            }
        )

        dataset = read_data(df)

        self.assertEqual([(CellType.TEXT, "a"), (CellType.TEXT, ""), (CellType.NOTHING, None)], cells(dataset, 0))
        self.assertEqual(["x", "y", "x"], [dataset.get_text(1, r) for r in range(3)])
        self.assertEqual(
            [(CellType.TEXT, "2024-01-02"), (CellType.NOTHING, None), (CellType.TEXT, "2024-01-02")], cells(dataset, 2)
        )
        self.assertEqual("2024-01-02 03:04:05.000000", dataset.get_text(3, 0))
        self.assertEqual("01:02:03", dataset.get_text(4, 0))
        self.assertEqual(["P1D", None, "PT1S"], [dataset.get_text(5, r) for r in range(3)])
        self.assertEqual([CellType.NOTHING] * 3, [dataset.get_type(6, r) for r in range(3)])

    def test_text_columns_keep_number_like_values_as_text(self):
        dataset = read_data(pl.DataFrame({"A": ["1", "TRUE"]}))

        self.assertEqual([(CellType.TEXT, "1"), (CellType.TEXT, "TRUE")], cells(dataset))

    def test_mixed_columns_detect_types(self):
        dataset = read_data(pl.DataFrame({"A": ["1.50", "TRUE", "x", None]}), mixed_columns=["A"])

        self.assertEqual(
            [
                (CellType.NUMBER, Decimal("1.50")),
                (CellType.BOOLEAN, True),
                (CellType.TEXT, "x"),
                (CellType.NOTHING, None),
            ],
            cells(dataset),
        )

    def test_unsupported_types_raise(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Parquet type of column A"):
            read_data(pl.DataFrame({"A": [[1, 2]]}))

    def test_file_name_is_not_a_glob_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            pl.DataFrame({"A": ["literal"]}).write_parquet(os.path.join(tmp, "data[1].parquet"))
            pl.DataFrame({"A": ["glob match"]}).write_parquet(os.path.join(tmp, "data1.parquet"))

            dataset = read_data_from_file(os.path.join(tmp, "data[1].parquet"))

            self.assertEqual("literal", dataset.get_text(0, 0))

    def test_cancellation(self):
        token = CancellationToken()
        token.cancel()

        with self.assertRaises(OperationCanceledError):
            read_data(pl.DataFrame({"A": [1]}), token=token)

    def test_command_line_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = os.path.join(tmp, "out.parquet")
            dset = os.path.join(tmp, "out.dset")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(0, dset_to_parquet_main([get_dataset_file_path("MixedDataset.dset"), parquet]))
                self.assertEqual(0, parquet_to_dset_main([parquet, dset]))
                self.assertEqual(1, parquet_to_dset_main([os.path.join(tmp, "missing.parquet"), dset]))

            self.assertIn("Finished", output.getvalue())
            self.assertIn("File does not exist", output.getvalue())
            self.assertEqual(import_expected_file("ExpectedMixedDataset.csv"), convert_dataset_to_csv(read_dset(dset)))

    def test_written_dset_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.dset")
            write_dset(read_data(pl.DataFrame({"A": [1.5, None], "B": ["x", "y"]})), path)

            result = read_dset(path)

            self.assertEqual([(CellType.NUMBER, Decimal("1.5")), (CellType.NOTHING, None)], cells(result))


if __name__ == "__main__":
    unittest.main()
