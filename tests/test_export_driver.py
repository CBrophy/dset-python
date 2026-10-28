import contextlib
import io
import os
import tempfile
import unittest
from decimal import Decimal

from csv_to_dset import infer_value, read_data, read_data_from_file
from csv_to_dset.csv_import_driver import _TAIL_BYTES
from csv_to_dset.program import main as csv_to_dset_main
from easymorph import ValueColumn, read_dset, write_dset
from easymorph.data import CellType, CompressedColumn, ConstantColumn, Dataset, Nothing
from easymorph.drivers import CancellationToken, DriverFactory, OperationCanceledError
from easymorph.drivers.import_driver import _BinaryReader

from .test_import_driver import (
    convert_dataset_to_csv,
    get_dataset_file_path,
    get_expected_file_path,
    import_expected_file,
)


def write_to_bytes(dataset):
    buffer = io.BytesIO()
    DriverFactory.get_export_driver().write_dataset(dataset, buffer)
    return buffer.getvalue()


def round_trip(dataset):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.dset")
        write_dset(dataset, path)
        return read_dset(path)


def read_items(data):
    """Returns (name, type, content) for every item in dset file bytes."""
    br = _BinaryReader(io.BytesIO(data))
    br.read_bytes(6)
    items = []
    while br.position < len(data):
        name = br.read_string()
        item_type = br.read_string()
        items.append((name, item_type, br.read_bytes(br.read_int64())))
    return items


class TestExportDriver(unittest.TestCase):
    def assert_same_as_source(self, file_name):
        with open(get_dataset_file_path(file_name), "rb") as f:
            expected = f.read()
        self.assertEqual(expected, write_to_bytes(read_dset(get_dataset_file_path(file_name))))

    def test_empty_dataset_matches_easymorph_output(self):
        self.assert_same_as_source("EmptyDataset.dset")

    def test_zero_length_dataset_matches_easymorph_output(self):
        self.assert_same_as_source("ZeroLengthDataset.dset")

    def test_mixed_dataset_round_trip(self):
        source = read_dset(get_dataset_file_path("MixedDataset.dset"))

        actual_csv = convert_dataset_to_csv(round_trip(source))

        self.assertEqual(import_expected_file("ExpectedMixedDataset.csv"), actual_csv)

    def test_mixed_dataset_column_blocks_match_easymorph_layout(self):
        # Vocabulary order differs from EasyMorph's, but vocabulary size, bit width and
        # therefore every column block's length should be identical.
        with open(get_dataset_file_path("MixedDataset.dset"), "rb") as f:
            expected_items = read_items(f.read())
        actual_items = read_items(write_to_bytes(read_dset(get_dataset_file_path("MixedDataset.dset"))))

        self.assertEqual(
            [(n, t, len(c)) for n, t, c in expected_items[2:]],
            [(n, t, len(c)) for n, t, c in actual_items[2:]],
        )

    def test_constant_columns(self):
        dataset = Dataset(
            [
                ValueColumn("Same", ["x", "x", "x"]),
                ValueColumn("Empty", [None, None, None]),
                ValueColumn("Mixed", ["x", None, "x"]),
            ],
            "",
        )

        result = round_trip(dataset)

        self.assertIsInstance(result.columns[0], ConstantColumn)
        self.assertIsInstance(result.columns[1], ConstantColumn)
        self.assertIsInstance(result.columns[2], CompressedColumn)
        self.assertEqual(["x", "x", "x"], [result.get_text(0, r) for r in range(3)])
        self.assertEqual([CellType.NOTHING] * 3, [result.get_type(1, r) for r in range(3)])
        self.assertEqual(["x", None, "x"], [result.get_text(2, r) for r in range(3)])

    def test_bit_widths(self):
        for unique_count in (2, 3, 4, 5, 255, 256, 257, 1000):
            with self.subTest(unique_count=unique_count):
                values = [f"v{i % unique_count}" for i in range(unique_count * 3 + 1)]

                result = round_trip(Dataset([ValueColumn("Col", values)], ""))

                self.assertEqual(values, [result.get_text(0, r) for r in range(len(values))])

    def test_numbers(self):
        values = [
            Decimal("0"),
            Decimal("-128"),
            Decimal("127"),
            Decimal("-32768"),
            Decimal("40996"),
            Decimal("-2147483648"),
            Decimal("2147483648"),
            Decimal("32.70"),
            Decimal("1.0"),
            Decimal("1"),
            Decimal("-0.0000000000000000000000000001"),
            Decimal("79228162514264337593543950335"),
            Decimal("1E+3"),
        ]

        result = round_trip(Dataset([ValueColumn("Num", values)], ""))

        actual = [result.get_number(0, r) for r in range(len(values))]
        self.assertEqual([format(v, "f") for v in values], [format(v, "f") for v in actual])

    def test_number_out_of_range_raises(self):
        dataset = Dataset([ValueColumn("Num", [Decimal("79228162514264337593543950336"), Decimal(1)])], "")

        with self.assertRaises(ValueError):
            write_to_bytes(dataset)

    def test_failed_export_keeps_existing_file(self):
        dataset = Dataset([ValueColumn("Num", [Decimal("1E+100"), Decimal(1)])], "")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.dset")
            with open(path, "w") as f:
                f.write("existing")

            with self.assertRaises(ValueError):
                write_dset(dataset, path)

            with open(path) as f:
                self.assertEqual("existing", f.read())
            self.assertEqual(["out.dset"], os.listdir(tmp))

    def test_cancellation(self):
        token = CancellationToken()
        token.cancel()

        with self.assertRaises(OperationCanceledError):
            DriverFactory.get_export_driver().write_dataset(Dataset([], ""), io.BytesIO(), token)

    def test_columns_of_different_lengths_raise(self):
        dataset = Dataset([ValueColumn("A", [1, 2]), ValueColumn("B", [1])], "")

        with self.assertRaisesRegex(ValueError, "same length"):
            write_to_bytes(dataset)

    def test_duplicate_column_names_raise(self):
        dataset = Dataset([ValueColumn("A", [1]), ValueColumn("B", [2]), ValueColumn("A", [3])], "")

        with self.assertRaisesRegex(ValueError, "Duplicate column names: A"):
            write_to_bytes(dataset)


class TestCsvImport(unittest.TestCase):
    def test_infer_value(self):
        cases = {
            "": Nothing.INSTANCE,
            "TRUE": True,
            "false": False,
            "42": Decimal("42"),
            "-3.50": Decimal("-3.50"),
            "0.5": Decimal("0.5"),
            "007": "007",
            "1e5": "1e5",
            "+1": "+1",
            " 1": " 1",
            ".5": ".5",
            "NaN": "NaN",
            "Infinity": "Infinity",
            "79228162514264337593543950336": "79228162514264337593543950336",
            "text": "text",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                actual = infer_value(text)
                self.assertEqual(expected, actual)
                self.assertIs(type(expected), type(actual))

        self.assertEqual("-3.50", format(infer_value("-3.50"), "f"))

    def test_expected_csv_round_trip(self):
        expected_csv = import_expected_file("ExpectedMixedDataset.csv")

        dataset = read_data(io.StringIO(expected_csv, newline=""), ";")
        result = round_trip(dataset)

        self.assertEqual(7, result.columns_count)
        self.assertEqual(26, result.rows_count)
        self.assertEqual(expected_csv, convert_dataset_to_csv(result))
        self.assertEqual(CellType.NUMBER, result.get_type(2, 0))
        self.assertEqual(CellType.BOOLEAN, result.get_type(1, 5))
        self.assertEqual(CellType.NOTHING, result.get_type(0, 0))

    def test_without_type_inference(self):
        dataset = read_data(io.StringIO("A,B\n1,TRUE\n,x\n"), infer_types=False)

        self.assertEqual(["1", "TRUE", "", "x"], [dataset.get_text(c, r) for r in range(2) for c in range(2)])

    def test_header_only(self):
        result = round_trip(read_data(io.StringIO("Foo,Bar,Baz\n")))

        self.assertEqual(["Foo", "Bar", "Baz"], [c.name for c in result.columns])
        self.assertEqual(0, result.rows_count)

    def test_empty_csv(self):
        result = round_trip(read_data(io.StringIO("")))

        self.assertEqual(0, result.columns_count)

    def test_short_rows_are_padded_and_long_rows_raise(self):
        dataset = read_data(io.StringIO("A,B\n1\n"))
        self.assertEqual(CellType.NOTHING, dataset.get_type(1, 0))

        with self.assertRaisesRegex(ValueError, "more values than there are columns"):
            read_data(io.StringIO("A,B\n1,2,3\n"))

    def test_trailing_blank_lines_are_ignored(self):
        cases = [
            ("A\n1\n", ["1"]),
            ("A\n1\n\n", ["1"]),
            ("A\n1\n\n\n\n", ["1"]),
            ("A\r\n1\r\n\r\n\r\n", ["1"]),
            ("A\n1" + "\n" * (3 * _TAIL_BYTES), ["1"]),
            # Blank lines between records and a quoted empty last value are data.
            ("A\n1\n\n2\n\n", ["1", None, "2"]),
            ('A\n1\n""\n\n', ["1", None]),
            ("A\n\n\n", []),
        ]
        for text, expected in cases:
            with self.subTest(text=text[:20]):
                dataset = read_data(io.StringIO(text, newline=""), infer_types=False)

                self.assertEqual(expected, [dataset.get_text(0, r) or None for r in range(dataset.rows_count)])

        dataset = read_data(io.StringIO("A,B\n1,2\n\n"))
        self.assertEqual(1, dataset.rows_count)

    def test_trailing_blank_lines_in_file_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.csv")
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbfA,B\r\n1,2\r\n\r\n")

            self.assertEqual(1, read_data_from_file(path).rows_count)

    def test_file_name_is_not_a_glob_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "data[1].csv"), "w") as f:
                f.write("A\nliteral\n")
            with open(os.path.join(tmp, "data1.csv"), "w") as f:
                f.write("A\nglob match\n")

            dataset = read_data_from_file(os.path.join(tmp, "data[1].csv"))

            self.assertEqual("literal", dataset.get_text(0, 0))

    def test_invalid_utf8_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.csv")
            with open(path, "wb") as f:
                f.write(b"A\n\xff\xfe\n")

            with self.assertRaisesRegex(ValueError, "utf-8"):
                read_data_from_file(path)

    def test_duplicate_column_names_raise(self):
        with self.assertRaises(ValueError):
            read_data(io.StringIO("A,B,A\n"))

    def test_quoted_values(self):
        dataset = read_data(io.StringIO('"Name";"Note"\n"a;b";"say ""hi""\nbye"\n'), ";")

        self.assertEqual("a;b", dataset.get_text(0, 0))
        self.assertEqual('say "hi"\nbye', dataset.get_text(1, 0))

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "out.dset")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = csv_to_dset_main(
                    [get_expected_file_path("ExpectedMixedDataset.csv"), destination, "--separator", ";"]
                )

            self.assertEqual(0, exit_code)
            self.assertIn("Finished", output.getvalue())
            self.assertEqual(
                import_expected_file("ExpectedMixedDataset.csv"), convert_dataset_to_csv(read_dset(destination))
            )


if __name__ == "__main__":
    unittest.main()
