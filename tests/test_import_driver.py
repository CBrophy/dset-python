import io
import os
import struct
import tempfile
import unittest
from decimal import Decimal
from unittest import mock

from dset_to_csv import write_data, write_data_to_file
from easymorph import ValueColumn
from easymorph.data import CellType, Dataset
from easymorph.drivers import CancellationToken, DriverFactory, ImportConfig, OperationCanceledError
from easymorph.drivers.vocabulary_codec import decode_vocabulary

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def get_dataset_file_path(file_name):
    return os.path.join(_TESTS_DIR, "source_datasets", file_name)


def get_expected_file_path(file_name):
    return os.path.join(_TESTS_DIR, "expected_files", file_name)


def import_source_dataset(dataset_file_name, token=None):
    driver = DriverFactory.get_import_driver()
    return driver.import_datasets(ImportConfig(get_dataset_file_path(dataset_file_name), token))


def import_expected_file(expected_file_name):
    with open(get_expected_file_path(expected_file_name), encoding="utf-8", newline="") as f:
        return f.read()


def convert_dataset_to_csv(dataset):
    writer = io.StringIO()
    write_data(dataset, writer, ";")
    return writer.getvalue()


class TestImportDriver(unittest.TestCase):
    def test_with_mixed_dataset(self):
        datasets = import_source_dataset("MixedDataset.dset")

        self.assertEqual(1, len(datasets))

        self.assertEqual(7, datasets[0].columns_count)
        self.assertEqual(26, datasets[0].rows_count)

        # The imported dataset is converted to CSV text and
        # compared with the expected CSV text stored in a file.
        actual_csv = convert_dataset_to_csv(datasets[0])
        expected_csv = import_expected_file("ExpectedMixedDataset.csv")

        self.assertEqual(expected_csv, actual_csv)

    def test_mixed_dataset_cell_values(self):
        dataset = import_source_dataset("MixedDataset.dset")[0]

        self.assertEqual(CellType.NOTHING, dataset.get_type(0, 0))
        self.assertEqual(CellType.TEXT, dataset.get_type(2, 0))
        self.assertEqual("32.53", dataset.get_text(2, 0))
        self.assertEqual(CellType.NUMBER, dataset.get_type(0, 2))
        self.assertEqual(Decimal("40996"), dataset.get_number(0, 2))
        self.assertIsNone(dataset.get_text(0, 2))
        self.assertEqual(CellType.BOOLEAN, dataset.get_type(1, 5))
        self.assertIs(True, dataset.get_boolean(1, 5))
        self.assertEqual(CellType.ERROR, dataset.get_type(0, 8))
        self.assertEqual("Error in 0", dataset.get_error(0, 8))

    def test_with_empty_dataset(self):
        datasets = import_source_dataset("EmptyDataset.dset")

        self.assertEqual(1, len(datasets))

        self.assertEqual(0, datasets[0].columns_count)
        self.assertEqual(0, datasets[0].rows_count)

    def test_with_zero_length_dataset(self):
        datasets = import_source_dataset("ZeroLengthDataset.dset")

        self.assertEqual(1, len(datasets))

        self.assertEqual(3, datasets[0].columns_count)
        self.assertEqual(0, datasets[0].rows_count)

        self.assertEqual(["Foo", "Bar", "Baz"], [c.name for c in datasets[0].columns])

    def test_with_dataset_with_wrong_bit_width(self):
        datasets = import_source_dataset("WrongBitWidth-DoNotResave.dset")

        self.assertEqual(1, len(datasets))

        self.assertEqual(1, datasets[0].columns_count)
        self.assertEqual(3, datasets[0].rows_count)

        self.assertEqual("Foo", datasets[0].get_text(0, 0))
        self.assertEqual("Bar", datasets[0].get_text(0, 1))
        self.assertEqual("", datasets[0].get_text(0, 2))

    def test_with_encrypted_file(self):
        with self.assertRaises(Exception) as ctx:
            import_source_dataset("EncryptedMixedDataset.dset")

        self.assertEqual("Unsupported item type: EasyMorph.EncryptedField.0", str(ctx.exception))

    def test_read_net_decimal(self):
        def net_decimal(mantissa, scale, negative):
            flags = (scale << 16) | (0x80000000 if negative else 0)
            return struct.pack(
                "<IIII", mantissa & 0xFFFFFFFF, (mantissa >> 32) & 0xFFFFFFFF, mantissa >> 64, flags
            )

        max_mantissa = 2**96 - 1
        cases = [
            (net_decimal(3270, 2, False), "32.70"),
            (net_decimal(15, 1, True), "-1.5"),
            (net_decimal(0, 3, False), "0.000"),
            (net_decimal(max_mantissa, 0, False), str(max_mantissa)),
            (net_decimal(max_mantissa, 28, True), "-7.9228162514264337593543950335"),
        ]
        for raw, expected in cases:
            vocabulary, pos = decode_vocabulary(b"\x10" + raw, 0, 1)
            self.assertEqual(17, pos)
            self.assertEqual((CellType.NUMBER, expected), vocabulary.row(0))

        # All cases in one vocabulary, decoded in bulk.
        vocabulary, _ = decode_vocabulary(b"".join(b"\x10" + raw for raw, _ in cases), 0, len(cases))
        self.assertEqual([expected for _, expected in cases], vocabulary["value"].to_list())

    def test_cancellation(self):
        token = CancellationToken()
        token.cancel()

        with self.assertRaises(OperationCanceledError):
            import_source_dataset("MixedDataset.dset", token)

    def test_write_data_to_file(self):
        dataset = import_source_dataset("MixedDataset.dset")[0]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            write_data_to_file(dataset, path, ";")
            with open(path, "rb") as f:
                content = f.read()

        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        expected = import_expected_file("ExpectedMixedDataset.csv").replace("\n", os.linesep)
        self.assertEqual(expected, content[3:].decode("utf-8"))

    def test_write_data_to_file_keeps_line_breaks_in_values(self):
        dataset = Dataset([ValueColumn("A", ["x\ny", "z"])], "")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            # Simulates Windows: lines end with CRLF, line breaks inside values are unchanged.
            with mock.patch.object(os, "linesep", "\r\n"):
                write_data_to_file(dataset, path, ",")
            with open(path, "rb") as f:
                content = f.read()

        self.assertEqual('\ufeff"A"\r\n"x\ny"\r\n"z"'.encode("utf-8"), content)


if __name__ == "__main__":
    unittest.main()
