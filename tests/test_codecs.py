import struct
import unittest

import numpy as np

from easymorph.data import CellType, make_vocabulary
from easymorph.data.vector import _CHUNK_ROWS, pack_vector, unpack_vector
from easymorph.drivers.vocabulary_codec import decode_vocabulary, encode_vocabulary


def reference_pack(indexes, bit_width):
    """Straightforward bit packing, least significant bit first."""
    value = 0
    for i, index in enumerate(indexes):
        value |= int(index) << (i * bit_width)
    return value.to_bytes((len(indexes) * bit_width + 7) // 8, "little")


class TestVector(unittest.TestCase):
    def test_matches_reference_packing(self):
        rng = np.random.default_rng(0)
        for bit_width in range(1, 32):
            with self.subTest(bit_width=bit_width):
                indexes = rng.integers(0, 2**bit_width, 101, dtype=np.uint64).astype(np.uint32)

                packed = pack_vector(indexes, bit_width)

                self.assertEqual(reference_pack(indexes, bit_width), packed)
                np.testing.assert_array_equal(indexes, unpack_vector(packed, bit_width, len(indexes)))

    def test_round_trip_across_chunks(self):
        rng = np.random.default_rng(1)
        length = 2 * _CHUNK_ROWS + 13
        for bit_width in (1, 7, 9, 20):
            with self.subTest(bit_width=bit_width):
                indexes = rng.integers(0, 2**bit_width, length, dtype=np.uint32)

                packed = pack_vector(indexes, bit_width)

                self.assertEqual((length * bit_width + 7) // 8, len(packed))
                np.testing.assert_array_equal(indexes, unpack_vector(packed, bit_width, length))

    def test_zero_width(self):
        self.assertEqual(b"", pack_vector(np.zeros(5, dtype=np.uint32), 0))
        np.testing.assert_array_equal(np.zeros(5), unpack_vector(b"", 0, 5))


def text_cell(text, symbol=b"\x20"):
    data = text.encode("utf-8")
    length = len(data)
    prefix = bytearray()
    while length >= 0x80:
        prefix.append((length & 0x7F) | 0x80)
        length >>= 7
    prefix.append(length)
    return symbol + bytes(prefix) + data


def decimal_cell(mantissa, scale, negative=False):
    flags = (scale << 16) | (0x80000000 if negative else 0)
    return b"\x10" + struct.pack("<IIII", mantissa & 0xFFFFFFFF, (mantissa >> 32) & 0xFFFFFFFF, mantissa >> 64, flags)


# (type, value, expected encoding)
CELLS = [
    (CellType.NOTHING, None, b"\x00"),
    (CellType.BOOLEAN, "TRUE", b"\x40"),
    (CellType.BOOLEAN, "FALSE", b"\x41"),
    (CellType.NUMBER, "0", b"\x01\x00"),
    (CellType.NUMBER, "-128", b"\x01\x80"),
    (CellType.NUMBER, "127", b"\x01\x7f"),
    (CellType.NUMBER, "128", b"\x02" + struct.pack("<h", 128)),
    (CellType.NUMBER, "-32768", b"\x02" + struct.pack("<h", -32768)),
    (CellType.NUMBER, "40996", b"\x04" + struct.pack("<i", 40996)),
    (CellType.NUMBER, "-2147483648", b"\x04" + struct.pack("<i", -(2**31))),
    (CellType.NUMBER, "2147483648", decimal_cell(2**31, 0)),
    (CellType.NUMBER, "32.70", decimal_cell(3270, 2)),
    (CellType.NUMBER, "-0.5", decimal_cell(5, 1, True)),
    (CellType.NUMBER, "0.000", decimal_cell(0, 3)),
    (CellType.NUMBER, "123456789012345678", decimal_cell(123456789012345678, 0)),
    (CellType.NUMBER, "9223372036854775808", decimal_cell(2**63, 0)),
    (CellType.NUMBER, "-7.9228162514264337593543950335", decimal_cell(2**96 - 1, 28, True)),
    (CellType.TEXT, "", text_cell("")),
    (CellType.TEXT, "Foo", text_cell("Foo")),
    (CellType.TEXT, "Тест ✓", text_cell("Тест ✓")),
    (CellType.TEXT, "x" * 127, text_cell("x" * 127)),
    (CellType.TEXT, "y" * 128, text_cell("y" * 128)),
    (CellType.TEXT, "z" * 20000, text_cell("z" * 20000)),
    (CellType.ERROR, "Error in 0", text_cell("Error in 0", b"\x80\x00\x00\x00\x00")),
]


class TestVocabularyCodec(unittest.TestCase):
    def test_encode_each_cell(self):
        for cell_type, value, expected in CELLS:
            with self.subTest(value=value):
                self.assertEqual(expected, encode_vocabulary(make_vocabulary([cell_type], [value])))

    def test_encode_and_decode_mixed_vocabulary(self):
        vocabulary = make_vocabulary([c[0] for c in CELLS], [c[1] for c in CELLS])
        expected = b"".join(c[2] for c in CELLS)

        encoded = encode_vocabulary(vocabulary)
        decoded, pos = decode_vocabulary(b"prefix" + encoded + b"suffix", 6, len(CELLS))

        self.assertEqual(expected, encoded)
        self.assertEqual(6 + len(encoded), pos)
        self.assertEqual(vocabulary.rows(), decoded.rows())

    def test_large_vocabulary_round_trip(self):
        rng = np.random.default_rng(2)
        count = 100_000
        types = rng.choice([CellType.NUMBER, CellType.TEXT, CellType.BOOLEAN, CellType.ERROR], count)
        values = []
        for i, cell_type in enumerate(types):
            if cell_type == CellType.NUMBER:
                values.append(str(int(rng.integers(-(10**12), 10**12))) + ("" if i % 3 else ".25"))
            elif cell_type == CellType.BOOLEAN:
                values.append("TRUE" if i % 2 else "FALSE")
            else:
                values.append(f"value {i} " * int(rng.integers(0, 30)))
        vocabulary = make_vocabulary(types.tolist(), values)

        encoded = encode_vocabulary(vocabulary)
        decoded, pos = decode_vocabulary(encoded, 0, count)

        self.assertEqual(len(encoded), pos)
        self.assertTrue(vocabulary.equals(decoded))

    def test_invalid_numbers_raise(self):
        for value in ("abc", "1e5", "79228162514264337593543950336", "0." + "1" * 29):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    encode_vocabulary(make_vocabulary([CellType.NUMBER], [value]))

    def test_unknown_symbol_raises(self):
        with self.assertRaisesRegex(Exception, "Unknown symbol type: 3"):
            decode_vocabulary(b"\x03", 0, 1)

    def test_truncated_input_raises(self):
        for data in (b"\x04\x00", text_cell("Foo")[:-1], b"\x20"):
            with self.subTest(data=data):
                with self.assertRaises(EOFError):
                    decode_vocabulary(data, 0, 1)


if __name__ == "__main__":
    unittest.main()
