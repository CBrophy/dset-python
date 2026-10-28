import os
import struct
from typing import Any, BinaryIO, List

from ..data import BlockMetadata, Column, CompressedColumn, ConstantColumn, Dataset
from .errors import DsetFormatError
from .import_config import CancellationToken, ImportConfig
from .vocabulary_codec import decode_vocabulary

# Version of dset file
VERSION00 = "HIBC00"
VERSION01 = "HIBC01"
# Constant which defines compressed type of column in file
COMPRESSED_COLUMN_TYPE_NAME = "Column.Compressed"
# Constant which defines constant type of column in file
CONSTANT_COLUMN_TYPE_NAME = "Column.Constant"
# Item types
FIELD_ITEM_TYPE = "EasyMorph.CompressedField.0"
TABLE_METADATA_ITEM_TYPE = "EasyMorph.TableMetadata.0"
COLUMNS_METADATA_ITEM_TYPE = "EasyMorph.ColumnsMetadata.0"


class _BinaryReader:
    """Reads primitives in the format written by .NET's System.IO.BinaryWriter."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    @property
    def position(self) -> int:
        return self.stream.tell()

    def read_bytes(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise EOFError("Unable to read beyond the end of the stream.")
        return data

    def _unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read_bytes(struct.calcsize(fmt)))[0]

    def read_byte(self) -> int:
        return self._unpack("<B")

    def read_int32(self) -> int:
        return self._unpack("<i")

    def read_int64(self) -> int:
        return self._unpack("<q")

    def read_7bit_encoded_int(self) -> int:
        result = 0
        for shift in range(0, 35, 7):
            b = self.read_byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
        raise DsetFormatError("Too many bytes in what should have been a 7-bit encoded integer.")

    def read_string(self) -> str:
        """Reads a length-prefixed UTF-8 string."""
        length = self.read_7bit_encoded_int()
        return self.read_bytes(length).decode("utf-8", errors="replace")


class ImportDriver:
    """Driver to import dataset."""

    def import_datasets(self, config: ImportConfig) -> List[Dataset]:
        """Imports datasets from the file given in config."""
        if config is None:
            raise ValueError("config should be not None")

        if not config.file_name or not config.file_name.strip():
            raise ValueError("config.file_name should be not empty")

        if not os.path.isfile(config.file_name):
            raise FileNotFoundError(f"File not found: {config.file_name}")

        columns: List[Column] = []
        table_name = ""

        with open(config.file_name, "rb") as fs:
            br = _BinaryReader(fs)
            file_length = os.fstat(fs.fileno()).st_size

            # Read version from file
            version = br.read_bytes(6).decode("utf-8", errors="replace") if file_length >= 6 else ""

            if version not in (VERSION00, VERSION01):
                raise DsetFormatError("Wrong file header format.")

            while br.position < file_length:
                config.token.throw_if_cancellation_requested()

                # Item name
                table_name = br.read_string()
                # Item type
                item_type = br.read_string()
                # Length of current block (field)
                block_length = br.read_int32() if version == VERSION00 else br.read_int64()

                block_start_position = br.position

                block_metadata = BlockMetadata(table_name, item_type, block_start_position, block_length)

                # Only non-encrypted field items are supported
                if item_type == FIELD_ITEM_TYPE:
                    columns.append(self._read_column(block_metadata, br, config.token))
                # Metadata items are ignored
                elif item_type not in (TABLE_METADATA_ITEM_TYPE, COLUMNS_METADATA_ITEM_TYPE):
                    raise DsetFormatError(f"Unsupported item type: {item_type}")

                # Seek to end of block
                fs.seek(block_start_position + block_length)

        return [Dataset(columns, table_name)]

    def _read_column(self, block_metadata: BlockMetadata, br: _BinaryReader, token: CancellationToken) -> Column:
        token.throw_if_cancellation_requested()

        # Read field name
        field_name = br.read_string()
        # Read column type Compressed / Constant
        column_type = br.read_string()

        # The rest of the block is parsed in memory.
        content_start = br.position
        content = br.read_bytes(block_metadata.block_start_position + block_metadata.block_length - content_start)

        if column_type == COMPRESSED_COLUMN_TYPE_NAME:
            return self._read_compressed_column(block_metadata, content, content_start, field_name, token)
        elif column_type == CONSTANT_COLUMN_TYPE_NAME:
            return self._read_constant_column(content, field_name)
        else:
            raise DsetFormatError(f"Unsupported column type: {column_type}")

    @staticmethod
    def _read_int32(content: bytes, pos: int) -> int:
        if pos + 4 > len(content):
            raise EOFError("Unable to read beyond the end of the stream.")
        return struct.unpack_from("<i", content, pos)[0]

    @staticmethod
    def _read_constant_column(content: bytes, field_name: str) -> ConstantColumn:
        # Read constant with its type
        vocabulary, pos = decode_vocabulary(content, 0, 1)
        # Read column length
        length = ImportDriver._read_int32(content, pos)

        return ConstantColumn(length, vocabulary, field_name)

    @staticmethod
    def _read_compressed_column(
        block_metadata: BlockMetadata, content: bytes, content_start: int, field_name: str, token: CancellationToken
    ) -> CompressedColumn:
        token.throw_if_cancellation_requested()

        # Read vocabulary
        vocabulary_length = ImportDriver._read_int32(content, 0)
        vocabulary, pos = decode_vocabulary(content, 4, vocabulary_length)
        # The max width of bit index in bits (-1 because we need max index).
        bit_width = max(vocabulary_length - 1, 0).bit_length()

        # Read vector length
        vector_length = ImportDriver._read_int32(content, pos)
        pos += 4

        # Handle columns with incorrect bit width.
        # For EasyMorph devs: See comments in the HIBC00\CompressedFieldParser.cs:ReadCompressedColumn()
        # method for details.

        vector_length_in_bits = vector_length * bit_width
        vector_length_in_bytes = (vector_length_in_bits + 7) // 8

        remaining_bytes_in_block = len(content) - pos

        # In case of remaining vector bytes treat bit width as incorrect
        # and try to increase it by 1.
        if remaining_bytes_in_block > vector_length_in_bytes:
            bit_width += 1
            vector_length_in_bits = vector_length * bit_width
            vector_length_in_bytes = (vector_length_in_bits + 7) // 8

            # JIC check that expected and recalculated actual block length are equal
            if vector_length_in_bytes != remaining_bytes_in_block:
                raise DsetFormatError("Recalculated vector length should be the same the remaining bytes number")

        number_of_bytes = (bit_width * vector_length + 7) // 8
        if number_of_bytes > remaining_bytes_in_block:
            raise EOFError("Unable to read beyond the end of the stream.")
        vector_bytes = content[pos : pos + number_of_bytes]

        # We can't detect a column with incorrect bit width (mentioned above)
        # by comparing vector length when vector contains only 1 byte.
        # But we can try to detect that that byte contains non-zero bits
        # after the expected length in bits.
        if vector_length_in_bytes == 1 and (vector_bytes[0] >> vector_length_in_bits) != 0:
            bit_width += 1

        return CompressedColumn(vector_bytes, vocabulary, bit_width, vector_length, field_name)
