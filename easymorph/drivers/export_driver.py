import io
import os
import struct
from collections import Counter
from typing import BinaryIO, Optional, Tuple
from xml.sax.saxutils import escape

import numpy as np
import polars as pl

from ..data import CellType, Column, Dataset, make_vocabulary
from ..data.vector import pack_vector
from .import_config import CancellationToken
from .import_driver import (
    COLUMNS_METADATA_ITEM_TYPE,
    COMPRESSED_COLUMN_TYPE_NAME,
    CONSTANT_COLUMN_TYPE_NAME,
    FIELD_ITEM_TYPE,
    TABLE_METADATA_ITEM_TYPE,
    VERSION01,
)
from .vocabulary_codec import encode_vocabulary

# Metadata items are written the same way EasyMorph writes them.
TABLE_METADATA_ITEM_NAME = "TableMetadata"
COLUMNS_METADATA_ITEM_NAME = "ColumnsMetadata"
_TABLE_METADATA_XML = "<TableMetadata>\r\n  <AnalysisView />\r\n</TableMetadata>"
_XML_ATTRIBUTE_ENTITIES = {'"': "&quot;", "\r": "&#xD;", "\n": "&#xA;", "\t": "&#x9;"}


class _BinaryWriter:
    """Writes primitives in the format read by .NET's System.IO.BinaryReader."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def write_bytes(self, data: bytes) -> None:
        self.stream.write(data)

    def write_byte(self, value: int) -> None:
        self.stream.write(struct.pack("<B", value))

    def write_int32(self, value: int) -> None:
        self.stream.write(struct.pack("<i", value))

    def write_int64(self, value: int) -> None:
        self.stream.write(struct.pack("<q", value))

    def write_7bit_encoded_int(self, value: int) -> None:
        while value >= 0x80:
            self.write_byte((value & 0x7F) | 0x80)
            value >>= 7
        self.write_byte(value)

    def write_string(self, value: str) -> None:
        """Writes a length-prefixed UTF-8 string."""
        data = value.encode("utf-8")
        self.write_7bit_encoded_int(len(data))
        self.write_bytes(data)


class ExportDriver:
    """Driver to export dataset into a dset file."""

    def export_dataset(self, dataset: Dataset, file_name: str, token: Optional[CancellationToken] = None) -> None:
        """Writes dataset to the file. The file is replaced only once writing succeeds."""
        if not file_name or not file_name.strip():
            raise ValueError("file_name should be not empty")

        temp_file_name = file_name + ".tmp"
        try:
            with open(temp_file_name, "wb") as fs:
                self.write_dataset(dataset, fs, token)
            os.replace(temp_file_name, file_name)
        finally:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    def write_dataset(self, dataset: Dataset, stream: BinaryIO, token: Optional[CancellationToken] = None) -> None:
        """Writes dataset to a binary stream."""
        token = token or CancellationToken()
        token.throw_if_cancellation_requested()
        self._validate(dataset)

        bw = _BinaryWriter(stream)
        bw.write_bytes(VERSION01.encode("ascii"))

        self._write_item(bw, TABLE_METADATA_ITEM_NAME, TABLE_METADATA_ITEM_TYPE, self._metadata_content(_TABLE_METADATA_XML))
        self._write_item(
            bw, COLUMNS_METADATA_ITEM_NAME, COLUMNS_METADATA_ITEM_TYPE, self._metadata_content(self._columns_metadata_xml(dataset))
        )

        for column in dataset.columns:
            token.throw_if_cancellation_requested()
            self._write_item(bw, column.name, FIELD_ITEM_TYPE, self._column_content(column, token))

    @staticmethod
    def _validate(dataset: Dataset) -> None:
        lengths = sorted({c.length for c in dataset.columns})
        if len(lengths) > 1:
            raise ValueError(f"All columns should have the same length, found lengths: {', '.join(map(str, lengths))}")
        duplicates = sorted(name for name, count in Counter(c.name for c in dataset.columns).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate column names: {', '.join(duplicates)}")

    @staticmethod
    def _write_item(bw: _BinaryWriter, name: str, item_type: str, content: bytes) -> None:
        bw.write_string(name)
        bw.write_string(item_type)
        bw.write_int64(len(content))
        bw.write_bytes(content)

    @staticmethod
    def _metadata_content(xml: str) -> bytes:
        buffer = io.BytesIO()
        _BinaryWriter(buffer).write_string(xml)
        return buffer.getvalue()

    @staticmethod
    def _columns_metadata_xml(dataset: Dataset) -> str:
        if dataset.columns_count == 0:
            return "<ColumnsMetadata />"
        lines = ["<ColumnsMetadata>", "  <MetaColumns>"]
        for name in sorted(c.name for c in dataset.columns):
            lines.append(f'    <MetaColumn OriginalName="{escape(name, _XML_ATTRIBUTE_ENTITIES)}" Note="" />')
        lines += ["  </MetaColumns>", "</ColumnsMetadata>"]
        return "\r\n".join(lines)

    @staticmethod
    def _build_vocabulary(column: Column) -> Tuple[pl.DataFrame, np.ndarray, bool]:
        """Returns the unique values used by the column, an index into them per row, and whether
        any row is Nothing. Like EasyMorph, index 0 is always reserved for Nothing."""
        vocabulary, indexes = column.get_vocabulary()

        if len(indexes):
            used = np.bincount(indexes, minlength=vocabulary.height) > 0
        else:
            used = np.zeros(vocabulary.height, dtype=bool)
        rows = vocabulary.with_row_index("old").filter(pl.Series(used))
        has_nothing = bool((rows["type"] == CellType.NOTHING).any())

        values = rows.filter(pl.col("type") != CellType.NOTHING)
        unique_values = values.select("type", "value").unique(maintain_order=True).with_row_index("new", offset=1)
        mapping = values.join(unique_values, on=["type", "value"], how="left")

        remap = np.zeros(vocabulary.height, dtype=np.uint32)
        remap[mapping["old"].to_numpy()] = mapping["new"].to_numpy()
        result = pl.concat([make_vocabulary([CellType.NOTHING], [None]), unique_values.select("type", "value")])
        return result, remap[indexes], has_nothing

    def _column_content(self, column: Column, token: CancellationToken) -> bytes:
        buffer = io.BytesIO()
        bw = _BinaryWriter(buffer)
        bw.write_string(column.name)

        length = column.length
        vocabulary, vector, has_nothing = self._build_vocabulary(column)
        token.throw_if_cancellation_requested()

        # Columns without values or with a single unique value are stored as constants.
        unique_values = vocabulary if has_nothing else vocabulary.slice(1)
        if unique_values.height <= 1:
            bw.write_string(CONSTANT_COLUMN_TYPE_NAME)
            bw.write_bytes(encode_vocabulary(unique_values.head(1) if unique_values.height else vocabulary))
            bw.write_int32(length)
            return buffer.getvalue()

        bw.write_string(COMPRESSED_COLUMN_TYPE_NAME)
        bw.write_int32(vocabulary.height)
        bw.write_bytes(encode_vocabulary(vocabulary))
        bw.write_int32(length)
        bw.write_bytes(pack_vector(vector, (vocabulary.height - 1).bit_length()))
        return buffer.getvalue()
