"""Vectorized conversion between vocabularies (see easymorph.data.vocabulary) and the
binary cell encoding of dset files: a SymbolType byte followed by the symbol's payload."""

from typing import List, Tuple

import numpy as np
import polars as pl

from ..data.cell_type import CellType
from ..data.vocabulary import VOCABULARY_SCHEMA, fits_net_decimal, number_parts
from .errors import DsetFormatError
from .symbol_type import SymbolType

# Total size in bytes of symbols without variable-length payload
_FIXED_SIZES = {
    SymbolType.NOTHING: 1,
    SymbolType.INT8: 2,
    SymbolType.INT16: 3,
    SymbolType.INT32: 5,
    SymbolType.DECIMAL: 17,
    SymbolType.BOOL_TRUE: 1,
    SymbolType.BOOL_FALSE: 1,
}
_SIZE_TABLE = np.zeros(256, dtype=np.int64)
_SIZE_TABLE[list(_FIXED_SIZES)] = list(_FIXED_SIZES.values())
_NUMBER_PATTERN = r"^-?[0-9]+(\.[0-9]+)?$"
# Strings copied into the output at a time, to bound temporary index arrays.
_TEXT_CHUNK_ROWS = 1 << 16


def _gather(data: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Returns the width bytes at each start as an (n, width) array."""
    return data[starts[:, None] + np.arange(width)]


def _format_decimal(mantissa: int, scale: int, negative: bool) -> str:
    digits = str(mantissa).rjust(scale + 1, "0")
    if scale:
        digits = digits[:-scale] + "." + digits[-scale:]
    return "-" + digits if negative and mantissa else digits


def _format_decimals(parts: np.ndarray) -> pl.Series:
    """Formats .NET decimals given as an (n, 4) array of lo, mid, hi, flags."""
    lo, mid, hi, flags = (parts[:, i].astype(np.int64) for i in range(4))
    scale = (flags >> 16) & 0xFF
    negative = (flags >> 31) & 1 == 1
    # Mantissas below 2^63 are formatted by polars, larger ones in Python.
    small = (hi == 0) & (mid < 2**31)

    frame = pl.DataFrame({"m": (mid << 32 | lo)[small], "scale": scale[small], "neg": negative[small]})
    digits = pl.col("m").cast(pl.String).str.zfill(pl.col("scale") + 1)
    point = digits.str.len_bytes().cast(pl.Int64) - pl.col("scale")
    unsigned = (
        pl.when(pl.col("scale") > 0)
        .then(digits.str.slice(0, point) + "." + digits.str.slice(point))
        .otherwise(digits)
    )
    sign = pl.when(pl.col("neg") & (pl.col("m") != 0)).then(pl.lit("-")).otherwise(pl.lit(""))
    formatted = frame.select(sign + unsigned).to_series()

    result = pl.repeat(None, len(parts), dtype=pl.String, eager=True)
    if small.any():
        result = result.scatter(np.flatnonzero(small), formatted)
    large = np.flatnonzero(~small)
    if len(large):
        result = result.scatter(
            large,
            [_format_decimal(int(hi[i]) << 64 | int(mid[i]) << 32 | int(lo[i]), int(scale[i]), bool(negative[i])) for i in large],
        )
    return result


def decode_vocabulary(buf: bytes, pos: int, count: int) -> Tuple[pl.DataFrame, int]:
    """Decodes count cells starting at buf[pos]. Returns the vocabulary and the position after it."""
    symbols: List[int] = []
    offsets: List[int] = []
    text_rows: List[int] = []
    texts: List[str] = []
    fixed_sizes = _FIXED_SIZES

    # Cells have variable length, so one pass finds where each starts. Texts are decoded here too.
    try:
        for i in range(count):
            symbol = buf[pos]
            symbols.append(symbol)
            offsets.append(pos)
            if symbol == SymbolType.TEXT or symbol == SymbolType.ERROR:
                # Error cells hold an unused int32 error code before the text.
                p = pos + (1 if symbol == SymbolType.TEXT else 5)
                length = buf[p]
                p += 1
                if length >= 0x80:
                    length &= 0x7F
                    shift = 7
                    while True:
                        b = buf[p]
                        p += 1
                        length |= (b & 0x7F) << shift
                        if b < 0x80:
                            break
                        shift += 7
                        if shift > 28:
                            raise DsetFormatError("Too many bytes in what should have been a 7-bit encoded integer.")
                pos = p + length
                text_rows.append(i)
                texts.append(buf[p:pos].decode("utf-8", errors="replace"))
            else:
                size = fixed_sizes.get(symbol)
                if size is None:
                    raise DsetFormatError(f"Unknown symbol type: {symbol}")
                pos += size
    except IndexError:
        raise EOFError("Unable to read beyond the end of the stream.") from None
    if pos > len(buf):
        raise EOFError("Unable to read beyond the end of the stream.")

    symbol_array = np.array(symbols, dtype=np.uint8)
    offset_array = np.array(offsets, dtype=np.int64)
    data = np.frombuffer(buf, dtype=np.uint8)
    types = np.full(count, CellType.NOTHING, dtype=np.uint8)
    values = pl.repeat(None, count, dtype=pl.String, eager=True)

    if text_rows:
        rows = np.array(text_rows, dtype=np.int64)
        types[rows] = np.where(symbol_array[rows] == SymbolType.TEXT, CellType.TEXT, CellType.ERROR)
        values = values.scatter(rows, pl.Series(texts, dtype=pl.String))

    rows = np.flatnonzero((symbol_array == SymbolType.BOOL_TRUE) | (symbol_array == SymbolType.BOOL_FALSE))
    if len(rows):
        types[rows] = CellType.BOOLEAN
        values = values.scatter(rows, np.where(symbol_array[rows] == SymbolType.BOOL_TRUE, "TRUE", "FALSE").tolist())

    for symbol, dtype, width in ((SymbolType.INT8, "<i1", 1), (SymbolType.INT16, "<i2", 2), (SymbolType.INT32, "<i4", 4)):
        rows = np.flatnonzero(symbol_array == symbol)
        if len(rows):
            numbers = _gather(data, offset_array[rows] + 1, width).view(dtype).ravel()
            types[rows] = CellType.NUMBER
            values = values.scatter(rows, pl.Series(numbers.astype(np.int64)).cast(pl.String))

    rows = np.flatnonzero(symbol_array == SymbolType.DECIMAL)
    if len(rows):
        types[rows] = CellType.NUMBER
        values = values.scatter(rows, _format_decimals(_gather(data, offset_array[rows] + 1, 16).view("<u4")))

    return pl.DataFrame({"type": types, "value": values}, schema=VOCABULARY_SCHEMA), pos


def _encode_numbers(values: pl.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the symbol, total size and a 16-byte payload buffer of each number."""
    negative, scale, digits = number_parts(pl.col("value"))
    parts = pl.DataFrame({"value": values}).select(
        valid=pl.col("value").str.contains(_NUMBER_PATTERN).fill_null(False),
        fits=fits_net_decimal(scale, digits).fill_null(False),
        negative=negative,
        scale=scale,
        digits=digits,
        length=digits.str.len_bytes().cast(pl.Int64),
        mantissa=digits.str.to_integer(strict=False).fill_null(0),
    )
    invalid = parts.filter(~pl.col("valid"))
    if invalid.height:
        raise ValueError(f"Invalid number: {values.filter(~parts['valid'])[0]}")
    if not parts["fits"].all():
        raise ValueError(f"Number is outside the range of a .NET decimal: {values.filter(~parts['fits'])[0]}")

    n = parts.height
    neg = parts["negative"].to_numpy()
    scale = parts["scale"].to_numpy()
    length = parts["length"].to_numpy()
    small = length <= 18
    mantissa = np.where(small, parts["mantissa"].to_numpy(), 0)

    # Integers without a fractional part use the smallest integer symbol.
    signed = np.where(neg, -mantissa, mantissa)
    integer = small & (scale == 0) & (signed >= -(2**31)) & (signed < 2**31)
    symbols = np.full(n, SymbolType.DECIMAL, dtype=np.uint8)
    symbols[integer & (signed >= -(2**15)) & (signed < 2**15)] = SymbolType.INT16
    symbols[integer & (signed >= -128) & (signed < 128)] = SymbolType.INT8
    symbols[integer & ((signed < -(2**15)) | (signed >= 2**15))] = SymbolType.INT32
    sizes = _SIZE_TABLE[symbols]

    lo = (mantissa & 0xFFFFFFFF).astype(np.uint32)
    mid = (mantissa >> 32).astype(np.uint32)
    hi = np.zeros(n, dtype=np.uint32)
    for i in np.flatnonzero(~small):
        big = int(parts["digits"][int(i)])
        lo[i], mid[i], hi[i] = big & 0xFFFFFFFF, (big >> 32) & 0xFFFFFFFF, big >> 64
    flags = (scale.astype(np.uint32) << 16) | np.where(neg & (length > 0), 0x80000000, 0).astype(np.uint32)
    payload = np.stack([lo, mid, hi, flags], axis=1).astype("<u4").view(np.uint8)

    integer_payload = np.zeros((n, 16), dtype=np.uint8)
    integer_payload[:, :4] = signed.astype("<i4").view(np.uint8).reshape(n, 4)
    payload = np.where(integer[:, None], integer_payload, payload)
    return symbols, sizes, payload


def encode_vocabulary(vocabulary: pl.DataFrame) -> bytes:
    """Encodes every vocabulary row as a cell."""
    count = vocabulary.height
    types = vocabulary["type"].to_numpy()
    values = vocabulary["value"]

    known = np.isin(types, [CellType.NUMBER, CellType.TEXT, CellType.BOOLEAN, CellType.NOTHING, CellType.ERROR])
    if not known.all():
        raise ValueError(f"Unknown cell type: {types[~known][0]}")

    symbols = np.zeros(count, dtype=np.uint8)  # SymbolType.NOTHING
    sizes = np.ones(count, dtype=np.int64)

    bool_rows = np.flatnonzero(types == CellType.BOOLEAN)
    symbols[bool_rows] = np.where(
        (values.gather(bool_rows) == "TRUE").to_numpy(), SymbolType.BOOL_TRUE, SymbolType.BOOL_FALSE
    )

    number_rows = np.flatnonzero(types == CellType.NUMBER)
    number_symbols, number_sizes, number_payload = _encode_numbers(values.gather(number_rows))
    symbols[number_rows] = number_symbols
    sizes[number_rows] = number_sizes

    text_rows = np.flatnonzero((types == CellType.TEXT) | (types == CellType.ERROR))
    texts = values.gather(text_rows).fill_null("")
    text_lengths = texts.str.len_bytes().to_numpy().astype(np.int64)
    is_error = types[text_rows] == CellType.ERROR
    symbols[text_rows] = np.where(is_error, SymbolType.ERROR, SymbolType.TEXT)
    varint_sizes = 1 + sum((text_lengths >= 1 << (7 * j)).astype(np.int64) for j in range(1, 5))
    # Error cells hold an unused int32 error code (zero) before the text.
    text_starts = 1 + 4 * is_error + varint_sizes
    sizes[text_rows] = text_starts + text_lengths

    offsets = np.cumsum(sizes) - sizes
    out = np.zeros(int(sizes.sum()), dtype=np.uint8)
    out[offsets] = symbols

    number_offsets = offsets[number_rows] + 1
    for j in range(16):
        mask = number_sizes - 1 > j
        out[number_offsets[mask] + j] = number_payload[mask, j]

    text_offsets = offsets[text_rows]
    varint_offsets = text_offsets + 1 + 4 * is_error
    for j in range(5):
        mask = varint_sizes > j
        byte = (text_lengths[mask] >> (7 * j)) & 0x7F
        out[varint_offsets[mask] + j] = np.where(varint_sizes[mask] - 1 > j, byte | 0x80, byte)

    source_starts = np.cumsum(text_lengths) - text_lengths
    destination_starts = text_offsets + text_starts
    for start in range(0, len(text_rows), _TEXT_CHUNK_ROWS):
        end = min(start + _TEXT_CHUNK_ROWS, len(text_rows))
        data = texts.slice(start, end - start).str.join("").item().encode("utf-8")
        if not data:
            continue
        lengths = text_lengths[start:end]
        shift = np.repeat(destination_starts[start:end] - source_starts[start:end], lengths)
        out[np.arange(source_starts[start], source_starts[start] + len(data)) + shift] = np.frombuffer(data, np.uint8)

    return out.tobytes()
