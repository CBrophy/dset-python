"""Bit-compressed vectors of vocabulary indexes, least significant bit first."""

import numpy as np

# Rows processed at a time. A multiple of 8, so every chunk starts on a byte boundary.
_CHUNK_ROWS = 1 << 18


def _index_dtype(bit_width: int) -> type:
    if bit_width <= 8:
        return np.uint8
    if bit_width <= 16:
        return np.uint16
    return np.uint32


def unpack_vector(data: bytes, bit_width: int, length: int) -> np.ndarray:
    """Decodes length indexes of bit_width bits each."""
    dtype = _index_dtype(bit_width)
    if bit_width == 0 or length == 0:
        return np.zeros(length, dtype=dtype)

    packed = np.frombuffer(data, dtype=np.uint8)
    weights = (np.uint32(1) << np.arange(bit_width, dtype=np.uint32)).astype(dtype)
    result = np.empty(length, dtype=dtype)
    for start in range(0, length, _CHUNK_ROWS):
        rows = min(_CHUNK_ROWS, length - start)
        first_byte = start * bit_width // 8
        byte_count = (rows * bit_width + 7) // 8
        bits = np.unpackbits(packed[first_byte : first_byte + byte_count], bitorder="little")
        result[start : start + rows] = bits[: rows * bit_width].reshape(rows, bit_width).astype(dtype) @ weights
    return result


def pack_vector(indexes: np.ndarray, bit_width: int) -> bytes:
    """Encodes indexes using bit_width bits each."""
    if bit_width == 0:
        return b""

    shifts = np.arange(bit_width, dtype=np.uint32)
    chunks = []
    for start in range(0, len(indexes), _CHUNK_ROWS):
        chunk = indexes[start : start + _CHUNK_ROWS].astype(np.uint32)
        bits = ((chunk[:, None] >> shifts) & 1).astype(np.uint8)
        chunks.append(np.packbits(bits.ravel(), bitorder="little").tobytes())
    return b"".join(chunks)
