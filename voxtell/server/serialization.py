"""Compact wire format for numpy arrays sent between the VoxTell client and server.

An array is framed with a small self-describing header (dtype + shape) and its raw
buffer is blosc2-compressed in <=1 GiB chunks with a dtype-aware typesize, so the
SHUFFLE filter groups like-significance bytes - which shrinks int16/float32 volumes
much better than byte-wise compression. Masks and medical volumes are mostly
homogeneous, so payloads stay tiny versus raw bytes or JSON/base64.

This module is deliberately tiny and dependency-light (numpy + blosc2) so an identical
copy can live in the torch-free napari client. Keep the two copies in sync.

Wire-format approach inspired by MIC-DKFZ's nnInteractive (Apache-2.0).
"""

from __future__ import annotations

import blosc2
import numpy as np

# Format identifier + version; bump if the framing below ever changes.
_MAGIC = b"VXA2"

# blosc2.compress accepts at most (2 GiB - 1) bytes per call, so large volumes are
# split into 1 GiB chunks (a multiple of any 1/2/4/8-byte dtype) compressed independently.
_CHUNK_SIZE = 1 << 30


def pack_array(arr: np.ndarray) -> bytes:
    """Serialize ``arr`` to the framed, dtype-aware blosc2-compressed byte format."""
    arr = np.ascontiguousarray(arr)
    dtype_str = arr.dtype.str.encode("ascii")  # e.g. b"<i2", preserves endianness
    itemsize = arr.dtype.itemsize
    raw = arr.tobytes(order="C")

    out = bytearray(_MAGIC)
    out += len(dtype_str).to_bytes(1, "little")
    out += dtype_str
    out += len(arr.shape).to_bytes(1, "little")
    for dim in arr.shape:
        out += int(dim).to_bytes(8, "little")
    out += len(raw).to_bytes(8, "little")  # total uncompressed length (for validation)
    for start in range(0, len(raw), _CHUNK_SIZE):
        chunk = raw[start:start + _CHUNK_SIZE]
        compressed = blosc2.compress(chunk, typesize=itemsize)
        out += len(chunk).to_bytes(8, "little")
        out += len(compressed).to_bytes(8, "little")
        out += compressed
    return bytes(out)


def unpack_array(data: bytes) -> np.ndarray:
    """Inverse of :func:`pack_array`."""
    view = memoryview(data)
    if bytes(view[:4]) != _MAGIC:
        raise ValueError("Not a VoxTell array payload (bad magic bytes).")

    pos = 4
    dtype_len = view[pos]
    pos += 1
    dtype_str = bytes(view[pos:pos + dtype_len]).decode("ascii")
    pos += dtype_len
    ndim = view[pos]
    pos += 1
    shape = []
    for _ in range(ndim):
        shape.append(int.from_bytes(view[pos:pos + 8], "little"))
        pos += 8
    total = int.from_bytes(view[pos:pos + 8], "little")
    pos += 8

    raw = bytearray()
    while pos < len(view):
        uncompressed_len = int.from_bytes(view[pos:pos + 8], "little")
        pos += 8
        compressed_len = int.from_bytes(view[pos:pos + 8], "little")
        pos += 8
        raw += blosc2.decompress(view[pos:pos + compressed_len])
        pos += compressed_len

    if len(raw) != total:
        raise ValueError(
            f"Corrupt array payload: expected {total} bytes, got {len(raw)}."
        )
    # frombuffer on the (mutable) bytearray yields a writable array without a copy.
    return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(tuple(shape))
