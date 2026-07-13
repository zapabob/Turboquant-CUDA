from __future__ import annotations

import math
from pathlib import Path
import struct
from typing import Any


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TENSOR_F32 = 0


def _write_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def _value_type(value: object) -> int:
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL
    if isinstance(value, int):
        if value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError("GGUF fixture integers must fit uint64")
        return GGUF_TYPE_UINT64 if value >= 0x1_0000_0000 else GGUF_TYPE_UINT32
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("GGUF fixture floats must be finite")
        return GGUF_TYPE_FLOAT32
    if isinstance(value, str):
        return GGUF_TYPE_STRING
    if isinstance(value, list):
        return GGUF_TYPE_ARRAY
    raise TypeError(f"unsupported GGUF metadata value type: {type(value).__name__}")


def _write_scalar(buffer: bytearray, value_type: int, value: object) -> None:
    if value_type == GGUF_TYPE_BOOL:
        buffer.extend(struct.pack("<?", bool(value)))
    elif value_type == GGUF_TYPE_UINT32:
        if not isinstance(value, int):
            raise TypeError("GGUF uint32 value must be int")
        buffer.extend(struct.pack("<I", value))
    elif value_type == GGUF_TYPE_UINT64:
        if not isinstance(value, int):
            raise TypeError("GGUF uint64 value must be int")
        buffer.extend(struct.pack("<Q", value))
    elif value_type == GGUF_TYPE_FLOAT32:
        if not isinstance(value, float):
            raise TypeError("GGUF float32 value must be float")
        buffer.extend(struct.pack("<f", value))
    elif value_type == GGUF_TYPE_STRING:
        _write_string(buffer, str(value))
    else:
        raise TypeError(f"unsupported GGUF scalar type {value_type}")


def _write_value(buffer: bytearray, value: object) -> None:
    value_type = _value_type(value)
    buffer.extend(struct.pack("<I", value_type))
    if value_type != GGUF_TYPE_ARRAY:
        _write_scalar(buffer, value_type, value)
        return
    if not isinstance(value, list):
        raise TypeError("GGUF array value must be list")
    items = value
    item_type = _value_type(items[0]) if items else GGUF_TYPE_STRING
    if item_type == GGUF_TYPE_ARRAY or any(
        _value_type(item) != item_type for item in items
    ):
        raise TypeError("GGUF metadata arrays must be flat and homogeneous")
    buffer.extend(struct.pack("<I", item_type))
    buffer.extend(struct.pack("<Q", len(items)))
    for item in items:
        _write_scalar(buffer, item_type, item)


def _align(value: int) -> int:
    return ((value + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT


def write_fixture_gguf(
    *,
    path: Path,
    metadata: dict[str, object],
    tensors: dict[str, dict[str, Any]],
) -> None:
    buffer = bytearray()
    ordered_tensors = sorted(tensors.items())
    buffer.extend(
        struct.pack(
            "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(ordered_tensors), len(metadata)
        )
    )
    for key, value in sorted(metadata.items()):
        if not key:
            raise ValueError("GGUF metadata keys must be non-empty")
        _write_string(buffer, key)
        _write_value(buffer, value)

    offsets: dict[str, int] = {}
    tensor_data: dict[str, list[float]] = {}
    current_offset = 0
    for name, tensor in ordered_tensors:
        if not name:
            raise ValueError("GGUF tensor names must be non-empty")
        if tensor.get("dtype") != "f32":
            raise ValueError(f"fixture tensor {name} must declare dtype f32")
        shape = [int(value) for value in tensor["shape"]]
        data = [float(value) for value in tensor["data"]]
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"tensor {name} shape must contain positive dimensions")
        if math.prod(shape) != len(data):
            raise ValueError(f"tensor {name} data length does not match shape")
        if not all(math.isfinite(value) for value in data):
            raise ValueError(f"tensor {name} data must be finite")
        tensor_data[name] = data
        offsets[name] = current_offset
        _write_string(buffer, name)
        buffer.extend(struct.pack("<I", len(shape)))
        for dimension in shape:
            buffer.extend(struct.pack("<Q", dimension))
        buffer.extend(struct.pack("<IQ", GGUF_TENSOR_F32, current_offset))
        current_offset = _align(current_offset + len(data) * 4)

    buffer.extend(b"\x00" * (_align(len(buffer)) - len(buffer)))
    data_base = len(buffer)
    for name, _tensor in ordered_tensors:
        expected_position = data_base + offsets[name]
        buffer.extend(b"\x00" * (expected_position - len(buffer)))
        data = tensor_data[name]
        buffer.extend(struct.pack(f"<{len(data)}f", *data))
    path.write_bytes(buffer)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("truncated GGUF")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def string(self) -> str:
        (length,) = self.unpack("<Q")
        return self.take(length).decode("utf-8")

    def scalar(self, value_type: int) -> object:
        if value_type == GGUF_TYPE_BOOL:
            return self.unpack("<?")[0]
        if value_type == GGUF_TYPE_UINT32:
            return self.unpack("<I")[0]
        if value_type == GGUF_TYPE_UINT64:
            return self.unpack("<Q")[0]
        if value_type == GGUF_TYPE_FLOAT32:
            return self.unpack("<f")[0]
        if value_type == GGUF_TYPE_STRING:
            return self.string()
        raise ValueError(f"unsupported GGUF metadata type {value_type}")

    def value(self) -> object:
        (value_type,) = self.unpack("<I")
        if value_type != GGUF_TYPE_ARRAY:
            return self.scalar(value_type)
        item_type, length = self.unpack("<IQ")
        return [self.scalar(item_type) for _ in range(length)]


def read_fixture_gguf(path: Path) -> dict[str, Any]:
    reader = _Reader(path.read_bytes())
    magic, version, tensor_count, metadata_count = reader.unpack("<IIQQ")
    if magic != GGUF_MAGIC or version != GGUF_VERSION:
        raise ValueError("unsupported GGUF header")
    metadata = {reader.string(): reader.value() for _ in range(metadata_count)}
    if len(metadata) != metadata_count:
        raise ValueError("GGUF contains duplicate metadata keys")
    tensor_infos: dict[str, dict[str, Any]] = {}
    for _ in range(tensor_count):
        name = reader.string()
        (dimension_count,) = reader.unpack("<I")
        if dimension_count == 0:
            raise ValueError(f"fixture tensor {name} must have at least one dimension")
        shape = [reader.unpack("<Q")[0] for _ in range(dimension_count)]
        if not name or any(dimension == 0 for dimension in shape):
            raise ValueError("GGUF tensor names and dimensions must be non-empty")
        tensor_type, tensor_offset = reader.unpack("<IQ")
        if tensor_type != GGUF_TENSOR_F32:
            raise ValueError(f"fixture tensor {name} must be F32")
        if name in tensor_infos:
            raise ValueError(f"GGUF contains duplicate tensor name {name}")
        if tensor_offset % GGUF_ALIGNMENT != 0:
            raise ValueError(f"fixture tensor {name} offset is not aligned")
        tensor_infos[name] = {"dtype": "f32", "shape": shape, "offset": tensor_offset}
    data_base = _align(reader.offset)
    tensors: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for name, info in sorted(
        tensor_infos.items(), key=lambda item: int(item[1]["offset"])
    ):
        count = math.prod(info["shape"])
        start = data_base + int(info["offset"])
        end = start + count * 4
        if end > len(reader.data):
            raise ValueError(f"truncated GGUF tensor {name}")
        if spans and start < spans[-1][1]:
            raise ValueError(f"fixture tensor {name} overlaps {spans[-1][2]}")
        spans.append((start, end, name))
        data = list(struct.unpack(f"<{count}f", reader.data[start:end]))
        tensors[name] = {"dtype": "f32", "shape": info["shape"], "data": data}
    if spans and spans[-1][1] != len(reader.data):
        raise ValueError("GGUF contains trailing or unreferenced tensor bytes")
    return {"metadata": metadata, "tensors": tensors}


__all__ = ["read_fixture_gguf", "write_fixture_gguf"]
