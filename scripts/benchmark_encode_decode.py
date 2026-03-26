from __future__ import annotations

from pathlib import Path
import time
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from turboquant.io_utils import ensure_dir
from turboquant.kv_codec import KVCodec, KVCodecConfig


def main() -> int:
    output_path = ensure_dir(Path("artifacts/reports")) / "benchmark.md"
    rows: list[str] = ["# Encode/Decode Benchmark", ""]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    for seq_len in (32, 128, 256):
        keys = torch.randn((1, 8, seq_len, 128), generator=generator)
        values = torch.randn((1, 8, seq_len, 128), generator=generator)
        codec = KVCodec(KVCodecConfig(head_dim=128, key_bits=3, value_bits=3))

        start = time.perf_counter()
        encoded_keys = codec.encode_keys(keys)
        encoded_values = codec.encode_values(values)
        encode_seconds = time.perf_counter() - start

        start = time.perf_counter()
        _ = codec.decode_keys(encoded_keys)
        _ = codec.decode_values(encoded_values)
        decode_seconds = time.perf_counter() - start

        rows.append(f"- seq_len={seq_len}: encode={encode_seconds:.6f}s decode={decode_seconds:.6f}s")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
