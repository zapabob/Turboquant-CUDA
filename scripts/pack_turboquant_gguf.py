from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant.gguf_profiles import (
    GGUF_HYPURA_COMPAT_AUTO,
    GGUF_TURBOQUANT_EXACT_PROFILE,
    build_paper_gguf_profile,
    build_so8_triality_vector_gguf_profile,
    infer_gguf_attention_head_dim,
    infer_gguf_block_count,
    package_turboquant_gguf,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package an existing GGUF into a single TurboQuant-aware GGUF with embedded runtime profiles."
    )
    parser.add_argument("--input-gguf", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument(
        "--profiles",
        default="paper,so8_triality_vector",
        help="Comma-separated embedded profile names. Supported: paper, so8_triality_vector",
    )
    parser.add_argument(
        "--default-profile",
        default=GGUF_TURBOQUANT_EXACT_PROFILE,
        help="Runtime default profile name. Best practice keeps this as 'exact'.",
    )
    parser.add_argument(
        "--hypura-compatible-profile",
        default=GGUF_HYPURA_COMPAT_AUTO,
        help=(
            "Optional Hypura/Kobold-compatible bridge profile. "
            "Use 'auto' to emit research bridge metadata when a supported profile is embedded, "
            "'off' to disable it, or pass an explicit embedded profile name."
        ),
    )
    parser.add_argument(
        "--bits",
        type=float,
        default=3.5,
        help="Concrete bits/channel setting baked into the embedded runtime profiles.",
    )
    parser.add_argument(
        "--rotation-dir",
        default="artifacts/research_extension/triality_full_train/rotations",
        help="Directory containing learned triality rotation .pt artifacts.",
    )
    parser.add_argument(
        "--paper-rotation-seed",
        type=int,
        default=0,
        help="Rotation seed recorded for the paper-faithful random-Haar profile.",
    )
    parser.add_argument(
        "--paper-qjl-seed",
        type=int,
        default=1,
        help="QJL seed recorded for the paper-faithful random-Haar profile.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite --output-gguf if it already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_gguf = Path(args.input_gguf)
    output_gguf = Path(args.output_gguf)
    requested_profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not requested_profiles:
        raise ValueError("At least one profile must be requested via --profiles")
    if output_gguf.exists():
        if not args.force:
            raise FileExistsError(f"Output GGUF already exists: {output_gguf}")
        output_gguf.unlink()

    head_dim = infer_gguf_attention_head_dim(input_gguf)
    block_count = infer_gguf_block_count(input_gguf)

    profiles = []
    for profile_name in requested_profiles:
        if profile_name == "paper":
            profiles.append(
                build_paper_gguf_profile(
                    bits_total=args.bits,
                    head_dim=head_dim,
                    rotation_seed=args.paper_rotation_seed,
                    qjl_seed=args.paper_qjl_seed,
                )
            )
            continue
        if profile_name == "so8_triality_vector":
            profiles.append(
                build_so8_triality_vector_gguf_profile(
                    rotation_dir=Path(args.rotation_dir),
                    bits_total=args.bits,
                    expected_head_dim=head_dim,
                    expected_block_count=block_count,
                )
            )
            continue
        raise ValueError(f"Unsupported embedded GGUF profile: {profile_name}")

    manifest = package_turboquant_gguf(
        source_path=input_gguf,
        output_path=output_gguf,
        profiles=profiles,
        default_profile=args.default_profile,
        hypura_compatibility_profile=args.hypura_compatible_profile,
    )
    embedded_profiles = ", ".join(manifest.profiles.keys())
    print(f"packaged_gguf={output_gguf}")
    print(f"default_profile={manifest.default_profile}")
    print(f"embedded_profiles={embedded_profiles}")
    print(f"base_architecture={manifest.base_architecture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
