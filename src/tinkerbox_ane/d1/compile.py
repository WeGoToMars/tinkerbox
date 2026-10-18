"""Compile d1 fused AOT deployment bundles for Apple Neural Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinkerbox_ane.coreai import compile_asset
from tinkerbox_ane.d1 import DEFAULT_ASSET, SOURCE
from tinkerbox_ane.d1.bundle import read_bundle
from tinkerbox_ane.d1.host import device_architecture, validate_architecture


def compile_model(
    source: Path = SOURCE,
    out: Path = DEFAULT_ASSET,
    architecture: str | None = None,
    force: bool = False,
    allow_unsupported: bool = False,
) -> Path:
    """Compile d1, then write its deployment manifest after verification."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("keep source exports and deployment bundles in separate directories")
    spec = read_bundle(source)
    src = source / spec["asset"]
    if src.suffix != ".aimodel":
        raise ValueError("compile requires a source .aimodel bundle")
    if architecture is None:
        architecture = device_architecture()
    if (not architecture or Path(architecture).name != architecture
            or "\\" in architecture or architecture in {".", ".."}):
        raise ValueError(f"invalid architecture: {architecture!r}")
    if not allow_unsupported:
        validate_architecture(architecture)
    target = out / f"fused.{architecture}.aimodelc"
    compile_asset(src, target, architecture=architecture, min_deployment_version="27.0",
                  expected_functions=(f"c{capacity}" for capacity in spec["tiers"]), force=force)
    spec.update(asset=target.name, architecture=architecture, min_os="27.0", deployment="aot-fused")
    vision_src_name = spec.get("vision_asset") or ("vision.aimodel" if (source / "vision.aimodel").is_dir() else None)
    if vision_src_name and (source / vision_src_name).exists():
        vision_src = source / vision_src_name
        vision_target = out / f"vision.{architecture}.aimodelc"
        compile_asset(vision_src, vision_target, architecture=architecture, min_deployment_version="27.0",
                      expected_functions=("main",), force=force)
        spec.update(vision_asset=vision_target.name)
    manifest = out / ".model.json"
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    manifest.replace(out / "model.json")
    return out



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--architecture", "-a", type=str, default=None,
                        help="Target architecture (defaults to current device)")
    parser.add_argument("--force", action="store_true", help="Recompile an existing target")
    parser.add_argument("--allow-unsupported", action="store_true",
                        help="Attempt compilation for an unsupported architecture")
    args = parser.parse_args()
    print(compile_model(args.source, args.out, architecture=args.architecture, force=args.force,
                        allow_unsupported=args.allow_unsupported))


if __name__ == "__main__":
    main()
