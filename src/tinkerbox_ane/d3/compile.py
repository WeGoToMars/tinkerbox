"""Compile d3 staged AOT deployment bundles for Apple Neural Engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinkerbox_ane.bundle import compile_bundle
from tinkerbox_ane.d3 import KIND


def compile_model(
    source: Path,
    out: Path,
    architecture: str | None = None,
    force: bool = False,
    allow_unsupported: bool = False,
) -> Path:
    """Compile a d3 export, then write its deployment manifest after verification."""
    return compile_bundle(source, out, KIND, architecture, force=force, allow_unsupported=allow_unsupported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
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
