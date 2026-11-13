"""Compile an exported bundle for one chip with the shared watchdog and full ANE placement verification."""

import argparse
import json
from pathlib import Path

from tinkerbox_ane.bundle import compile_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="export directory containing model.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", help="target, such as h16g for standard M4 (defaults to this Mac)")
    parser.add_argument("--force", action="store_true", help="recompile an existing target")
    args = parser.parse_args(argv)
    kind = json.loads((args.source / "model.json").read_text())["kind"]
    print(compile_bundle(args.source, args.out, kind, args.architecture, args.force))


if __name__ == "__main__":
    main()
